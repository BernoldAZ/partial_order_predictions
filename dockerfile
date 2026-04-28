# ─── Base image ───────────────────────────────────────────────────────────────
# pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime  →  Python 3.11, PyTorch 2.4.0
#
# Why cuda12.4 on a driver 545.23 / CUDA 12.3 host?
#   • CUDA minor-version compatibility: containers built against CUDA 12.x run on
#     any driver that supports CUDA 12.x (545.xx ships with CUDA 12.3, which fully
#     satisfies the 12.4 runtime inside the container via backward compatibility).
#   • No official pytorch/pytorch image exists for cuda12.3; 12.4 is the nearest.
#
# Why NOT cuda12.1 (previous image)?
#   • Stayed on Python 3.10, blocking pandas 3.x, scipy 1.16+, scikit-learn 1.7+,
#     networkx 3.4+, and numpy 2.x.
#   • cuda12.4 + Python 3.11 lifts all those caps.
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /workspace

# Copy requirements
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ─── PyTorch Geometric ────────────────────────────────────────────────────────
# PyG publishes wheels for PyTorch 2.4.x against cu118, cu121, and cu124.
# cu121 is the closest available index to CUDA 12.3 (minor-version compatible).
RUN pip install --no-cache-dir \
    pyg-lib \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

# Install torch_geometric itself
RUN pip install torch-geometric

# Expose Jupyter port
EXPOSE 8887

# Launch Jupyter
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8887", "--no-browser", "--allow-root", "--NotebookApp.token=''"]

# docker build -t ml-jupyter-gpu .
# docker run -it --rm --gpus all -p 8887:8887 -v $(pwd):/workspace ml-jupyter-gpu
