# **ASTON**: Activity Suffix predicTiOn based on eNcoder-decoder

> Implementation of a deep learning architecture for activity suffix prediction based on an encoder-decoder and GRNNs.

### Environment creation

* Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html "Miniconda") in your system.
* ``conda create -n ASTON python=3.9 pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia``
* ``conda activate ASTON``
* ``python -m pip install pm4py==2.2.29 d2l scikit-learn pyyaml stringdist==1.0.9 matplotlib==3.5.1 pandas``
* ``python -m pip install networkx==2.8.4 lion-pytorch``

### Experimentation execution

* An experimentation run can be executed as follows: ``python aston.py --dataset DATASET --execution_id ID --num_epochs 5 --num_folds 5 --fold_num 0 --postprocessing beam_length_normalized --train``

Example: ``python aston.py --dataset BPI_Challenge_2012_A --execution_id trial --num_epochs 5 --num_folds 5 --fold_num 0 --postprocessing beam_length_normalized --train``
