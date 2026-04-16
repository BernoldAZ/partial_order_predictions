# Benchmarking Distributional Similarity Between Activities (Certain + Uncertain Event Data)

This repository contains code, (processed) data, and results for **two closely related papers**:

- **ICPM 2025 (certain event data)**: Henrik Kirchmann, Stephan A. Fahrenkrog-Petersen, Xixi Lu, Matthias Weidlich, *Let’s Simply Count: Quantifying Distributional Similarity between Activities in Event Data*, `10.1109/ICPM66919.2025.11220676`
- **Journal extension (certain + uncertain event data)**: Henrik Kirchmann, Stephan A. Fahrenkrog-Petersen, Xixi Lu, Matthias Weidlich, *Distributional Similarity Between Activities in Certain and Uncertain Event Data* (Process Science collection “Best Process Science Conference Papers 2025”, under review)

If you want to reproduce the paper plots for the journal extension, you can do so directly from the CSVs/XES files shipped in this repository (see **Journal extension** below).

## Quick navigation

- **Track A (ICPM 2025 / certain event data)**: see [ICPM 2025: Certain Event Data](#icpm-2025-certain-event-data)
- **Track B (Journal extension / uncertain event data)**: see [Journal extension: Uncertain Event Data](#journal-extension-uncertain-event-data)
- **Where to get the uncertain IKEA ASM logs**: see [IKEA ASM uncertain event logs](#ikea-asm-uncertain-event-logs)

## Installation

Use **Python 3.11**.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Optional: Autoencoder GPU setup

The autoencoder baseline requires a compatible CUDA/cuDNN setup. If you do not have a suitable GPU environment, do **not** run the autoencoder baseline.

## Repository layout (at a glance)

- **Certain (ICPM 2025)**
  - Methods: `distances/activity_distances/`
  - Benchmarks: `evaluation/evaluation_of_activity_distances/`
  - Results: `results/activity_distances/`
- **Uncertain (Journal extension)**
  - Methods: `distances/uncertain_activity_distances/`
  - Scripts (intrinsic/next-activity/runtimes/plots): `uncertain_scripts/`
  - Utilities: `uncertain_utils/`
  - IKEA ASM-derived uncertain logs used in the paper: `uncertain_event_data/ikea_asm/`
  - Uncertain intrinsic plots: `results/activity_distances/intrinsic_uncertain_summary/paper_plots/`
  - Uncertain next-activity plots: `results/next_activity_prediction_uncertain_evermann/paper_plots/`

## ICPM 2025: Certain Event Data

This track corresponds to the ICPM 2025 paper (*Let’s Simply Count*). Below we keep the original, detailed repository documentation for reproducing the **certain event data** experiments.

## Setup

Make sure you are using **Python 3.11** to run the scripts.

Install all required packages using the [`requirements.txt`](requirements.txt) file.

**cuDNN & CUDA Setup (Required only if using the Autoencoder approach):**\
   Ensure that a compatible GPU, CUDA, and cuDNN are properly installed and configured.\
If this setup is not available, do NOT include 'Autoencoder' in the activity_distance_functions list.\
 (See "Similarity / Embedding Methods" under "Configurable Settings" for more details.) \
 Run the following command in your terminal:
  ```bash
  pip install torch==2.1.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
  ```

---

## 📁 Benchmark Overview
We provide a comprehensive benchmarking framework to evaluate distributional similarity between activities in event data for process mining based on:

| Benchmark                                              | Description |
|--------------------------------------------------------|-------------|
| [Intrinsic Evaluation](#test_tube-intrinsic-evaluation)       | Assesses the quality of distributional similarity between activities using synthetic ground truth logs. |
| [Next Activity Prediction](#fast_forward-next-activity-prediction) | Benchmarks the embeddings in the task of next activity prediction. |
| [Runtime Analysis](#hourglass_flowing_sand-runtime-analysis)              | Analyzes runtime performance across methods. |


## ⚙️ Methods Compared in the Benchmark

The benchmarking framework evaluates both **newly proposed methods** and **existing methods** for computing distributional similarity between activities in event data.

The implementation for each method can be found in [`distances/activity_distances`](./distances/activity_distances) folder.


### 🆕 New Count-based Methods

Our proposed methods derive activity embeddings and distributional similarity from event logs using simple and interpretable **count-based approaches**. Each variant is defined by its matrix type, context interpretation, and optional postprocessing.

**📁 Where to find them:**

- **Activity-activity co-occurrence matrices:**  
  [`distances/activity_distances/activity_activity_co_occurence/activity_activity_co_occurrence.py`](./distances/activity_distances/activity_activity_co_occurence/activity_activity_co_occurrence.py)
- **Activity-context frequency matrices:**  
  [`distances/activity_distances/activity_context_frequency/activity_contex_frequency.py`](./distances/activity_distances/activity_context_frequency/activity_contex_frequency.py)
- **PMI & PPMI postprocessing:**  
  [`distances/activity_distances/pmi/pmi.py`](./distances/activity_distances/pmi/pmi.py)  



###  📖 Existing Methods

These baselines are implemented or re-used from prior work.

| Method                        | Description                                         | Paper                                                                                                                                                                     | Original Implementation | Name in this Repo                          | File Location in this Repo                                                                                                                                                                                                                           |
|------------------------------|-----------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------|--------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Substitution Scores**      | Co-occurrence-based similarity using log-ratios     | Context Aware Trace Clustering: Towards Improving Process Mining Results; Bose & van der Aalst (2009), [DOI](https://doi.org/10.1137/1.9781611972795.35)                  | Re-implemented from authors' Java code | Bose 2009 Substitution Scores              | [`distances/activity_distances/bose_2009_context_aware_trace_clustering/algorithm.py`](./distances/activity_distances/bose_2009_context_aware_trace_clustering/algorithm.py)                                                                         |
| **act2vec** (CBOW)           | Neural embeddings adapted from Word2Vec (CBOW)      | act2vec, trace2vec, log2vec, and model2vec: Representation Learning for Business Processes; De Koninck et al. (2018), [DOI](https://doi.org/10.1007/978-3-319-98648-7_18) | [processmining.be/replearn](https://processmining.be/replearn/) | De Koninck 2018 act2vec CBOW               | [`distances/activity_distances/de_koninck_2018_act2vec/algorithm.py`](./distances/activity_distances/de_koninck_2018_act2vec/algorithm.py)                                                                                                           |
| **act2vec** (Skip-gram)      | Neural embeddings adapted from Word2Vec (Skip-gram) | act2vec, trace2vec, log2vec, and model2vec: Representation Learning for Business Processes; De Koninck et al. (2018), [DOI](https://doi.org/10.1007/978-3-319-98648-7_18) | [processmining.be/replearn](https://processmining.be/replearn/) | De Koninck 2018 act2vec skip-gram          | [`distances/activity_distances/de_koninck_2018_act2vec/algorithm.py`](./distances/activity_distances/de_koninck_2018_act2vec/algorithm.py)                                                                                                           |
| **Embedding Process Structure** | Feature vectors based on Petri nets and process structure | Embedding Process Structure in Activities for Process Mapping and Comparison; Chiorrini et al. (2022)  [DOI](https://doi.org/10.1007/978-3-031-15743-1_12)                | [GitHub Repo](https://github.com/KDMG/Embedding-Structure-in-Activities) | Chiorrini 2022 Embedding Process Structure | [`distances/activity_distances/chiorrini_2022_embedding_process_structure/embedding_process_structure.py`](./distances/activity_distances/chiorrini_2022_embedding_process_structure/embedding_process_structure.py)                                 |
| **Autoencoder**              | Context-based representation learning via autoencoders | Learning Context-Based Representations of Events in Complex Processes; Gamallo-Fernandez et al. (2023), [DOI](https://doi.org/10.1109/ICWS60048.2023.00041)               | [GitLab Source](https://gitlab.citius.gal/pedro.gamallo/PM_Embeddings/-/blob/master/src/embeddings_generator/aerac.py?ref_type=heads) | Gamallo Fernandez 2023 Context Based       | [`distances/activity_distances/gamallo_fernandez_2023_context_based_representations/src/embeddings_generator/main_new.py`](./distances/activity_distances/gamallo_fernandez_2023_context_based_representations/src/embeddings_generator/main_new.py) |

---

#### 📄 Existing Methods – Implementation Details

- **Substitution Scores:**\
  Re-implemented in Python from the Java code shared by the authors.

- **act2vec:**\
  Used the default configurations provided by the authors from the official implementation.

- **Embedding Process Structure:**\
 We re-implemented the computation of the path length, parallelism path length, and parallelism features.\
 The original implementation for the path lengths feature had a slow runtime that was not feasible for our intrinsic evaluation, so we optimized it, reducing computation time for some logs from several minutes to mere milliseconds. The original computation iterated over all transitions of a Petri net, and within each iteration, it further iterated over all paths leading to the transition. Within this second iteration, there was yet another loop that iterated over all nodes within each path.\
 In contrast, our implementation computes the path length feature using a bottom-up approach on the process tree, requiring only a single iteration over all nodes in the tree. \
 Additionally, the parallelism and parallelism path length features were initially only compatible with the authors' Petri net files from their GitHub repository.

- **Autoencoder:**\
  The original method used 5-fold cross-validation (64% train, 16% val, 20% test).  
  After completing all five folds, the embedding that performs best on the test fold is selected.\
  To reduce the considerable runtime of the full 5-fold procedure, we employed a simpler 80\%/20\% train/validation split. The authors do not specify an exact embedding dimension; instead, they match it to the requirements of the prediction model. Following their observation that larger embeddings improve performance, we fix the embedding dimension to 128. \
  Due to the long runtime, we tested the autoencoder only at a window size of 3, following the original study’s finding that this setting yields the best predictive performance.


---

## :test_tube: Intrinsic Evaluation

### 🔧 How to Run the Script

To execute the benchmark evaluation, simply run the script [`evaluation/evaluation_of_activity_distances/intrinsic_evaluation/intrinsic_evaluation.py`](evaluation/evaluation_of_activity_distances/intrinsic_evaluation/evaluation_activity_distance_intrinsic.py)

---

### ⚙️ Configurable Settings

You can customize the experiment by editing the `if __name__ == '__main__':` section of the script. Here's a breakdown of what you can change:

#### Similarity / Embedding Methods
These define which methods will be evaluated. They are appended to `activity_distance_functions`. You can **enable/disable methods** by commenting/uncommenting `append()` lines.

Example:
```python
# activity_distance_functions.append("Unit Distance")  # Not used
activity_distance_functions.append("Bose 2009 Substitution Scores")  # Used
```

You can also adjust the **window sizes** that are evaluated via:
```python
window_size_list = [3, 5, 9]
```

---

####  Parameters for Ground Truth Log Creation
These control the synthetic event log creation: For each original log, we create sampling_size many logs for each combination of
how many differnt activities are replaced \{1, 2, ..., \min(|A_L|, r\_min)\} and with how many new activites are each original activty are replaced with \{2, 3, ..., w\}. As described in the paper. 
```python
r_min         = 10     # How many differnt activities are replaced  
w             = 5      # How many new activites are used to replace each original activty with
sampling_size = 5      # How many ground truth logs to generate
load_ground_truth_logs = False 
# If True, load existing ground truth logs if available; otherwise, create and save new ones. 
# If False, always create new logs without loading existing ones.
# If True, if results of the corresponding parameters exist for the evaluated original log and method, load them from evaluation / evaluation_of_activity_distances / intrinsic_evaluation / results, if results file daoes not exist do the experiemnt. 
# To rerun the experiemnts with new event logs to store them, delete the evaluation / evaluation_of_activity_distances / intrinsic_evaluation / results folder.

```

> 💡 When you want to use the ground truth log we used, set `load_ground_truth_logs = True`, make sure to unzip and place the folders of the pre-generated logs from  
> [https://box.hu-berlin.de/d/7a97101239654eae8e6c/](https://box.hu-berlin.de/d/7a97101239654eae8e6c/) into:  
> [`evaluation/evaluation_of_activity_distances/intrinsic_evaluation/newly_created_logs/`](evaluation/evaluation_of_activity_distances/intrinsic_evaluation/newly_created_logs/)\
> **Note:** The unzipped folder requires **24.7 GB** of storage space.\
> If you want to recompute the results with the existing results, delete the folders in [`evaluation/evaluation_of_activity_distances/intrinsic_evaluation/results`](evaluation/evaluation_of_activity_distances/intrinsic_evaluation/newly_created_logs/).



---

#### Event Logs to Evaluate

By default, only the `'Sepsis'` log is evaluated:
```python
log_list.append('Sepsis')
```

To evaluate more logs, uncomment the entries in the multi-line string:
```python
"""
log_list.append('BPIC12')
log_list.append('BPIC13_incidents')
...
"""
```


---

You can download the original event logs from the following location:

[https://box.hu-berlin.de/f/aa5905ab235e444b8ffa/](https://box.hu-berlin.de/f/aa5905ab235e444b8ffa/)

Unpack the folder and place all `.xes.gz` files into the directory: [`event_logs`](./event_logs) 


---
### 💾 Saving Results in Intrinsic Evaluation

When you run the intrinsic evaluation, **the following types of results** are stored:

---

#### 1. Intermediate Results to Save and Reuse 

Evaluation results are **saved per log per method per r and w and s values** as:

```
results/
└── evaluation_of_activity_distances/
    └── intrinsic_evaluation/
        └──results
            └── <log_name>/
                └── <method>/
                    └── r_<r>_w_<w>_s_<sampling_size>.pkl
```

Each `.pkl` file contains a tuple with the metrics:
```
(r, w, diameter, precision@w-1, precision@1, triplet)
```

---

#### 2. CSV Files: Per-sample Detailed Results

In addition to saving, full individual evaluation results, results are stored as CSVs under:

```
results/
└── activity_distances/
    └── intrinsic/
        └── <log_name>/
            └── <log_name>_distfunc_<method>_r<r>_w<w>_samplesize_<sampling_size>.csv
```

Each row in the CSV includes:

- `r`: Number of activities replaced  
- `w`: Number of new activities used for replacement  
- `diameter`: Log embedding diameter  
- `precision@w-1`: Precision at window size - 1  
- `precision@1`: Nearest neighbor accuracy  
- `triplet`: Triplet loss consistency score  

---

#### 3. Aggregated Results (`.pkl` DataFrame)

After evaluating all methods, the script computes the **average values across the results of all r and w ** per method. These are stored in a single aggregated `.pkl` file:

```
results/activity_distances/intrinsic_df_avg/<log_name>/dfavg_r<r>_w<w>_samplesize_<sampling_size>.pkl
```

This file contains a `DataFrame` with one row per method and the following columns:

- `Log Name`  
- `Distance Function`  
- `diameter` (average)  
- `precision@w-1` (average)  
- `precision@1` (average)  
- `triplet` (average)


---
### 📊 Analyzing Results in Intrinsic Evaluation

To analyze and visualize the intrinsic evaluation results across all logs, run:
> [`additional_scripts/calculate_average_intrinsic_results.py`](additional_scripts/calculate_average_intrinsic_results.py)

Before running the analysis script:

1. **Configure target logs**: Modify the `all_logs` list in the script to specify which logs you want to analyze

2. **Verify prerequisite data**: Ensure the aggregated results have been generated by completing the intrinsic evaluation first

3. **Check results location**: Confirm that results are stored in the expected directory:
   ```
   results/activity_distances/intrinsic_df_avg/
   ```

### 🔎 Intrinsic Evaluation Measures 

#### Shared Notation for Intrinsic Evaluation Measures

Let $L$ be an original event log, and $L'$ the modified log as described in *Algorithm 1*. Define:

- **$A_r = A(L) \setminus A(L')$**: the set of activities replaced in $L$
- **$A_{new} = A(L') \setminus A(L)$**: the set of new activities
- **$\psi: A_r \to \mathcal{P}(A_{new})$**: maps each replaced activity to the set of activities that replaced it
- **$\phi: A(L') \to A(L)$**: maps each new activity to the original activity it replaced (identity for others)
- **$d: A(L') \times A(L') \to \mathbb{R}$**: a distance function on activities (low distance reflects high similarity be evaluated method)
- **$\hat{d}: A(L') \times A(L') \to [0, 1]$**: normalized version of $d$, scaled to $[0, 1]$

---

#### Compactness (called diameter in this repo)

This measure captures the average intra-class distance, normalized by the largest observed distance. If distances can be negative (e.g., see Bose et al., 2009), we shift them before normalization.

**Definition:**

```math
I_{comp} = \frac{1}{|A_r|} \sum_{a_r \in A_r} \frac{1}{|\psi(a_r)|^2 - |\psi(a_r)|} \sum_{\substack{a_i, a_j \in \psi(a_r) \\ a_i \neq a_j}} \hat{d}(a_i, a_j)
```
In the paper, the values for $I_{comp}$ are computed as $1 - I_{comp}$, to align with other metrics, ensuring that higher values indicate better performance.

---

#### Nearest Neighbor (called precision@1 in this repo)

This measure counts how often the nearest neighbor of a new activity belongs to the same class.

**Definition:**

```math
I_{nn} = \frac{1}{|A_r|} \sum_{a_r \in A_r} \frac{1}{|\psi(a_r)|} \sum_{a_{new} \in \psi(a_r)} \mathbf{1}(a_{new})
```

Where:

```math
\mathbf{1}(a_{new}) =
\begin{cases}
1, & \phi(a_{new}) = \phi(\arg\min_{a \neq a_{new}} d(a_{new}, a)) \\
0, & \text{otherwise}
\end{cases}
```

---

#### Precision@k

For each new activity, we check what proportion of its top-`k` nearest neighbors belong to the same class.

**Definition:**

```math
I_{prec} = \frac{1}{|A_r|} \sum_{a_r \in A_r} \frac{1}{|\psi(a_r)|} \sum_{a_{new} \in \psi(a_r)} \frac{1}{k} \left|\left\{ a \in \mathrm{KNN}(a_{new}) \;\middle|\; \phi(a) = \phi(a_{new}) \right\}\right|
```

Where:

- $KNN(a_{new})$ returns the $k = |\psi(a_{new})| - 1$ nearest neighbors of $a_new$ in $A(L') \setminus \{a_{new}\}$ according to $d$.

---

#### Triplet

We measure how often an out-of-class activity is farther from a given anchor than another in-class activity.

**Definition:**

```math
I_{tri} = \frac{1}{|A_r|} \sum_{a_r \in A_r} \frac{1}{|\psi(a_r)|^2 - |\psi(a_r)|} \sum_{\substack{a_i, a_j \in \psi(a_r) \\ a_i \neq a_j}} \frac{1}{|A(L') \setminus \psi(a_r)|} \sum_{a_k \notin \psi(a_r)} \mathbf{1}(a_i, a_j, a_k)
```

Where:

```math
\mathbf{1}(a_i, a_j, a_k) =
\begin{cases}
1, & d(a_i, a_j) < d(a_i, a_k) \\
0, & \text{otherwise}
\end{cases}
```


## :fast_forward: Next Activity Prediction

Following the approach of **Gamallo-Fernandez et al. (2023)** — *"Learning Context-Based Representations of Events in Complex Processes"* ([DOI](https://doi.org/10.1109/ICWS60048.2023.00041)) — we initialize the embedding layer of a next-activity prediction model with **pre-trained activity embeddings**, and **freeze its weights during training** to evaluate their effectiveness.

---

### 📁 Datasets

- Raw event logs are located in:  
  [`evaluation/evaluation_of_activity_distances/next_activity_prediction/raw_datasets/`](evaluation/evaluation_of_activity_distances/next_activity_prediction/raw_datasets)
and [`evaluation/evaluation_of_activity_distances/next_activity_prediction/raw_datasets_that_are_not_evaluated/`](evaluation/evaluation_of_activity_distances/next_activity_prediction/raw_datasets_that_are_not_evaluated)

- We have split with [`evaluation/evaluation_of_activity_distances/next_activity_prediction/generate_new_event_log_splits.py`](evaluation/evaluation_of_activity_distances/next_activity_prediction/generate_new_event_log_splits.py) the logs into:
  - 64% training
  - 16% validation
  - 20% test

- The splits are saved under:  
  [`evaluation/evaluation_of_activity_distances/next_activity_prediction/split_datasets/`](evaluation/evaluation_of_activity_distances/next_activity_prediction/split_datasets/)

All logs found in the [`evaluation/evaluation_of_activity_distances/next_activity_prediction/raw_datasets/`](evaluation/evaluation_of_activity_distances/next_activity_prediction/raw_datasets) folder will be used automatically for training and evaluation.
You can move the other logs you want to evaluate from [`evaluation/evaluation_of_activity_distances/next_activity_prediction/raw_datasets_that_are_not_evaluated`](evaluation/evaluation_of_activity_distances/next_activity_prediction/raw_datasets_that_are_not_evaluated) to this folder.

---

### 🤖 Evermann Model

> Evermann, J., Rehse, J.-R., & Fettke, P.  
> *"Predicting process behaviour using deep learning."*  
> Decision Support Systems 100 (2017): 129–140.  
> [DOI](https://doi.org/10.48550/arXiv.1612.04600)

Our implementation follows the codebase provided in:  
[github.com/ERamaM/PMDLComparator](https://github.com/ERamaM/PMDLComparator/blob/master/evermann/train.py)  
as part of the paper:  
Rama-Maneiro et al. (2021), *"Deep learning for predictive business process monitoring: Review and benchmark."*  
IEEE Transactions on Services Computing, 16(1), 739–756.

---

#### ▶️ Running the Benchmark

To start training and evaluation of the Evermann model, execute the following script:

> [`evaluation/evaluation_of_activity_distances/next_activity_prediction/next_activity_prediction_everman.py`](evaluation/evaluation_of_activity_distances/next_activity_prediction/next_activity_prediction_everman.py)

---

#### ⚙️ Configurable Settings

You can configure which similarity / embedding methods the Evermann model will be evaluated with by modifying the `embedding_methods` list in the script.

- **`'one_hot'`** represents the **baseline** configuration.  
  In this mode, the embedding layer is **randomly initialized** and **trained from scratch** during model training.
---

#### 📊 Evermann Model Results

For each `<log_name>` and embedding `<method>`, results are saved to [`evaluation/evaluation_of_activity_distances/next_activity_prediction/results_everman`](evaluation/evaluation_of_activity_distances/next_activity_prediction/results_everman):

```
results_everman/<log_name>/<method>/
└── <log_name>.txt       # Summary of evaluation metrics
```


#### <log_name>.txt:
- Accuracy
- Matthews Correlation Coefficient (MCC)
- Brier Score
- Weighted Precision
- Weighted Recall
- Weighted F1 Score

---

#### 📈 Analyzing Evermann Results

To aggregate and analyze model performance across logs, run:
[`additional_scripts/calculate_next_activity_results_everman.py`](additional_scripts/calculate_next_activity_results_everman.py)

Before Running:

Make sure to configure the `all_logs` list in the script to specify which logs you want to include in the analysis.

### Tax Model

> Tax, Niek, et al. "Predictive business process monitoring with LSTM neural networks."  
> *Advanced Information Systems Engineering: 29th International Conference, CAiSE 2017.*  
> [DOI: 10.1007/978-3-319-59536-8_30](https://doi.org/10.1007/978-3-319-59536-8_30)

Our implementation follows the codebase provided in:  
[github.com/ERamaM/PMDLComparator](https://github.com/ERamaM/PMDLComparator/blob/master/tax/code/train.py)

---

#### ▶️ Running the Benchmark

To train and evaluate the Tax model on your logs, run:


[`evaluation/evaluation_of_activity_distances/next_activity_prediction/next_activity_prediction_tax.py`](evaluation/evaluation_of_activity_distances/next_activity_prediction/next_activity_prediction_tax.py)


This will process all logs in [`evaluation/evaluation_of_activity_distances/next_activity_prediction/raw_datasets/`](evaluation/evaluation_of_activity_distances/next_activity_prediction/raw_datasets), using predefined splits and save results accordingly.

---

#### 📁 Results

All output files will be stored in:


[`evaluation/evaluation_of_activity_distances/next_activity_prediction/results_tax/`](evaluation/evaluation_of_activity_distances/next_activity_prediction/results_tax)

---

#### 📊 Analyze Results

To aggregate and visualize the performance across different logs:

[`additional_scripts/calculate_next_activity_results_tax.py`](additional_scripts/calculate_next_activity_results_tax.py)


---

## :hourglass_flowing_sand: Runtime Analysis

Run [`evaluation/evaluation_of_activity_distances/runtime_analysis/runtime_analysis.py`](evaluation/evaluation_of_activity_distances/runtime_analysis/runtime_analysis.py)

Modify the `log_list` list in the script to specify which logs you want to analyze from the folder [`event_logs`](event_logs). 

Adjust the value of `number_of_repetitions = 5` to change how many independent runs each method is averaged over.


## 🗃️ Datasets


### Event Log Statistics

| Event Log                                                                                                                                     |   # Unique Activities |   # of Traces |   #Variants/#Traces | Min. Trace Length |   Avg. TL |   Max. TL | Intrinsic   | Next Act.   | Runtime   |
|:----------------------------------------------------------------------------------------------------------------------------------------------|----------------------:|--------------:|--------------------:|------------------:|----------:|----------:|:------------|:------------|:----------|
| [BPIC12](https://data.4tu.nl/collections/BPI_Challenge_2012/5065419)                                                                          |                    24 |         13087 |                0.33 |                 3 |     20.04 |       175 | ✓           | ×           | ×         |
| [BPIC12 A](https://data.4tu.nl/collections/BPI_Challenge_2012/5065419)                                                                        |                    10 |         13087 |                0    |                 3 |      4.65 |         8 | ✓           | ✓           | ×         |
| [BPIC12 C.](https://data.4tu.nl/collections/BPI_Challenge_2012/5065419)                                                                       |                    23 |         13087 |                0.33 |                 3 |     12.57 |        96 | ✓           | ×           | ×         |
| [BPIC12 O](https://data.4tu.nl/collections/BPI_Challenge_2012/5065419)                                                                        |                     7 |          5015 |                0.03 |                 3 |      6.23 |        30 | ✓           | ✓           | ×         |
| [BPIC12 W](https://data.4tu.nl/collections/BPI_Challenge_2012/5065419)                                                                        |                     7 |          9658 |                0.27 |                 2 |     17.61 |       156 | ✓           | ✓           | ×         |
| [BPIC12 W C.](https://data.4tu.nl/collections/BPI_Challenge_2012/5065419)                                                                     |                     6 |          9658 |                0.23 |                 1 |      7.5  |        74 | ✓           | ×           | ×         |
| [BPIC13 C. P.](https://data.4tu.nl/collections/_/5065448/1)                                                                             |                     7 |          1487 |                0.22 |                 1 |      4.48 |        35 | ✓           | ✓           | ×         |
| [BPIC13 I.](https://data.4tu.nl/collections/_/5065448/1)                                                                                |                     4 |          7554 |                0.2  |                 1 |      8.68 |       123 | ✓           | ✓           | ×         |
| [BPIC13 O. P.](https://data.4tu.nl/collections/_/5065448/1)                                                                             |                     3 |           819 |                0.13 |                 1 |      2.87 |        22 | ✓           | ✓           | ×         |
| [BPIC15 1](https://data.4tu.nl/collections/_/5065424/1)                                                                                 |                   398 |          1199 |                0.98 |                 2 |     43.55 |       101 | ✓           | ✓           | ✓         |
| [BPIC15 2](https://data.4tu.nl/collections/_/5065424/1)                                                                                       |                   410 |           832 |                1    |                 1 |     53.31 |       132 | ✓           | ✓           | ×         |
| [BPIC15 3](https://data.4tu.nl/collections/_/5065424/1)                                                                                       |                   383 |          1409 |                0.96 |                 3 |     42.36 |       124 | ✓           | ✓           | ×         |
| [BPIC15 4](https://data.4tu.nl/collections/_/5065424/1)                                                                                       |                   356 |          1053 |                1    |                 1 |     44.91 |       116 | ✓           | ✓           | ×         |
| [BPIC15 5](https://data.4tu.nl/collections/_/5065424/1)                                                                                       |                   389 |          1156 |                1    |                 5 |     51.11 |       154 | ✓           | ✓           | ×         |
| [BPIC17](https://data.4tu.nl/articles/dataset/BPI_Challenge_2017/12696884/1)                                                                  |                    26 |         31509 |                0.51 |                10 |     38.16 |       180 | ✓           | ×           | ×         |
| [BPIC18](https://data.4tu.nl/articles/dataset/BPI_Challenge_2018/12688355/1)                                                                  |                    41 |         43809 |                0.65 |                24 |     57.39 |      2973 | ✓           | ×           | ×         |
| [BPIC19](https://data.4tu.nl/articles/dataset/BPI_Challenge_2019/12715853/1)                                                                  |                    42 |        251734 |                0.05 |                 1 |      6.34 |       990 | ✓           | ×           | ✓         |
| [BPIC20 D. D.](https://data.4tu.nl/datasets/6a0a26d2-82d0-4018-b1cd-89afb0e8627f)                                                                  |                    17 |         10500 |                0.01 |                 1 |      5.37 |        24 | ✓           | ✓           | ×         |
| [BPIC20 I. D.](https://data.4tu.nl/datasets/91fd1fa8-4df4-4b1a-9a3f-0116c412378f)                                                                  |                    34 |          6449 |                0.12 |                 3 |     11.19 |        27 | ✓           | ✓           | ×         |
| [BPIC20 P. L.](https://data.4tu.nl/datasets/db35afac-2133-40f3-a565-2dc77a9329a3)                                                                  |                    51 |          7065 |                0.21 |                 3 |     12.25 |        90 | ✓           | ✓           | ×         |
| [BPIC20 P. T. C.](https://data.4tu.nl/datasets/fb84cf2d-166f-4de2-87be-62ee317077e5)                                                               |                    29 |          2099 |                0.1  |                 1 |      8.69 |        21 | ✓           | ✓           | ×         |
| [BPIC20 R. F. P.](https://data.4tu.nl/datasets/a6f651a7-5ce0-4bc6-8be1-a7747effa1cc)                                                               |                    19 |          6886 |                0.01 |                 1 |      5.34 |        20 | ✓           | ✓           | ×         |
| [CCC19](https://data.4tu.nl/articles/_/12714932/1)                                                                                            |                    29 |            20 |                1    |                52 |     69.7  |       118 | ✓           | ×           | ×         |
| [Env Permit](https://data.4tu.nl/articles/dataset/Receipt_phase_of_an_environmental_permit_application_process_WABO_CoSeLoG_project/12709127) |                    27 |          1434 |                0.08 |                 1 |      5.98 |        25 | ✓           | ✓           | ×         |
| [Helpdesk](https://data.4tu.nl/articles/_/12675977/1)                                                                                         |                    14 |          4580 |                0.05 |                 2 |      4.66 |        15 | ✓           | ✓           | ×         |
| [Hospital Billing](https://data.4tu.nl/articles/_/12705113/1)                                                                                 |                    18 |        100000 |                0.01 |                 1 |      4.51 |       217 | ✓           | ×           | ×         |
| [Nasa](https://data.4tu.nl/articles/_/12696995/1)                                                                                             |                    47 |          2566 |                0.98 |                12 |     28.7  |        50 | ×           | ✓           | ×         |
| [RTFM](https://data.4tu.nl/articles/_/12683249/1)                                                                                             |                    11 |        150370 |                0    |                 2 |      3.73 |        20 | ✓           | ×           | ×         |
| [Sepsis](https://data.4tu.nl/articles/dataset/Sepsis_Cases_-_Event_Log/12707639/1)                                                            |                    16 |          1050 |                0.81 |                 3 |     14.49 |       185 | ✓           | ✓           | ×         |

## Journal extension: Uncertain Event Data

This track corresponds to the journal extension (*Distributional Similarity Between Activities in Certain and Uncertain Event Data*). It extends distributional similarity to uncertain event data by using **expected counts** (probability-weighted frequencies) and by adapting act2vec via an expected-loss objective.

### Data: IKEA ASM uncertain event logs

The uncertain event logs used in the journal experiments are derived from **IKEA ASM** and action recognition outputs.

- **In this repository (ready-to-run)**: processed uncertain XES logs and metadata are included under:
  - `uncertain_event_data/ikea_asm/`
- **Provenance / generation pipeline / additional documentation**:
  - [GitHub: `henrikkirchmann/IKEA_ASM_UncertainEventLogs`](https://github.com/henrikkirchmann/IKEA_ASM_UncertainEventLogs)

### Key semantics and parameters (paper terminology)

- **Uncertain event**: each event has a probability distribution over candidate activities, with missing mass interpreted as **non-existence** (IKEA ASM uses label `NA`).
- **Non-existence (`NA`)**: treated as “event absent” (skipped) when constructing trace realizations and context windows.
- **Uncertainty level \(u\)**: we construct controlled inputs by keeping the top-\(u\) most probable realizations per event (including possible non-existence) and **renormalizing**.  
  - The paper evaluates **\(u\in\{1,2,3\}\)**; the code supports larger values where feasible.
- **Window sizes**:
  - Intrinsic (certain): \(w\in\{3,5,9\}\)
  - Intrinsic (uncertain) in the paper: \(w\in\{3,5\}\) (feasibility)
  - Next-activity prediction (uncertain): \(w\in\{3,5\}\)

### Methods (uncertain)

#### Uncertain count-based family (expected counts)

We evaluate 12 variants obtained by combining:
- **Matrix type**: AA (activity–activity co-occurrence) vs AC (activity–context frequency)
- **Context interpretation**: Seq vs MSet
- **Post-processing**: none vs PMI vs PPMI

Implementation (uncertain methods):
- `distances/uncertain_activity_distances/`
- Shared method dispatch / naming lives in `evaluation/data_util/util_activity_distances_uncertain.py`

#### Uncertain act2vec (expected loss)

We evaluate act2vec CBOW and Skip-gram adapted to uncertain logs by minimizing an expected loss over probabilistically realized context windows.

Entry script:
- `uncertain_scripts/run_uncertain_act2vec.py`

---

## :test_tube: Intrinsic Evaluation (uncertain event data)

### 🔧 How to run the intrinsic benchmark (uncertain)

Run:

```bash
python evaluation/evaluation_of_activity_distances/intrinsic_evaluation_uncertain/evaluation_activity_distance_intrinsic_uncertain.py
```

### ⚙️ Configurable settings (important parameters)

This script is IDE/PyCharm-friendly and configured by editing the constants in the `if __name__ == "__main__":` section:

- **Methods**: `activity_distance_functions` (12 uncertain count-based + 2 uncertain act2vec), then window suffixes are added via `add_window_size_evaluation(...)`
- **Window sizes**: `window_size_list = [3, 5, 9]` (set to `[3, 5]` to match the paper’s uncertain setup)
- **Ground-truth generation (mirrors certain benchmark)**:
  - `r_min`: max number of replaced activities (paper uses smaller ranges for feasibility in the uncertain setting)
  - `w`: class size (number of replacements per replaced activity)
  - `sampling_size`: number of sampled ground-truth logs
  - `load_ground_truth_logs`: cache toggle
- **Evaluated uncertain logs**: `log_list` (base names of XES files searched under `uncertain_event_logs/`)

Uncertainty handling is controlled near the top of the file:
- `BASE_TOPK`: constructs a common base log by truncation+renormalization to top-\(k\) candidates
- `UNCERTAINTY_LEVELS`: the evaluated \(u\) levels (paper uses \(u\in\{1,2,3\}\))

### 💾 Outputs (where results are written)

The intrinsic uncertain benchmark writes:

- **Cached ground-truth logs**:
  - `evaluation/evaluation_of_activity_distances/intrinsic_evaluation_uncertain/newly_created_logs/<log_name>/r_<r>_w_<w>_s_<s>.pkl`
- **Cached per-method results**:
  - `evaluation/evaluation_of_activity_distances/intrinsic_evaluation_uncertain/results/<log_name>/<method>/r_<r>_w_<w>_s_<s>_u_<u>.pkl`
- **Per-run CSVs**:
  - `results/activity_distances/intrinsic_uncertain/<log_name>/<log_name>_distfunc_<method>_u<u>_r<r>_w<w>_samplesize_<s>.csv`

### 📊 Summarizing + plotting intrinsic uncertain results (paper-quality)

To aggregate the per-run pickles into a single CSV and then create the paper plots:

```bash
python uncertain_scripts/summarize_uncertain_intrinsic_results.py
python uncertain_scripts/plot_uncertain_intrinsic_results_pretty.py
```

Main inputs/outputs for the plot script:
- **Input CSV**: `results/activity_distances/intrinsic_uncertain_summary/intrinsic_uncertain_aggregated_mean.csv`
- **Plots (PDF/SVG)**: `results/activity_distances/intrinsic_uncertain_summary/paper_plots/`

---

## :fast_forward: Next-Activity Prediction (uncertain event data; Evermann)

### 🔧 Running the benchmark

Run:

```bash
python uncertain_scripts/run_uncertain_next_activity_prediction_evermann.py
```

In the paper, we report results for two IKEA ASM logs:
- `clip_based__i3d__dev2__pretrained__rgb` (I3D, RGB, dev2)
- `pose_based__HCN_32__pretrained__pose__dev3` (HCN-32, Pose, dev3)

### ⚙️ Configurable settings (important parameters)

Edit these constants in `uncertain_scripts/run_uncertain_next_activity_prediction_evermann.py`:

- **Evaluated IKEA ASM logs**: `MODEL_IDS`  
  (must match folders under `uncertain_event_data/ikea_asm/split=test/model=<MODEL_ID>/`)
- **Window sizes**: `WINDOW_SIZES = [3, 5]`
- **Uncertainty cap per segment**: `TOP_K_EVENT = 3`
- **Embedding training variants** (paper notation \(u\) via training input):
  - `top3_uncertain` (uses uncertain inputs; corresponds to \(u=3\))
  - `top1_determinized` (determinized; corresponds to \(u=1\))
- **Representations**:
  - `expected_embedding` (probability-weighted sum of embeddings)
  - `scaled_concat_full` (probability-scaled concatenation; skipped for all `Uncertain AC*` methods for feasibility)
  - `argmax_onehot`, `weighted_onehot` (baselines)
- **Training parameters**: `EPOCHS`, `BATCH_SIZE`, `MAX_LEN`, `SEED`, `SPLIT_STRATEGY`, and `TF_DEVICE` (GPU vs CPU)

### 💾 Outputs

The runner writes per-run CSVs to `results/`:
- `results/next_activity_prediction_uncertain_evermann__<MODEL_ID>.csv`

### 📊 Paper-quality plot

```bash
python uncertain_scripts/plot_uncertain_next_activity_prediction_pretty.py
```

By default, the plotting script reads:
- `results/next_activity_prediction_uncertain_evermann__clip_based__i3d__dev2__pretrained__rgb.csv`
- `results/next_activity_prediction_uncertain_evermann__pose_based__HCN_32__pretrained__pose__dev3.csv`

Output:
- `results/next_activity_prediction_uncertain_evermann/paper_plots/uncertain_next_activity_prediction_test_acc_2panel.pdf`

---

## :hourglass_flowing_sand: Runtime Analysis (uncertain event data)

Run:

```bash
python uncertain_scripts/run_uncertain_runtime_analysis.py
```

Key parameters in `uncertain_scripts/run_uncertain_runtime_analysis.py`:
- `LOG_LIST`: which IKEA ASM uncertain logs to benchmark
- `WINDOW_SIZES`: e.g., `[3,5,9]`
- `METHODS`: explicit list of methods (count-based + neural) to evaluate
- `UNCERTAINTY_LEVEL_U`: apply top-\(u\) truncation + renormalization before measuring runtime (paper uses \(u=3\))
- `REPETITIONS`: number of repetitions per configuration

Output CSV:
- `results/runtime_results_uncertain_pycharm_<REPETITIONS>_repetitions.csv`

---

## Uncertain count-based runners (optional; exact expected counts)

If you want to recompute uncertain expected-count distances from XES (instead of using shipped CSVs/results), the most robust exact runner is:

- `uncertain_scripts/run_uncertain_window_based_all_methods_all_windows_sqlite_exact.py`

This uses a SQLite aggregation backend to keep RAM usage manageable (exact results, lower peak RAM).

## Citation

### ICPM 2025 paper

Please cite:

```bibtex
@inproceedings{kirchmann2025lets,
  title        = {Let’s Simply Count: Quantifying Distributional Similarity between Activities in Event Data},
  author       = {Kirchmann, Henrik and Fahrenkrog-Petersen, Stephan A. and Lu, Xixi and Weidlich, Matthias},
  booktitle    = {International Conference on Process Mining (ICPM)},
  year         = {2025},
  doi          = {10.1109/ICPM66919.2025.11220676}
}
```

### Journal extension

If you cite the uncertain-data extension, please cite the journal version (under review; BibTeX will be updated upon publication).
