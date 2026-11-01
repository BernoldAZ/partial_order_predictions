# Partial-Order Suffix Prediction

Predictive process monitoring for event logs with **partial orders**: events in a case that share
a timestamp form a concurrent block. Our approach (**POPS**) encodes the prefix as a block graph
(GATv2 encoder) and decodes the suffix event by event with a GRU. Each step predicts the activity
and a *new-block* bit, so the predicted suffix is itself a partial order.

We compare POPS against the suffix-prediction baselines SEP-LSTM, ED-LSTM and SuTraN (NDA), plus an
Oracle-Sequential reference point. All of them use the same splits and the same metric code.

## Repository layout

```
partial_order_predictions/
├── dockerfile                 # image "ml-jupyter-gpu"  (approach, results, statistics)
├── requirements.txt
├── evaluate_suffix.ipynb      # results tables, ablation tables, plots, metric example
├── Check_logs.ipynb           # exploratory statistics on concurrency in the logs
├── approach_suffix_v2/        # our approach (POPS)
│   ├── Logs/                  # raw event logs (.xes / .xes.gz)
│   ├── Generate_splits.ipynb  # preprocessing of the approach
│   ├── data_creation/         # graph / sequence dataset builders (+ Preprocessing/)
│   ├── models_v2/             # models, train/eval scripts, metrics, oracle
│   ├── run_all_suffix.py      # train + evaluate a variant on all logs, 5 runs
│   ├── run_eval_flipped.py    # ablation: evaluate on prefix-flipped test sets
│   ├── run_eval_random.py     # ablation: evaluate on prefix-shuffled test sets
│   ├── generate_baseline_results.py   # main results table
│   ├── generate_ablation_results.py   # ablation tables
│   └── significance_tests.py          # Friedman + Nemenyi / CD diagrams
└── baselines/
    ├── dockerfile             # image "ppm-sutran-best"  (baselines)
    ├── requirements.txt
    ├── Logs/                  # raw event logs (same files as approach_suffix_v2/Logs)
    ├── run_evaluate_suffix_baselines.ipynb  # preprocessing of the baselines
    ├── create_general_data.py # preprocessing (SuTraN pipeline)
    ├── run_all_suffix_baselines.py          # train + evaluate all baselines on all logs, 5 runs
    ├── eval_ttne_partial_order.py           # partial order from predicted TTNE -> GES / BOS
    └── SuTraN/, LSTM_seq2seq/, OneStepAheadBenchmarks/, ...  # model code
```

## Event logs

We use these 8 logs:

| Log name (file stem)    | Source |
|-------------------------|--------|
| `BPIC15_1` … `BPIC15_5` | BPI Challenge 2015, municipalities 1–5 |
| `BPI_Challenge_2012_A`  | BPI Challenge 2012, `A_` sub-process |
| `BPI_Challenge_2012_O`  | BPI Challenge 2012, `O_` sub-process |
| `Sepsis`                | Sepsis Cases |

Put the files (`<name>.xes` or `<name>.xes.gz`) in **both** `approach_suffix_v2/Logs/` and
`baselines/Logs/`. The two folders must contain identical files. The file stem is used as the log name everywhere.

## Metrics

All metrics compare the predicted suffix with the ground-truth suffix. **The END token is
excluded from every metric**: both suffixes are cut at their first END token.

- **GES**: graph edit similarity between the block graphs of the two suffixes. It uses the exact GED, solved as a binary program with PuLP/CBC.
- **BOS**: bucket-order similarity, i.e. the Kendall distance with penalty ½ on the shared activity occurrences (*BOS order*), multiplied by the content F1 (*BOS content*).
- **DLS**: normalised Damerau-Levenshtein similarity of the activity sequences.

The metric code is `partial_order_metrics.py`, and the same file is used in both folders. The baselines predict
a total order, so we derive their blocks from the predicted time-till-next-event (see
[Run the baselines](#4-run-the-baselines)).

---

## 1. Docker and Jupyter

You need two images. Both expose JupyterLab on port 8887. On PowerShell, write `${PWD}` instead of `$(pwd)`.

### `ml-jupyter-gpu`: approach, results, statistics

Built from the root `dockerfile` (PyTorch 2.4, CUDA 12.4, PyTorch Geometric). Build and run it from the
**repository root**:

```bash
docker build -t ml-jupyter-gpu .

# JupyterLab on http://localhost:8887
docker run -it --rm --gpus all -p 8887:8887 -v $(pwd):/workspace ml-jupyter-gpu

# or run a single script
docker run -it --rm --gpus all -v $(pwd):/workspace ml-jupyter-gpu python approach_suffix_v2/run_all_suffix.py --help
```

### `ppm-sutran-best`: baselines

Built from `baselines/dockerfile` (CUDA 11.8, PyTorch 2.1). Build and run it from **inside `baselines/`**:

```bash
cd baselines
docker build -t ppm-sutran-best .

# JupyterLab on http://localhost:8887
docker run -it --rm --gpus all -p 127.0.0.1:8887:8887 -v $(pwd):/app ppm-sutran-best

# or run a single script
docker run -it --rm --gpus all -v $(pwd):/app ppm-sutran-best python3 run_all_suffix_baselines.py --help
```

Inside JupyterLab, open a notebook and run its cells from top to bottom. Each notebook already uses
the working directory its paths expect.

---

## 2. Preprocessing

Both sides use the same SuTraN preprocessing pipeline with the same defaults:

- chronological out-of-time split, `preferred` mode
- 20 % test, and 20 % of the remaining cases for validation
- window size = 98.5th percentile of the case lengths
- standardisation with training statistics

So the train / validation / test partitions are identical.

### Approach (`ml-jupyter-gpu`)

Open `approach_suffix_v2/Generate_splits.ipynb` and run:

| Cell | Output in `approach_suffix_v2/results_per_log/<log>/` | Needed for |
|------|-------------------------------------------------------|------------|
| *Graph input* | `{train,val,test}_graphdataset.pt`, summary `split_summary.csv` | main approach, decoder ablation |
| *Sequential* | `{train,val,test}_seqgraphdataset.pt`, `split_summary_seq.csv` | input ablation |
| *Sequential prefix random order* | `test_seqgraphdataset_prefixrandom.pt` | input ablation |
| *Sequential prefix flipped order* | `test_seqgraphdataset_prefixflip.pt` | input ablation |

### Baselines (`ppm-sutran-best`)

Open `baselines/run_evaluate_suffix_baselines.ipynb` and run the cell *Preproses all datasets sutran*.
It writes the tensors to `baselines/results_per_log/<log>/` and the summary to
`split_summary_sutran_best.csv`. You can skip this step: `run_all_suffix_baselines.py` builds any missing
datasets itself before training.

**Check:** `approach_suffix_v2/split_summary.csv` and `baselines/split_summary_sutran_best.csv`
must report the same `n_train / train_pairs / n_val / val_pairs / n_test / test_pairs` per log.

---

## 3. Run the approach (POPS)

In `ml-jupyter-gpu`, from the repository root:

```bash
python approach_suffix_v2/run_all_suffix.py --model suffix_v2 --run-ids 1 2 3 4 5 --workers 8
```

- Each log runs in its own subprocess. If a log runs out of GPU memory, it is retried on the CPU.
- Run *N* uses random seed *N*.
- Results go to `approach_suffix_v2/results_time_gatv2_gru_nb_v2_actnb/run_N/`:
  - `results_suffix_time_gnn.csv`: one row per log
  - the trained model `<log>_gatv2_gru_nb_v2_actnb.pt`
  - per-sample metrics `<log>_per_sample_metrics.pt`
  - per prefix/suffix-length dicts
- Logs that already have a result are skipped, so an interrupted run can simply be restarted.

| Flag | Effect |
|------|--------|
| `--model` | `suffix_v2` (POPS), `suffix_v2_seq`, `suffix_v2_block`, `suffix_v2_block_seq` |
| `--run-ids 1 2 3 4 5` | several repetitions with one shared worker pool (`--run-id N` for one) |
| `--workers N` | logs trained in parallel |
| `--no-train` | load the saved model instead of training |
| `--no-eval` | only time the inference (writes `inference_times.csv`) |
| `--force` | re-evaluate even if a result exists (implies `--no-train`) |

**Oracle-Sequential** is a reference point. It predicts every activity correctly, but places each event in its own block.
It needs no training:

```bash
python approach_suffix_v2/models_v2/run_oracle_sequential.py
# -> approach_suffix_v2/results_oracle_sequential/run_1/results_suffix_time_gnn.csv
```

---

## 4. Run the baselines

In `ppm-sutran-best`, from inside `baselines/`:

```bash
# 1. train + evaluate every model on every log, 5 repetitions (seed = run id)
python3 run_all_suffix_baselines.py --run-ids 1 2 3 4 5 --workers 8

# 2. derive the partial order from the predicted TTNE and compute GES / BOS
python3 eval_ttne_partial_order.py --workers 10 --chunksize 1
```

- Step 1 writes `results_per_log/<log>/<MODEL>_results_runN/TEST_SET_RESULTS/`, including
  `averaged_results.pkl` (with `'DL sim'`) and the decoded predictions. The flags `--no-train`,
  `--no-eval` and `--force` work as for the approach.
- Step 2 builds the baselines' blocks: suffix event *j* joins the block of event *j−1* when its predicted TTNE is ≤ *X*.
  - *X* is rate-matched per log, model and run: it is the *q*-quantile of the predicted test TTNEs, where *q* is the fraction of concurrent events in the training suffixes. No test labels are used.
  - It writes `po_ttne_results.pkl` next to each run and the summary `po_ttne_summary.csv`.
  - Optional: `python3 eval_ttne_partial_order_zero.py` compares this threshold with a fixed threshold *X = 0* (`po_ttne_zero_vs_theta.csv`).

The comparison uses **SEP-LSTM, ED-LSTM and SuTraN (NDA)**. The scripts also contain SuTraN (DA),
BEST, CRTP-LSTM and ASTON (`create_aston_data.py`, `TRAIN_EVAL_ASTON.py`). These are not part of the reported results.

---

## 5. Ablation tests

All ablations run in `ml-jupyter-gpu`, from the repository root. They reuse the same 5 runs.

**Input structure.** The decoder is the same and only the prefix encoding changes. The variants are block graph (POPS), sequential chain, and sequential chain with the
within-block order of the prefix reversed or randomly shuffled. The prefix is changed only on the test set, after the
prefix/suffix cut, so the labels are unchanged.

```bash
python approach_suffix_v2/run_all_suffix.py   --model suffix_v2_seq --run-ids 1 2 3 4 5 --workers 8
python approach_suffix_v2/run_eval_flipped.py --model suffix_v2_seq_prefixflip   --workers 8
python approach_suffix_v2/run_eval_random.py  --model suffix_v2_seq_prefixrandom --workers 8
```

The flip and random evaluations load the trained `suffix_v2_seq` models. They write
`results_suffix_time_gnn_prefixflip.csv` and `results_suffix_time_gnn_prefixrandom.csv` next to the
normal results.

**Decoder structure.** This compares the event-at-a-time decoder (POPS) with a block-at-a-time decoder, on the same graph input:

```bash
python approach_suffix_v2/run_all_suffix.py --model suffix_v2_block --run-ids 1 2 3 4 5 --workers 8
```

---

## 6. Results and statistical tests

### Tables (`evaluate_suffix.ipynb`, `ml-jupyter-gpu`)

Open `evaluate_suffix.ipynb` in the repository root:

- **Evaluate metrics using running example**: the GES / BOS / DLS values on a toy suffix.
- **Compare with baselines**: `generate_baseline_results.run("ttne")`.
  - It takes the mean ± std over the runs, per log, for POPS, Oracle-Sequential and the baselines. The baselines' GES/BOS come from the TTNE-derived partial order.
  - Output: `approach_suffix_v2/baseline_results.txt` (quality) and `approach_suffix_v2/scalability_results.txt` (training / inference time, parameters).
- **Ablation tests**: `generate_ablation_results.run()` prints the input and decoder ablation tables.
- **Results: per prefix-suffix len**: GES / BOS as a function of suffix length. Output: `approach_suffix_v2/ges_bos_*.pdf`.

### Significance tests

From the repository root (`ml-jupyter-gpu`):

```bash
python approach_suffix_v2/significance_tests.py all
```

- For each metric (GES, BOS, DLS), the test runs across the 8 logs (block = log, score = mean over runs).
- It runs a Friedman test, then a Nemenyi post-hoc test with a critical-difference diagram if the Friedman test is significant.
- With only two models, it runs a sign test instead.

| Mode | Comparison |
|------|------------|
| `main` | POPS vs SEP-LSTM, ED-LSTM, SuTraN (NDA) |
| `ablation-input` | Graph vs Seq vs SeqFlip (`--input-variants` to choose) |
| `ablation-decoder` | event-at-a-time vs block-at-a-time decoder (GES, BOS) |
| `all` | all three |

Options: `--runs 1,2,3,4,5`, `--run N`, `--alpha 0.05`. Outputs go to
`approach_suffix_v2/significance_tests/`: `significance_tests.txt`, `significance_tests.csv` and
`cd_<table>_<metric>.png`.

---

## 7. Check partial orders in other event logs

`Check_logs.ipynb` (repository root, `ml-jupyter-gpu`) measures how much partial order an event log
contains, i.e. how often events in a case share a timestamp. Use it to check whether another log is
suitable for this approach:

1. Download event logs, e.g. from <https://www.processmining.org/> or <https://data.4tu.nl/>.
2. Create a folder `all_event_logs/` in the repository root and save the logs there as `.xes` or
   `.xes.gz` files. The folder is git-ignored.
3. Open `Check_logs.ipynb` and run the cells.

For every log, the notebook reports:
- the number of traces and variants
- the share of traces that contain events with the same timestamp (`multiple_ts_%`)
- the share of concurrent activities (`%_conc_act`)
- block sizes and trace lengths

The table is saved to `all_logs_analysis.csv`. The last cells of the notebook draw the block graph of
single traces from a log.
