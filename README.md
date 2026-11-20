# Tracking and Forecasting Pledges in Indian Election Manifestos

This repository detects pledges in English Lok Sabha manifestos, forecasts which policy topics may return, and compares individual promises across elections.

[Data, methods, analysis, results and conclusion](REPORT.md) · [Follow-up experiments](EXPERIMENTS.md) · [Manifesto inventory](data/sources.csv) · [Pledge comparison codebook](data/review/CODEBOOK.md)

## Changes in this version

- `src/07_train_cross_encoder.py` sets `epochs = 5` for MiniLM training. The epoch counter reads from that setting and displays the correct total.
- `run_all.py` prepares a project-local Python 3.14 environment with CUDA-enabled PyTorch. Its default input is the master CSV copied from the sentiment project.

## Input and setup

Copy `master_passages.csv` from the sentiment project into this project’s `data/processed/` folder. Use the same source editions to reproduce the reference panel and page links. To use a master CSV in a different location, pass `--passages "PATH_TO_MASTER_PASSAGES.csv"`.

Open this folder in VS Code with 64-bit Python 3.14 selected, then run `run_all.py`. The first run creates `.venv314/`, installs PyTorch 2.10.0 with CUDA 12.6, and checks the GPU. The pipeline downloads the pinned MiniLM, ManifestoBERTa, and comparison checkpoints to `.cache/models/`. You can also start it from PowerShell:

```powershell
py -3.14 run_all.py
```

After setup, select `.venv314\Scripts\python.exe` in VS Code to run an individual script. [Windows CUDA wheels](https://download.pytorch.org/whl/cu126/torch/)

## Run each step yourself

| Command | What it produces |
|---|---|
| `python src/01_build_quasi_sentences.py` | A statement-level CSV with parent passage IDs, PDF pages and character offsets. |
| `python src/02_encode_statements.py` | One PyTorch MiniLM vector per statement. |
| `python src/03_detect_pledges.py` | Weak-label votes, detected pledges and a review sample. |
| `python src/04_classify_topics.py` | ManifestoBERTa policy categories for each statement. |
| `python src/05_forecast_topics.py` | Topic shares, the 2024 backtest and exploratory 2029 forecasts. |
| `python src/06_retrieve_pledges.py` | BM25, MiniLM and hybrid search candidates from the same party’s previous election. |
| `python src/07_train_cross_encoder.py` | Historical training examples and a fine-tuned MiniLM checkpoint. |
| `python src/08_compare_pledges.py` | Same, revised, different or unclear decisions and candidate evidence spans. |
| `python src/09_evaluate.py` | Retrieval and classification metrics using the fixed reference cases. |
| `python src/10_make_plots.py` | Coverage, retrieval, relation, evidence audit and forecast plots. |

To rerun steps 07–10 after editing the trainer, use `py -3.14 run_all.py --start 7` after generating the retrieval pool.

## Results from the local run

The five-epoch MiniLM cross-encoder agreed with 31 of 35 scored reference pairs and detected one of the three reviewed revisions. Its macro-F1 was 0.726 and its revision F1 was 0.400. The reference decisions came from a language model. [Comparison metrics](outputs/comparison/test_metrics.csv)

| Output | What to look for |
|---|---|
| [Best matches](outputs/comparison/best_matches.csv) · [Test decisions](outputs/comparison/test_decisions.csv) | Earlier and later statements, predicted relationships, and the evidence spans used for inspection. |
| [Retrieval comparison](outputs/figures/03_retrieval.png) | Hybrid search recovered 4 of 7 reference matches in its top five with the pledge filter and 6 of 7 without it. |
| [Relationship confusion matrix](outputs/figures/04_relations.png) · [Revision audit](outputs/figures/05_revision_audit.png) | The comparison results and how many flagged 2024 revisions could be checked against the archived review. |
| [Forecast summary](outputs/pledges/forecast_summary.csv) · [Forecast figure](outputs/figures/06_forecast.png) | Carrying forward the previous election’s topic shares had the lowest 2024 backtest error among the tested approaches. |

## Units, labels and evaluation

- Join a statement’s `passage_id` to `master_passages.csv`. PDF pages start at 1. Character ranges start at 0 and exclude the end. The 5–90 word interval flags unusual lengths for review.
- Rules vote for pledge or non-pledge, and abstain on unclear cases. The binary classifier learns from weak labels through 2019. The relation model learns from historical pairs and controlled edits. The current PyTorch trainer uses cross-entropy, AdamW, five fixed epochs and batches of 16.
- The exploratory judgments in the shipped review panels and evidence audit were produced with a language model. `python src/make_review_form.py` prepares blank forms for independent evaluation. After independent review, `python src/09_evaluate.py --independent-review REVIEWED_FILE.csv` scores the completed form against the generated predictions.

The `reference/` directory contains the prior verified PyTorch output tables, source-linked examples, and selected plots. The numbered scripts generate new files under `data/processed/` and `outputs/`. The supplied results cover the registered editions and use the pinned checkpoints in [models.json](data/models.json).
