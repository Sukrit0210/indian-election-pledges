# Tracking and Forecasting Pledges in Indian Election Manifestos

This project uses English Lok Sabha manifestos to find election pledges, compare them with an earlier election, and forecast how much attention different pledge topics may receive. It starts from the passage dataset built by the sentiment project. [Project report](REPORT.md)

## Research Questions

1. Can a classifier learn to find pledges from automatically assigned training labels?
2. Can a search model find an earlier pledge by the same party?
3. Can a model tell whether a promise was repeated, revised, or different, and point to the words that show the change?
4. Does extending a topic trend improve on using the previous election’s topic shares?

## Folder Structure

```text
data/processed/        Master passages, quasi-sentences, and statement topics
data/review/           Comparison criteria and reference cases
outputs/pledges/       Pledge scores, topic shares, and forecast results
outputs/comparison/   Retrieval candidates, pair decisions, and metrics
outputs/figures/      Plots in PNG and SVG
reference/            Saved results and review material from earlier runs
src/                  Numbered Python scripts
REPORT.md              Methods, results, and conclusions
requirements.txt       Python dependencies
```

## Scripts

Copy `master_passages.csv` from the sentiment project’s `data/processed/` folder into this project’s `data/processed/` folder. Install 64-bit Python 3.14, open PowerShell in the pledge project folder, and run:

```powershell
py -3.14 run_all.py
```

The launcher creates `.venv314/`, installs the pinned dependencies and PyTorch 2.10.0 with CUDA 12.6, checks the GPU, and runs stages 01–10. The first run needs internet access for the packages and model files. If the CSV is stored elsewhere, use `py -3.14 run_all.py --passages "PATH_TO_MASTER_PASSAGES.csv"`.

To run scripts individually after setup, select `.venv314\Scripts\python.exe` in VS Code and follow the order below. When running stage 01 directly with your copied dataset, pass its path explicitly:

```powershell
& .\.venv314\Scripts\python.exe src/01_build_quasi_sentences.py --passages data/processed/master_passages.csv
```

After stages 01–06 have completed, `py -3.14 run_all.py --start 7` retrains and reruns comparison, evaluation, and plotting. To regenerate evaluation and plots from existing predictions, close any open output CSVs and use `py -3.14 run_all.py --start 9`.

## What Each Script Does

### 1. Build Quasi-Sentences

[01_build_quasi_sentences.py](src/01_build_quasi_sentences.py) splits passages into shorter candidate statements and records their locations in the original passage. The script writes `data/processed/quasi_sentences.csv`.

### 2. Encode the Statements

[02_encode_statements.py](src/02_encode_statements.py) uses MiniLM in PyTorch to convert each statement into a numerical vector. The classifier and retrieval stages use these vectors.

### 3. Detect Pledges

[03_detect_pledges.py](src/03_detect_pledges.py) uses rules to label clear commitments and clear non-pledges. The rules can abstain. A logistic regression classifier learns from the historical rule labels using TF-IDF and MiniLM embeddings. [Pledge scores](outputs/pledges/pledge_scores.csv) · [Rule coverage](outputs/pledges/weak_coverage.csv)

### 4. Assign Policy Topics

[04_classify_topics.py](src/04_classify_topics.py) assigns a ManifestoBERTa category to each statement. The forecast aggregates these categories among detected pledges. The script writes `data/processed/statement_topics.csv`.

### 5. Forecast Topic Shares

[05_forecast_topics.py](src/05_forecast_topics.py) tests ways to forecast 2024 topic shares using earlier elections and exports exploratory 2029 estimates. [2024 backtest](outputs/pledges/forecast_summary.csv) · [Forecasts](outputs/pledges/topic_forecasts.csv)

### 6. Find Earlier Pledges

[06_retrieve_pledges.py](src/06_retrieve_pledges.py) searches the same party’s previous manifesto. It compares keyword search (BM25), MiniLM similarity, and a combination of the two rankings. [Retrieval candidates](outputs/comparison/retrieval_pool.csv)

### 7. Train the Comparison Model

[07_train_cross_encoder.py](src/07_train_cross_encoder.py) prepares historical training pairs and controlled quantity edits, then fine-tunes MiniLM in PyTorch with AdamW and cross-entropy. The current setting is **five fixed epochs**, and the progress message uses that count. The checkpoint is saved to `.cache/trained_cross_encoder/`. [Training pairs](outputs/comparison/training_pairs.csv)

### 8. Compare Pledges

[08_compare_pledges.py](src/08_compare_pledges.py) compares pairs and records their relationship. *Same* means the commitment is repeated, *revised* means a material part of the commitment changed, and *different* means the texts express separate commitments. The pipeline can also return *unclear*. For flagged revisions, rules suggest words to inspect. [Best matches](outputs/comparison/best_matches.csv) · [Codebook](data/review/CODEBOOK.md)

### 9. Evaluate

[09_evaluate.py](src/09_evaluate.py) checks whether search finds an earlier reference match and whether the comparison model agrees with the fixed reference pairs. Those reference judgments were made by a language model. [Test decisions](outputs/comparison/test_decisions.csv) · [Evaluation metrics](outputs/comparison/test_metrics.csv)

For future independent evaluation, [make_review_form.py](src/make_review_form.py) prepares blank review forms. After review, pass the completed CSV to stage 09 with `--independent-review REVIEWED_FILE.csv`.

### 10. Make the Plots

[10_make_plots.py](src/10_make_plots.py) generates figures for [corpus coverage](outputs/figures/01_coverage.png), [retrieval](outputs/figures/03_retrieval.png), [relationship decisions](outputs/figures/04_relations.png), [revision evidence](outputs/figures/05_revision_audit.png), and the [forecast backtest](outputs/figures/06_forecast.png). The optional scripts and their commands are documented separately. [Follow-up experiments](EXPERIMENTS.md)

## Dataset

The project uses **21 English manifestos** from BJP, Congress, CPI(M), AIADMK, DMK, and TMC, from the 2009, 2014, 2019, and 2024 Lok Sabha elections. It splits **9,572 passages** into **15,376 candidate quasi-sentences**. Five parties have 2024 editions in the corpus. [Source inventory](data/sources.csv) · [Edition notes](data/edition_notes.md)

Each row contains `statement_id`, `passage_id`, `document_id`, `party`, `year`, `page`, `start_char`, `end_char`, and `text`. Join on `passage_id` to find the source passage. PDF pages start at 1. Character offsets start at 0 and exclude the end. The 5–90 word interval is used to flag unusual statement lengths for review.

## Main Results

| Task | Result from the completed run |
|---|---|
| Pledge detection | The classifier agreed with 97.7% of the 962 statements that received a weak label in 2024. |
| Retrieval | Hybrid search recovered 4 of 7 earlier reference matches in its top five with the pledge filter and 6 of 7 without it. |
| Pledge comparison | The five-epoch cross-encoder agreed with 31 of 35 scored reference pairs. Macro-F1 was 0.726 and revision F1 was 0.400. |
| Topic forecasting | Carrying forward the previous election’s topic shares gave lower error than extending half the recent trend in the 2024 backtest. Mean total variation distance was 0.186 versus 0.215. |

The [reference folder](reference/) holds saved results from earlier runs. The metrics above come from the completed five-epoch run in `outputs/`. [Detailed results and figures](REPORT.md)

## Limitations

- The classifier learns from rule labels and the pair evaluation uses language-model-reviewed examples. An independently annotated sample is needed to measure accuracy on natural pledges.
- The comparison panel contains only three scored examples of natural revisions. Training revisions were generated by changing quantities, so they cover a limited set of changes.
- The search challenge is small. The forecast uses a short election history and estimates topic shares among detected pledges.

## Conclusion

Searching without the pledge filter recovered more earlier matches on the challenge. The comparison model found one reviewed revision, with highlights that missed the change in the obligation. Carrying forward the earlier topic shares was the strongest tested forecasting baseline in 2024. [Full analysis](REPORT.md)

## References

- [Fornaciari et al., identifying election pledges with language models](https://aclanthology.org/2021.findings-acl.301/)
- [Manifesto Project coding handbook](https://manifestoproject.wzb.eu/information/documents?name=handbook_v4)
- [Model versions and checkpoint hashes](data/models.json)
