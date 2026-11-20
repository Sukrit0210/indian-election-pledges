# Tracking and Forecasting Pledges in Indian Election Manifestos

## 1. Introduction

Parties often address the same issues in successive elections. Their commitments can stay the same or change in ways that matter, such as a change in the action, eligibility, or amount. This project uses text from my larger manifesto corpus to find related pledges, compare their wording, and check whether earlier elections help forecast the mix of pledge topics.

## 2. Objectives

1. Detect pledges using automatically assigned training labels.
2. Retrieve earlier pledges from the same party and classify them as the same, revised, or different.
3. Check whether the highlighted words show the change in a revised pledge.
4. Compare simple forecasts of pledge topic shares on the 2024 election.

## 3. Data Sources

The corpus contains 21 English Lok Sabha manifestos from BJP, Congress, CPI(M), AIADMK, DMK and TMC. It covers the 2009, 2014, 2019, and 2024 elections. BJP, Congress, and CPI(M) have all four editions. DMK and TMC start in 2014. AIADMK is represented through 2019. Existing English editions were required. [Source inventory](data/sources.csv) · [Edition notes](data/edition_notes.md)

The passage table contains **848 OCR passages from five documents**. Four documents account for 840 of those passages. AIADMK 2014 contributes the remaining eight. Extraction counts document how much text depended on OCR. Word error rates require source transcriptions. The audit script exports a spot-check form with document links and page numbers. [OCR counts](reference/followup/ocr_passages_by_document.csv)

## 4. Dataset Construction

The master passage CSV is copied from the sentiment project and the completed run used byte-identical copies in both repositories. Splitting 9,572 passages produced 15,376 candidate quasi-sentences. Each keeps the source document ID, PDF page, parent passage ID, and character offsets. The 5–90 word interval flags unusual statement lengths for review. The statement dataset is generated at `data/processed/quasi_sentences.csv`. [Segmentation script](src/01_build_quasi_sentences.py)

## 5. Pledge Detection Method

Rules identify future commitments and clear descriptions of the past, and abstain in uncertain cases. Train a logistic regression classifier using TF-IDF and frozen MiniLM embeddings on historical weak labels through 2019.

## 6. Policy Topics and Forecast Method

Use ManifestoBERTa’s 56 policy categories. Compare repeating the previous election’s topic shares with extending the recent trend. Evaluate on 2024 topic shares estimated from detected pledges. Use the latest available shares to generate exploratory 2029 forecasts for the five parties with 2024 coverage.

## 7. Retrieval and Comparison Method

Search the same party’s previous manifesto with BM25 and MiniLM, then combine their ranks. Fine-tune a MiniLM cross-encoder in PyTorch on rule-labeled historical pairs and controlled edits. The model classifies the retrieved pairs as same, revised, different or unclear. Rules highlight quantities or edited phrases when a pair is classified as a revision.

The comparison labels are **same** for a repeated commitment, **revised** for a material change to the earlier commitment, and **different** for separate commitments. The comparison stage can abstain with **unclear**. The trainer runs for five fixed epochs. It screens both statements in every candidate training pair against the pilot, test, and constructed control sets. It checks source IDs and normalized exact text. For text of at least five tokens it also flags a word-trigram Jaccard similarity of 0.80 or greater. [Comparison codebook](data/review/CODEBOOK.md) · [Training script](src/07_train_cross_encoder.py) · [Split checks](src/split_audit.py)

## 8. Evaluation Method

A targeted challenge measures whether earlier matches appear in the top five. A fixed set of language-model-reviewed pairs measures relation agreement and whether the highlighted text supports the claimed change.

The full model names and pinned revisions are recorded in the checkpoint manifest. The relation model uses 0.65 as a working abstention cutoff. The retrospective threshold check below uses the saved test probabilities. [Checkpoints and file hashes](data/models.json)

## 9. Pledge Detection Results

The detector trained on 1,906 rule-labelled statements from elections through 2019. In 2024 the rules labelled 962 of 4,020 statements, and the detector agreed with 97.7% of those labels. The rules abstained on the other 3,058 statements. The percentage measures agreement on the subset with weak labels. Independent pledge judgements would be needed to estimate detector accuracy. [Rule agreement](outputs/pledges/agreement_with_2024_rules.json) · [Rule coverage](outputs/pledges/weak_coverage.csv)

## 10. Retrieval Results

The completed run searched for earlier matches for 7,625 later statements across 15 adjacent party-election transitions. It scored 38,125 top-five pairs. [Selected matches](outputs/comparison/best_matches.csv)

The exploratory challenge contains 11 later statements, seven with an earlier reference match. The pledge filter admitted five of those seven. The following table counts a hit when the method ranks at least one earlier reference in its top five. The end-to-end system selected a correct earlier statement for two of the seven queries. [Retrieval results](outputs/figures/03_retrieval.csv) · [End-to-end outcomes](outputs/comparison/retrieval_challenge.csv)

| Method | With pledge filter | Without pledge filter |
|---|---:|---:|
| BM25 | 3/7 | 5/7 |
| Dense MiniLM | 4/7 | 6/7 |
| Hybrid rank fusion | 4/7 | 6/7 |

## 11. Pledge Comparison and Evidence Results

The split check excluded **92 of 2,063** candidate pairs. The first matching check was a shared statement ID for 78 pairs, normalized exact text for two, and similar wording for 12. The trainer used the remaining **1,971 pairs** (820 same, 684 different, and 467 revised). All 467 revised training pairs came from controlled quantity edits. The final audit recorded zero matches under the implemented checks. [Split audit](outputs/comparison/training_pairs_split_audit.csv) · [Excluded pairs](outputs/comparison/training_pairs_excluded.csv) · [Retained training pairs](outputs/comparison/training_pairs.csv)

The test file yielded 38 unique 2019–2024 pairs after deduplication. Three reference labels were unclear, leaving **35 scored pairs** (25 different, seven same, and three revised). A language model assigned the reference labels. [Test decisions](outputs/comparison/test_decisions.csv)

| Method | Agreement with reference | Abstentions | Macro-F1 | Revision F1 |
|---|---:|---:|---:|---:|
| Always different | 25/35 | 0 | 0.278 | 0.000 |
| Rule baseline | 21/35 | 13 | 0.473 | 0.000 |
| Five-epoch MiniLM cross-encoder | 30/35 | 1 | 0.592 | 0.000 |

At the 0.65 cutoff, the cross-encoder matched all 25 different pairs and five of seven same pairs. It missed **all three reference revisions**, predicting different for two and abstaining on one. Its sole revised prediction was a pair labelled same by the reference. On the 2014–2019 pilot panel it matched 32 of 37 determinate labels (macro-F1 0.710) and identified one of four reference revisions. Neither panel had a correctly labelled revision with matching evidence spans on both sides. [Test metrics](outputs/comparison/test_metrics.csv) · [Pilot metrics](outputs/comparison/pilot_metrics.csv) · [Evidence counts](outputs/comparison/evaluation.json)

In a retrospective threshold check, 0.50 gave 31 matches, one correctly flagged revision, and no abstentions. The tested cutoffs of 0.60, 0.65, 0.70, 0.80, and 0.90 each gave 30 matches, zero correctly flagged revisions, and one abstention. The main results use the existing 0.65 setting. [Saved scores](outputs/comparison/panel_predictions.csv)

![Current pledge relation results](outputs/figures/04_relations.png)

The end-to-end system flagged 32 pairs as revised in 2024. Sixteen matched entries in the archived review and 16 await review. The archive marked seven of the 16 matched pairs as revisions. For four of those seven, the highlighted evidence supported the change. These counts apply to the subset covered by the archived review. [Revision audit](outputs/figures/05_revision_audit.csv)

## 12. Topic Forecast Results

Carrying forward the previous election's topic shares gave lower 2024 total variation distance than the fixed half-trend for all five evaluated parties. The mean distances were 0.186 for carrying forward the shares and 0.215 for the half-trend. The historical tuning selected a trend weight of zero, so the tuned forecast equals the carry-forward forecast. These scores measure error against topic shares estimated from detected pledges. The pipeline also exports exploratory 2029 estimates. [Forecast summary](outputs/pledges/forecast_summary.csv) · [Historical tuning](outputs/pledges/historical_tuning.csv)

![Forecast error by party in the 2024 backtest](outputs/figures/06_forecast.png)

## 13. Discussion

Errors can arise at several steps. A relevant pledge can be filtered out before search, lost among the retrieved candidates, or assigned the wrong relationship. Even a correctly classified revision can have an incomplete highlight. The forecast measures a different outcome, the mix of policy topics among the pledges the system detected.

### Follow-up experiments

The reranker matched the hybrid method's top-five hit counts. Both recovered 6 of 7 exploratory matches without the pledge filter and 4 of 7 with it. Across the 14 exact-text controls, all four tested search methods recovered 14 without the filter and 13 with it. [Retrieval summary](outputs/retrieval_checks/summary.csv) · [Per-query results](outputs/retrieval_checks/per_query.csv)

The mixed variant retained **1,883 training pairs** from 1,889 candidates. Its 419 revised pairs comprised 413 quantity edits, five scope edits, and one policy instrument edit. The historical mining step returned zero candidates. The split audit found zero remaining matches under its checks after excluding six pairs for similar wording. The mixed variant also changes the quantity rules and training pool, so the comparison does not isolate the effect of the six extra edits. [Training counts](outputs/training_ablation/training_counts.csv) · [Split audit](outputs/training_ablation/mixed_revisions_training_pairs_split_audit.csv) · [Historical candidates](outputs/training_ablation/historical_candidates.csv)

| Method | Pilot macro-F1 (37 pairs) | Test macro-F1 (35 pairs) | Revisions identified out of 3 on test | Test pairs predicted revised | Control macro-F1 (16 pairs) |
|---|---:|---:|---:|---:|---:|
| Screened MiniLM | 0.710 | 0.592 | 0 | 1 | 0.515 |
| Mixed-revision MiniLM | 0.835 | 0.586 | 0 | 0 | 0.457 |
| Qwen | 0.661 | 0.560 | 1 | 8 | 0.390 |
| DeBERTa | 0.340 | 0.313 | 2 | 16 | 0.186 |

The mixed variant improved macro-F1 on the pilot panel. Its score was slightly lower on the test panel, where both MiniLM variants missed all three reference revisions. Qwen and DeBERTa found some revisions by assigning the revised label more often. On the constructed controls, Qwen found all six revisions while assigning the label to 14 of 16 pairs. [Model metrics](outputs/baseline_comparison/metrics.csv) · [Results by control category](outputs/baseline_comparison/control_results.csv)

## 14. Limitations

The relation test has three scored natural revisions and uses judgments from a language model. Its scores describe agreement on those examples. The main training set represents revisions through synthetic changes to quantities. The split check can miss paraphrases even when it finds zero matches by ID, exact text, and the defined similarity threshold. The forecast has few election cycles per party, and OCR and missing English editions limit source coverage.

## 15. Future Improvements

A subsequent study should build an independently human-annotated benchmark of natural revisions and retrieval queries. Two readers would code the same held-out pairs using the source PDFs and the codebook, record the words that change the obligation, and resolve disagreements after recording their initial judgments. The study would report inter-annotator agreement and class counts, document any oversampling of likely revisions, and audit OCR and pledge detection against the corresponding source pages. These data would support a stronger estimate of how well the system finds and explains material changes across parties.

## 16. Conclusion

In the saved challenge, filtering later statements reduced how often search recovered an earlier match. The screened five-epoch model agreed with 30 of 35 test reference decisions, but found none of the three revisions at the working cutoff. The aggregate score reflects a panel dominated by pairs labelled different. The 2024 backtest favours carrying the previous election’s topic shares forward among the tested approaches. The 2029 values remain exploratory forecasts of model-assigned topic shares.

## 17. References

- Fornaciari, T., Runeson, J., Hovy, D., Thomson, R., Naurin, E., and Adhikari, P. (2021). “We will Reduce Taxes” — Identifying Election Pledges with Language Models. *Findings of ACL-IJCNLP*, 3406–3419. This study supplies the closest precedent for pledge detection in Indian and Swedish manifestos. [ACL Anthology](https://aclanthology.org/2021.findings-acl.301/)
- Thomson, R., Royed, T., Naurin, E., et al. (2017). The Fulfillment of Parties’ Election Pledges: A Comparative Study on the Impact of Power Sharing. *American Journal of Political Science*, 61(3), 527–542. This study supplies the comparative political-science background for defining election commitments. [Journal article](https://onlinelibrary.wiley.com/doi/10.1111/ajps.12313)
- Manifesto Project. *Manifesto Coding Instructions*, fourth edition. The codebook supplies the quasi-sentence concept and standard policy categories used by ManifestoBERTa. [Coding handbook](https://manifestoproject.wzb.eu/information/documents?name=handbook_v4)
- Shaar, S., Babulkov, N., Da San Martino, G., and Nakov, P. (2020). That is a Known Lie: Detecting Previously Fact-Checked Claims. *Proceedings of ACL*. Earlier-claim retrieval provides a methodological precedent for searching a collection for a prior statement. [ACL Anthology](https://aclanthology.org/2020.acl-main.332/)
- Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., and Gurevych, I. (2021). BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. *NeurIPS Datasets and Benchmarks*. BEIR motivates comparing lexical, dense and reranking approaches on the same retrieval task. [Proceedings](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/65b9eea6e1cc6bb9f0cd2a47751a186f-Abstract-round2.html)
