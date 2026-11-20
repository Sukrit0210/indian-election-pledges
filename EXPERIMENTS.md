# Follow-up experiments

Set up the project with the [README instructions](README.md#scripts). The default launcher includes the three follow-up experiments. From saved stage 10 outputs, run `py -3.14 run_all.py --start 11`. The commands below can rerun individual experiments.

## Audit the saved run

```bash
python src/11_audit_pilot.py
```

This reads the archived model outputs and writes threshold counts, confusion counts, extraction and detector counts, and a source-linked spot-check form to `outputs/pilot_audit/`. The threshold grid is descriptive. Use a prespecified threshold or choose it on development examples in a future evaluation.

## Check retrieval and reranking

```bash
python run_all.py --stop 3
python src/12_retrieval_checks.py
```

The script compares BM25, dense retrieval, reciprocal rank fusion and a frozen MS MARCO MiniLM reranker on the top 20 hybrid candidates. The pledge filter is applied to the later query. The earlier search pool stays fixed. A dropped query counts as a miss. Add `--skip-reranker` for the three first-stage scores.

The evaluation files cover two different checks. The seven exploratory predecessor judgments are subjective review cases. The 14 exact-text controls are mechanically matched same-party 2019–2024 repeats of at least 12 normalized words. Normalization ignores case and punctuation and removes numeric grouping commas. These controls test recovery of identical wording. They are not verified pledge annotations. The additional 40 later statements were selected across all five parties with 2024 coverage, before consulting detector predictions. The sample uses systematic spacing by source ID after deduplicating normalized text for each party and excluding the pilot and exact repeats. Their candidate pool is exported for future review. Those 40 statements have no relevance labels.

## Compare relation models

```bash
python src/13_compare_baselines.py --method nli
python src/13_compare_baselines.py --method both
```

The second command scores both frozen baselines. To compare them against the screened and mixed-revision MiniLM checkpoints after training, run `py -3.14 run_all.py --start 13`. The script compares the archived rules and cross-encoder with a frozen DeBERTa NLI model and a zero-shot Qwen prompt with greedy decoding and a limit of eight generated tokens. DeBERTa scores the three relation descriptions as entailment hypotheses. Qwen receives the same definitions and returns one label. Invalid replies count as abstentions. Both use the same text pairs. The code records the output label, coverage, macro-F1, revision precision and recall, and raw confusion counts. Each model also scores 16 controlled comparisons. The controls test known changes to examples constructed for this purpose. See [controlled pairs](data/review/controlled_pairs.csv).

The model revisions are `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli@0e2603d5d3d3ef9b2910814b34eebe1a2101da65` and `Qwen/Qwen2.5-1.5B-Instruct@989aa79`. The complete model IDs and checksums for the original pipeline are in [models.json](data/models.json).

## Test the effect of training data

```bash
python run_all.py --stop 6
python src/14_training_ablation.py --mine-only
python src/14_training_ablation.py --variant all
python src/13_compare_baselines.py --method both --checkpoint legacy=.cache/legacy_edits --checkpoint safe=.cache/safe_quantities --checkpoint historical=.cache/safe_plus_historical --checkpoint mixed=.cache/mixed_revisions
```

All four variants reuse the original PyTorch trainer and initialize from the same pinned checkpoint. `legacy_edits` uses the original weak and synthetic labeling scheme on the guarded historical split. `safe_quantities` keeps the original synthetic quantity strategy with calendar years and decimal amounts excluded and removes earlier rule-labeled revisions. `safe_plus_historical` adds naturally occurring historical text pairs where an aligned amount, age, deadline, or legal action differs under a conservative pattern rule. Each candidate includes the original text and PDF page for both sides. These remain weak labels. `mixed_revisions` adds controlled phrase edits for beneficiaries, scope, deadlines, and policy instruments. The trainer screens the candidate pairs against both sides of the pilot, test, and control examples by statement ID, normalized text, and close wording. Its audit files record the exclusions.

The saved follow-up run includes the retrieval reranker, the mixed-revision variant, and the frozen baselines. See the [report](REPORT.md#follow-up-experiments) for the results and the [comparison table](outputs/baseline_comparison/metrics.csv) for the full set of model scores.
