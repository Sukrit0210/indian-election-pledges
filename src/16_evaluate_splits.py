"""Fit fold-specific detectors and comparison models and score the frozen panels."""
from collections import Counter
import importlib

import numpy as np
from sklearn.metrics import f1_score
import torch

from common import CACHE, OUTPUTS, read, write
from lookup import Search
from model_files import model_dir
from pair_model import LABELS
from relations import rule
from split_audit import Holdout

DEST = OUTPUTS / "expanded_evaluation"
METHODS = ("bm25", "dense", "fusion")


def score_detector(fold, directory, search, source_by_id, weak_labels, holdout):
    eligible = {row["statement_id"] for row in read(directory / "eligible_statements.csv")}
    party, cutoff = fold["held_party"], int(fold["training_year_max"])
    for sid in eligible:
        row = source_by_id[sid]
        if row["party"] == party or int(row["year"]) > cutoff or sid in holdout.ids or holdout.text_match(row["text"]):
            raise ValueError(f"{fold['fold']}: invalid detector training statement {sid}. Rebuild stage 14.")
    indices = np.array([i for i, row in enumerate(search.rows)
                        if row["statement_id"] in eligible and weak_labels[i] >= 0])
    if len(indices) == 0 or len(set(weak_labels[indices])) != 2:
        raise ValueError(f"{fold['fold']}: both detector classes must be present in training.")
    fit_detector = importlib.import_module("03_detect_pledges").fit_detector
    _, _, probabilities = fit_detector(search.rows, search.emb, weak_labels, indices)
    scored = []
    for row, weak, probability in zip(search.rows, weak_labels, probabilities, strict=True):
        in_test = row["party"] == party if party else row["year"] == "2024"
        if in_test:
            scored.append(dict(statement_id=row["statement_id"], party=row["party"], year=row["year"],
                               weak_label={0: "non_pledge", 1: "pledge", -1: "uncertain"}[int(weak)],
                               pledge_probability=float(probability),
                               prediction="pledge" if probability >= .5 else "non_pledge"))
    write(directory / "detector_predictions.csv", scored)
    known = [r for r in scored if r["weak_label"] != "uncertain"]
    return probabilities, dict(fold=fold["fold"], train_labels=len(indices), test_statements=len(scored),
                               weak_labelled=len(known),
                               agreeing=sum(r["weak_label"] == r["prediction"] for r in known))


def retrieval_metrics(fold, directory, queries, search, probabilities):
    results = []
    for query in queries:
        candidates = search.query(query["later_id"])
        targets = set(query["earlier_ids"].split("|"))
        passed = probabilities[search.i[query["later_id"]]] >= .5
        for method in METHODS:
            rank = min(r[method + "_rank"] for r in candidates if r["earlier_id"] in targets)
            results.append(dict(fold=fold, party=query["party"], year=query["year"], query_id=query["later_id"],
                                method=method, first_rank=rank, hit_at_5=int(rank <= 5),
                                passed_detector=int(passed), filtered_hit_at_5=int(passed and rank <= 5)))
    fields = ["fold", "party", "year", "query_id", "method", "first_rank", "hit_at_5",
              "passed_detector", "filtered_hit_at_5"]
    write(directory / "retrieval_predictions.csv", results, fields=fields)
    return [dict(fold=fold, method=method, queries=len(queries),
                 hits_at_5=sum(r["hit_at_5"] for r in results if r["method"] == method),
                 filtered_hits_at_5=sum(r["filtered_hit_at_5"] for r in results if r["method"] == method))
            for method in METHODS]


def metrics(rows, fold):
    results = []
    for dataset in ("natural", "controlled"):
        for method in sorted({r["method"] for r in rows if r["dataset"] == dataset}):
            subset = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
            for category in ("all", *sorted({r["category"] for r in subset})):
                group = subset if category == "all" else [r for r in subset if r["category"] == category]
                scored = [r for r in group if r["reference"] in LABELS]
                truth, guess = [r["reference"] for r in scored], [r["prediction"] for r in scored]
                support = Counter(truth)
                labels = [label for label in LABELS if support[label]]
                f1 = f1_score(truth, guess, labels=labels, average="macro", zero_division=0) if labels else ""
                results.append(dict(fold=fold, dataset=dataset, method=method, category=category,
                                    pairs=len(group), scored=len(scored), reference_uncertain=len(group)-len(scored),
                                    same_support=support["same"], revised_support=support["revised"],
                                    different_support=support["different"], abstentions=guess.count("unclear"),
                                    correct=sum(a == b for a, b in zip(truth, guess)),
                                    revisions_found=sum(a == b == "revised" for a, b in zip(truth, guess)),
                                    predicted_revised=guess.count("revised"),
                                    labels_scored="|".join(labels),
                                    macro_f1=f1 if len(labels) == len(LABELS) else "",
                                    macro_f1_available_labels=f1))
    return results


def comparison_predictions(fold, directory, panels, checkpoint):
    baselines = importlib.import_module("13_compare_baselines")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for dataset, cases in panels:
        answers = {
            "screened_minilm": baselines.trained(cases, checkpoint, device, .65),
            "always_different": ["different"] * len(cases),
        }
        if dataset == "natural":
            answers["rule"] = [rule(r["earlier_text"], r["later_text"], float(r["cosine"])) for r in cases]
        # Compare frozen baselines on the main temporal panels. The strict split has the same panel.
        if fold == "temporal":
            answers["deberta"] = baselines.nli(cases, device)
            answers["qwen"] = baselines.llm(cases, device)
        for method, labels in answers.items():
            rows.extend(dict(fold=fold, dataset=dataset, method=method, pair_id=row["pair_id"],
                             party=row["party"], year=row["year"], category=row["category"],
                             reference=row["reference"], prediction=label)
                        for row, label in zip(cases, labels, strict=True))
    write(directory / "comparison_predictions.csv", rows)
    return rows


def main():
    search = Search()
    sources = {r["statement_id"]: r for r in search.rows}
    by_id = {row["statement_id"]: row["weak_label"] for row in read(OUTPUTS / "pledges/pledge_scores.csv")}
    weak_labels = np.array([{"non_pledge": 0, "pledge": 1, "abstain": -1}[by_id[row["statement_id"]]]
                            for row in search.rows])
    trainer = importlib.import_module("07_train_cross_encoder")
    base = model_dir("comparison")
    all_rows, all_metrics, detector_rows, retrieval_rows = [], [], [], []
    lopo_rows = []
    folds = read(DEST / "folds.csv")
    progress = [dict(fold=fold["fold"], status="pending") for fold in folds]
    write(DEST / "run_progress.csv", progress)
    for index, fold in enumerate(folds):
        name = fold["fold"]
        directory = DEST / name
        print(f"Evaluating {name}", flush=True)
        progress[index]["status"] = "running"
        write(DEST / "run_progress.csv", progress)
        natural = read(directory / "natural_pairs.csv")
        controlled = read(directory / "controlled_pairs.csv")
        retrieval = read(directory / "retrieval_queries.csv")
        holdout = Holdout(extra_pairs=[*natural, *controlled], extra_queries=retrieval, sources=sources)
        probabilities, detector_summary = score_detector(fold, directory, search, sources, weak_labels, holdout)
        detector_rows.append(detector_summary)
        retrieval_rows.extend(retrieval_metrics(name, directory, retrieval, search, probabilities))
        training = read(directory / "training_pairs.csv")
        eligible = {r["statement_id"] for r in read(directory / "eligible_statements.csv")}
        if any(r["source_a"] not in eligible or r["source_b"] not in eligible for r in training):
            raise ValueError(f"{name}: training pair outside the fold. Rebuild stage 14.")
        if any(holdout.reason(row) for row in training):
            raise ValueError(f"{name}: overlap in frozen training pairs. Rebuild stage 14.")
        checkpoint = CACHE / "expanded_evaluation" / name
        trainer.run(base, checkpoint, training, directory / "fitted_training_pairs.csv", holdout=holdout)
        rows = comparison_predictions(name, directory, (("natural", natural), ("controlled", controlled)), checkpoint)
        all_rows.extend(rows)
        all_metrics.extend(metrics(rows, name))
        if fold["held_party"]:
            lopo_rows.extend(rows)
        write(DEST / "comparison_predictions.csv", all_rows)
        write(DEST / "comparison_metrics.csv", all_metrics)
        write(DEST / "detector_summary.csv", detector_rows)
        write(DEST / "retrieval_summary.csv", retrieval_rows)
        progress[index]["status"] = "complete"
        write(DEST / "run_progress.csv", progress)
    if lopo_rows:
        write(DEST / "comparison_metrics.csv", all_metrics + metrics(lopo_rows, "lopo_pooled"))


if __name__ == "__main__":
    main()
