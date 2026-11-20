"""Audit the reference run from saved predictions and source metadata."""
from collections import Counter
from pathlib import Path
import random

from common import ROOT, read, write
from relations import words

LABELS = ("same", "revised", "different")
THRESHOLDS = (0.50, 0.60, 0.65, 0.70, 0.80, 0.90)


def threshold_audit(root, output):
    predictions = {r["pair_id"]: r for r in read(root / "comparison/panel_predictions.csv")}
    counts, summary = [], []
    for split in ("pilot", "test"):
        panel = [r for r in read(root / f"comparison/{split}_decisions.csv")
                 if r["reference"] in LABELS]
        for threshold in THRESHOLDS:
            matrix = Counter()
            for row in panel:
                score = predictions[row["pair_id"]]
                guess = max(LABELS, key=lambda label: float(score["p_" + label]))
                if float(score["p_" + guess]) < threshold:
                    guess = "unclear"
                matrix[row["reference"], guess] += 1
            covered = sum(matrix[a, b] for a in LABELS for b in LABELS)
            correct = sum(matrix[a, a] for a in LABELS)
            revised = sum(matrix["revised", b] for b in (*LABELS, "unclear"))
            predicted_revised = sum(matrix[a, "revised"] for a in LABELS)
            summary.append(dict(split=split, threshold=threshold, evaluated=len(panel),
                answered=covered, coverage=covered / len(panel), correct=correct,
                actual_revised=revised, predicted_revised=predicted_revised,
                correct_revised=matrix["revised", "revised"]))
            counts.extend(dict(split=split, threshold=threshold, reference=a, prediction=b,
                               count=matrix[a, b])
                          for a in LABELS for b in (*LABELS, "unclear"))
    write(output / "threshold_sensitivity.csv", summary)
    write(output / "relation_confusion_counts.csv", counts)


def corpus_audit(root, output):
    source = "statement_id"
    statements = {r[source]: r for r in read(root / "data/quasi_sentences.csv")}
    passages = {r["passage_id"]: r for r in read(root / "data/master_passages.csv")}
    scores = read(root / "pledges/pledge_scores.csv")
    status = Counter()
    for row in scores:
        unit = statements[row[source]]
        status[unit["document_id"], unit["extraction"], row["weak_label"], row["prediction"]] += 1
    write(output / "extraction_and_detector_counts.csv",
          [dict(document_id=k[0], extraction=k[1], weak_label=k[2], prediction=k[3], count=v)
           for k, v in sorted(status.items())])

    # A stratified spot-check form, with source pages and detector decisions.
    rng = random.Random(2024)
    candidates = {}
    for row in scores:
        unit = statements[row[source]]
        group = unit["extraction"], row["weak_label"], row["prediction"]
        candidates.setdefault(group, []).append(row)
    sample = []
    for group, items in sorted(candidates.items()):
        for row in rng.sample(items, min(len(items), 5)):
            unit = statements[row[source]]
            parent = passages[unit["passage_id"]]
            sample.append(dict(statement_id=row[source], document_id=unit["document_id"],
                page=unit["page"], source_url=parent["source_url"], extraction=group[0],
                weak_label=group[1], detector_prediction=group[2],
                word_count=len(words(unit["text"])), text=unit["text"],
                source_transcription="", pledge_review="", notes=""))
    write(output / "ocr_and_detector_spotcheck.csv", sample)


if __name__ == "__main__":
    reference = ROOT / "reference"
    destination = ROOT / "outputs" / "pilot_audit"
    threshold_audit(reference, destination)
    corpus_audit(reference, destination)
