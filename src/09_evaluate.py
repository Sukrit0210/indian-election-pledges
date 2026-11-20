"""Score reviewed relations, two-sided evidence spans and independently searched query matches."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from common import OUT, REVIEW, read, write
from relations import words
from sklearn.metrics import f1_score

CLASSES = ["same", "revised", "different"]


def span_f1(a, b):
    x = Counter(words(a))
    y = Counter(words(b))
    return 2 * sum((x & y).values()) / (x.total() + y.total()) if x and y else 0.0


def panel(file, predictions, name):
    rows = read(file)
    results = []
    seen = set()
    for r in rows:
        if name == "independent" and not r["relation"].strip():
            continue
        if r["relation"] not in CLASSES + ["unclear"]:
            raise ValueError(f"Unknown decision for {r['pair_id']}")
        if name == "independent" and r["relation"] == "revised":
            for side in ["earlier", "later"]:
                if (not r[side + "_evidence"] and not r["notes"].strip()) or r[
                    side + "_evidence"
                ] not in r[side + "_text"]:
                    raise ValueError(f"Enter a verbatim {side} span for {r['pair_id']}")
        key = (
            r["party"],
            r["earlier_text"].lower().strip(),
            r["later_text"].lower().strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        p = predictions[r["pair_id"]]
        out = dict(
            pair_id=r["pair_id"],
            party=r["party"],
            reference=r["relation"],
            rule=p["rule_label"],
            cross_encoder=p["model_label"],
            notes=r["notes"],
        )
        for side in ["earlier", "later"]:
            gold = r[side + "_evidence"]
            guess = p[side + "_evidence"]
            out["reference_" + side] = gold
            out["model_" + side] = guess
            out[side + "_span_f1"] = (
                span_f1(gold, guess) if r["relation"] == "revised" else ""
            )
        results.append(out)
    scored = [r for r in results if r["reference"] in CLASSES]
    if not scored:
        raise ValueError("At least one determinate reviewed pair is required.")
    write(OUT / f"{name}_decisions.csv", results)
    metrics = []
    for model in ["always_different", "rule", "cross_encoder"]:
        y = [r["reference"] for r in scored]
        z = ["different" if model == "always_different" else r[model] for r in scored]
        scores = f1_score(y, z, labels=CLASSES, average=None, zero_division=0)
        metrics.append(
            dict(
                method=model,
                scored=len(y),
                abstentions=z.count("unclear"),
                macro_f1=mean(scores),
                correct=sum(a == b for a, b in zip(y, z)),
                **{"f1_" + l: v for l, v in zip(CLASSES, scores)},
            )
        )
    write(OUT / f"{name}_metrics.csv", metrics)
    rev = [r for r in scored if r["reference"] == "revised"]
    return dict(
        rows=len(rows),
        unique_pairs=len(results),
        reference_counts=dict(Counter(r["reference"] for r in results)),
        metrics=metrics,
        reference_revisions=len(rev),
        correct_revision_labels=sum(r["cross_encoder"] == "revised" for r in rev),
        both_spans_overlap=sum(
            r["earlier_span_f1"] > 0 and r["later_span_f1"] > 0 for r in rev
        ),
        mean_evidence_f1=mean(
            (r["earlier_span_f1"] + r["later_span_f1"]) / 2 for r in rev
        )
        if rev
        else None,
    )


def evaluate(independent=None):
    pred = {r["pair_id"]: r for r in read(OUT / "pair_predictions.csv")}
    pred.update({r["pair_id"]: r for r in read(OUT / "panel_predictions.csv")})
    if independent:
        return panel(independent, pred, "independent")
    from lookup import Search, ablation

    ablation(Search())
    result = {
        s: panel(REVIEW / (s + "_reference.csv"), pred, s) for s in ["pilot", "test"]
    }
    pool = defaultdict(list)
    for r in read(OUT / "retrieval_pool.csv"):
        pool[r["later_id"]].append(r)
    best = {r["later_id"]: r for r in read(OUT / "best_matches.csv")}
    with (REVIEW / "retrieval_reference.tsv").open(encoding="utf-8") as f:
        refs = list(csv.DictReader(f, delimiter="\t"))
    out = []
    for ref in refs:
        later = ref["later_id"]
        targets = set(filter(None, ref["earlier_ids"].split("|")))
        row = dict(**ref, detector_selected=int(later in best))
        for method in ["bm25", "dense", "fusion"]:
            row[method + "_hit_at_5"] = (
                int(
                    any(
                        p["earlier_id"] in targets and int(p[method + "_rank"]) <= 5
                        for p in pool[later]
                    )
                )
                if targets
                else ""
            )
        pick = best.get(later, {})
        row["pipeline_decision"] = pick.get("decision", "upstream_filter")
        row["pipeline_selected_id"] = pick.get("earlier_id", "")
        row["pipeline_correct_id"] = (
            int(row["pipeline_selected_id"] in targets) if targets else ""
        )
        out.append(row)
    known = [r for r in out if r["earlier_ids"]]
    write(OUT / "retrieval_challenge.csv", out)
    result["retrieval"] = dict(
        queries=len(out),
        known_match=len(known),
        passed_detector=sum(r["detector_selected"] for r in known),
        **{
            m + "_hits": sum(r[m + "_hit_at_5"] for r in known)
            for m in ["bm25", "dense", "fusion"]
        },
        selected_correct=sum(r["pipeline_correct_id"] for r in known),
    )
    (OUT / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--independent-review", type=Path)
    print(json.dumps(evaluate(p.parse_args().independent_review), indent=1))
