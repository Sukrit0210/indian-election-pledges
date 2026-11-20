"""Prepare an unscored, model-blind 2024 review set. All decisions stay blank."""

import random
import re
from collections import defaultdict

from common import DATA, OUT, REVIEW, read, write
from relations import MEASURE


def make():
    docs = {r["statement_id"]: r for r in read(DATA / "quasi_sentences.csv")}
    parents = {r["passage_id"]: r["text"] for r in read(DATA / "master_passages.csv")}
    norm = lambda t: " ".join(re.findall(r"\w+", t.lower()))
    seen = set()
    for file in ["test_reference.csv", "flagged_2024_audit.csv"]:
        for r in read(REVIEW / file):
            seen.add((r["party"], norm(r["earlier_text"]), norm(r["later_text"])))
    bins = defaultdict(list)
    for r in read(OUT / "retrieval_pool.csv"):
        if r["year"] != "2024" or int(r["fusion_rank"]) > 5:
            continue
        a = docs[r["earlier_id"]]["text"]
        b = docs[r["later_id"]]["text"]
        if not (5 <= len(a.split()) <= 85 and 5 <= len(b.split()) <= 85):
            continue
        if (r["party"], norm(a), norm(b)) in seen:
            continue
        x = MEASURE.findall(a)
        y = MEASURE.findall(b)
        bucket = (
            "measure"
            if x and y and x != y and float(r["cosine"]) >= 0.55
            else "close"
            if float(r["cosine"]) >= 0.72
            else "other"
        )
        bins[r["party"], bucket].append(r)
    rng = random.Random(1072)
    form = []
    design = []
    for (party, bucket), items in sorted(bins.items()):
        rng.shuffle(items)
        accepted = 0
        for r in items:
            a, b = [docs[r[k]] for k in ("earlier_id", "later_id")]
            surface = party, norm(a["text"]), norm(b["text"])
            if surface in seen:
                continue
            seen.add(surface)
            form.append(
                dict(
                    pair_id=r["pair_id"],
                    party=party,
                    earlier_id=r["earlier_id"],
                    later_id=r["later_id"],
                    earlier_text=a["text"],
                    later_text=b["text"],
                    earlier_context=parents[a["passage_id"]],
                    later_context=parents[b["passage_id"]],
                    relation="",
                    earlier_evidence="",
                    later_evidence="",
                    notes="",
                )
            )
            design.append(
                dict(pair_id=r["pair_id"], party=party, sampling_stratum=bucket)
            )
            accepted += 1
            if accepted == 10:
                break
    rng.shuffle(form)
    write(OUT / "fresh_human_review.csv", form)
    write(OUT / "fresh_sampling_design.csv", design)
    print(len(form), "unscored pairs")


if __name__ == "__main__":
    make()
