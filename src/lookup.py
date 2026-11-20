"""Retrieve earlier statements by later ID, including statements skipped by the upstream pledge detector."""

import argparse
import csv
from collections import defaultdict

import numpy as np
from common import DATA, OUT, REVIEW, VECTORS, fingerprint, read, write
from sklearn.feature_extraction.text import CountVectorizer


class Search:
    def __init__(self):
        self.rows = read(DATA / "quasi_sentences.csv")
        self.i = {r["statement_id"]: j for j, r in enumerate(self.rows)}
        with np.load(VECTORS) as blob:
            if str(blob["fingerprint"]) != fingerprint(self.rows):
                raise ValueError("Re-run step 02 on the current corpus")
            self.emb = blob["emb"]
        self.by_doc = defaultdict(list)
        for j, r in enumerate(self.rows):
            self.by_doc[r["party"], int(r["year"])].append(j)
        self.index = {}

    def query(self, statement_id):
        j = self.i[statement_id]
        r = self.rows[j]
        key = (r["party"], int(r["year"]) - 5)
        if key not in self.by_doc:
            raise ValueError(f"No preceding manifesto in the corpus for {key}")
        idx = self.by_doc[key]
        if key not in self.index:
            vec = CountVectorizer(stop_words="english", token_pattern=r"(?u)\b\w+\b")
            x = vec.fit_transform([self.rows[a]["text"] for a in idx]).astype(float)
            df = np.asarray((x > 0).sum(0)).ravel()
            idf = np.log1p((len(idx) - df + 0.5) / (df + 0.5))
            lengths = np.asarray(x.sum(1)).ravel()
            mat = x.tocoo()
            mat.data = (
                mat.data
                * 2.2
                / (
                    mat.data
                    + 1.2 * (0.25 + 0.75 * lengths[mat.row] / max(1, lengths.mean()))
                )
                * idf[mat.col]
            )
            self.index[key] = (vec, mat.tocsr().T)
        vec, mat = self.index[key]
        q = vec.transform([r["text"]])
        q.data[:] = 1
        bm = (q @ mat).toarray()[0]
        dense = self.emb[idx] @ self.emb[j]
        ranks = lambda v: (-v).argsort(kind="stable").argsort() + 1
        rb = ranks(bm)
        rd = ranks(dense)
        rf = ranks(1 / (30 + rd) + np.where(bm > 0, 1 / (30 + rb), 0.0))
        return [
            dict(
                earlier_id=self.rows[n]["statement_id"],
                text=self.rows[n]["text"],
                bm25_rank=int(rb[k]),
                dense_rank=int(rd[k]),
                fusion_rank=int(rf[k]),
            )
            for k, n in enumerate(idx)
        ]


def ablation(search):
    with (REVIEW / "retrieval_reference.tsv").open(encoding="utf-8") as f:
        refs = list(csv.DictReader(f, delimiter="\t"))
    out = []
    for ref in refs:
        ids = set(ref["earlier_ids"].split("|")) - {""}
        candidates = search.query(ref["later_id"])
        row = dict(later_id=ref["later_id"], earlier_ids=ref["earlier_ids"])
        for m in ["bm25", "dense", "fusion"]:
            row[m + "_first_reference_rank"] = min(
                (a[m + "_rank"] for a in candidates if a["earlier_id"] in ids),
                default="",
            )
        out.append(row)
    write(OUT / "retrieval_without_detector.csv", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--later-id")
    p.add_argument("--evaluate-without-detector", action="store_true")
    a = p.parse_args()
    index = Search()
    if a.evaluate_without_detector:
        ablation(index)
    if a.later_id:
        print(index.rows[index.i[a.later_id]]["text"])
        for r in sorted(index.query(a.later_id), key=lambda x: x["fusion_rank"])[:5]:
            print(r["fusion_rank"], r["earlier_id"], r["text"])
    if not a.later_id and not a.evaluate_without_detector:
        p.error("Provide a later ID or evaluation flag")
