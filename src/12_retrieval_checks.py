"""Compare retrieval methods on exact repeats and the fixed pilot challenge."""
import argparse
import csv
import re

from common import OUTPUTS, REVIEW, read, write
from lookup import Search
from model_files import model_dir

RANKS = ("bm25", "dense", "fusion", "reranker")
DEST = OUTPUTS / "retrieval_checks"


def canonical(text):
    text = re.sub(r"(?<=\d),(?=\d)", "", text.casefold())
    return " ".join(re.findall(r"\w+", text))


class Reranker:
    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        path = model_dir("comparison")
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path, local_files_only=True
        ).to(self.device).eval()

    def rank(self, query, candidates):
        pool = sorted(candidates, key=lambda row: row["fusion_rank"])[:20]
        inp = self.tokenizer([query] * len(pool), [r["text"] for r in pool], padding=True,
                             truncation=True, max_length=256, return_tensors="pt")
        with self.torch.inference_mode():
            scores = self.model(**inp.to(self.device)).logits.flatten().tolist()
        ranked = sorted(zip(pool, scores, strict=True), key=lambda item: -item[1])
        for rank, (row, _) in enumerate(ranked, 1):
            row["reranker_rank"] = rank


def check(search, benchmark, name, detected, reranker):
    out, summary = [], []
    methods = RANKS if reranker else RANKS[:-1]
    for ref in benchmark:
        targets = set(ref["earlier_ids"].split("|")) - {""}
        if not targets:
            continue
        later_id = ref["later_id"]
        candidates = search.query(later_id)
        later = search.rows[search.i[later_id]]
        if reranker:
            reranker.rank(later["text"], candidates)
        for method in methods:
            ranks = [r[method + "_rank"] for r in candidates
                     if r["earlier_id"] in targets and method + "_rank" in r]
            first_rank = min(ranks, default=None)
            for mode in ("all_queries", "pledge_filter"):
                enabled = mode == "all_queries" or later_id in detected
                out.append(dict(benchmark=name, party=later["party"], later_id=later_id,
                                earlier_ids=ref["earlier_ids"], method=method, mode=mode,
                                rank=first_rank if enabled and first_rank else "",
                                hit_at_5=int(enabled and first_rank is not None and first_rank <= 5)))
    for method in methods:
        for mode in ("all_queries", "pledge_filter"):
            cohort = [r for r in out if r["method"] == method and r["mode"] == mode]
            hits = sum(r["hit_at_5"] for r in cohort)
            summary.append(dict(benchmark=name, method=method, mode=mode, queries=len(cohort),
                                hits_at_5=hits, hit_rate_at_5=hits / len(cohort) if cohort else ""))
    return out, summary


def export_query_sample(search):
    queries = read(REVIEW / "unlabelled_query_sample.csv")
    candidates = []
    for row in queries:
        statement_id = row["later_id"]
        for earlier in search.query(statement_id):
            if min(earlier[m + "_rank"] for m in RANKS[:-1]) <= 5:
                candidates.append(dict(later_id=statement_id, **earlier))
    write(DEST / "unlabelled_candidate_pool.csv", candidates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-reranker", action="store_true")
    args = parser.parse_args()
    search = Search()
    detected = {r["statement_id"] for r in read(OUTPUTS / "pledges/pledge_scores.csv")
                if r["prediction"] == "pledge"}
    with (REVIEW / "retrieval_reference.tsv").open(encoding="utf-8") as stream:
        pilot = list(csv.DictReader(stream, delimiter="\t"))
    controls = read(REVIEW / "exact_text_controls.csv")
    for r in controls:
        later = search.rows[search.i[r["later_id"]]]
        for earlier_id in r["earlier_ids"].split("|"):
            earlier = search.rows[search.i[earlier_id]]
            if canonical(later["text"]) != canonical(earlier["text"]):
                raise ValueError("This corpus does not match the frozen retrieval controls")
    reranker = None if args.skip_reranker else Reranker()
    outcomes, summary = [], []
    for benchmark, name in [(pilot, "exploratory_reference"), (controls, "exact_text_control")]:
        results, metrics = check(search, benchmark, name, detected, reranker)
        outcomes.extend(results)
        summary.extend(metrics)
    write(DEST / "per_query.csv", outcomes)
    write(DEST / "summary.csv", summary)
    export_query_sample(search)


if __name__ == "__main__":
    main()
