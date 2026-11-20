"""Train paired variants to test date-safe edits and historical candidate supervision."""
import argparse
from collections import Counter
import importlib
import re

from common import CACHE, DATA, OUT, OUTPUTS, REVIEW, read, write
from model_files import model_dir
from relations import MEASURE, norm, words
from weak_pairs import training_examples

DEST = OUTPUTS / "training_ablation"
VARIANTS = ("legacy_edits", "safe_quantities", "safe_plus_historical")
LEGAL_ACTION = re.compile(r"\b(enact|repeal|amend|implement)\b", re.I)


def surface(text):
    return " ".join(words(re.sub(r"(?<=\d),(?=\d)", "", text)))


def held_out_statements():
    ids, texts = set(), set()
    for subset in ("pilot", "test"):
        for row in read(REVIEW / f"{subset}_reference.csv"):
            for side in ("earlier", "later"):
                ids.add(row[side + "_id"])
                texts.add(surface(row[side + "_text"]))
    return ids, texts


def historical_change(earlier, later):
    # Accept a candidate only when the surrounding commitment stays aligned.
    before = list(MEASURE.finditer(earlier))
    after = list(MEASURE.finditer(later))
    if len(before) == len(after) == 1:
        a, b = before[0], after[0]
        if (a.lastgroup == b.lastgroup and a.lastgroup != "date"
                and norm(a.group()) != norm(b.group())
                and surface(earlier[:a.start()]) == surface(later[:b.start()])
                and surface(earlier[a.end():]) == surface(later[b.end():])):
            return a.lastgroup, a.group(), b.group()
    a, b = LEGAL_ACTION.search(earlier), LEGAL_ACTION.search(later)
    if (a and b and a.group().lower() != b.group().lower()
            and re.search(r"\b(?:Act|Bill|Regulations?)\b", earlier)
            and surface(earlier[:a.start()]) == surface(later[:b.start()])
            and surface(earlier[a.end():]) == surface(later[b.end():])):
        return "legal_action", a.group(), b.group()


def mine(pool, statements, pledge_votes):
    found, seen = [], set()
    for pair in pool:
        a, b = pair["earlier_id"], pair["later_id"]
        if int(pair["fusion_rank"]) > 5 or a not in pledge_votes or b not in pledge_votes:
            continue
        earlier, later = statements[a], statements[b]
        change = historical_change(earlier["text"], later["text"])
        text_key = surface(earlier["text"]), surface(later["text"])
        if not change or text_key in seen:
            continue
        seen.add(text_key)
        kind, before, after = change
        found.append(dict(label="revised", earlier_text=earlier["text"], later_text=later["text"],
            origin="historical_aligned_" + kind, source_a=a, source_b=b,
            earlier_document=earlier["document_id"], earlier_page=earlier["page"],
            later_document=later["document_id"], later_page=later["page"],
            earlier_change=before, later_change=after))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), default="all")
    parser.add_argument("--mine-only", action="store_true")
    args = parser.parse_args()
    excluded_ids, excluded_text = held_out_statements()
    statements = {r["statement_id"]: r for r in read(DATA / "quasi_sentences.csv")
                  if int(r["year"]) <= 2019 and r["statement_id"] not in excluded_ids
                  and surface(r["text"]) not in excluded_text}
    pool = [r for r in read(OUT / "retrieval_pool.csv")
            if r["earlier_id"] in statements and r["later_id"] in statements]
    pledge_votes = {r["statement_id"] for r in read(OUTPUTS / "pledges/pledge_scores.csv")
                    if r["weak_label"] == "pledge"}
    mined = mine(pool, statements, pledge_votes)
    fields = ["label", "earlier_text", "later_text", "origin", "source_a", "source_b"]
    write(DEST / "historical_candidates.csv", mined, fields=fields + [
          "earlier_document", "earlier_page", "later_document", "later_page",
          "earlier_change", "later_change"])
    print(f"Found {len(mined)} historical weak candidates")
    if args.mine_only:
        return
    train = importlib.import_module("07_train_cross_encoder").run
    count_rows = []
    for name in VARIANTS if args.variant == "all" else (args.variant,):
        safe = name != "legacy_edits"
        examples = training_examples(pool, statements, safe_quantities=safe)
        if safe:
            examples = [r for r in examples if not (r["origin"] == "historical_rule" and r["label"] == "revised")]
        if name == "safe_plus_historical":
            seen = {(r["earlier_text"], r["later_text"]) for r in examples}
            examples.extend({field: row[field] for field in fields} for row in mined
                            if (row["earlier_text"], row["later_text"]) not in seen)
        examples = [r for r in examples if surface(r["earlier_text"]) not in excluded_text
                    and surface(r["later_text"]) not in excluded_text]
        counts = Counter((r["label"], r["origin"]) for r in examples)
        count_rows.extend(dict(variant=name, label=label, origin=origin, count=count)
                          for (label, origin), count in sorted(counts.items()))
        write(DEST / "training_counts.csv", count_rows)
        if len({r["label"] for r in examples}) != 3:
            raise ValueError(f"{name} has no examples for at least one label")
        train(model_dir("comparison"), CACHE / name, examples, DEST / f"{name}_training_pairs.csv")


if __name__ == "__main__":
    main()
