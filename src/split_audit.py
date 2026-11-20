"""Keep evaluation statements and close wording out of relation training."""
from collections import Counter
from functools import lru_cache
import re

from common import REVIEW, read, write
from relations import words


def surface(text):
    return " ".join(words(re.sub(r"(?<=\d),(?=\d)", "", text)))


def shingles(text):
    tokens = text.split()
    return set(zip(tokens, tokens[1:], tokens[2:]))


class Holdout:
    def __init__(self):
        self.ids = set()
        self.texts = set()
        for filename in ("pilot_reference.csv", "test_reference.csv", "controlled_pairs.csv"):
            for row in read(REVIEW / filename):
                for side in ("earlier", "later"):
                    if row.get(side + "_id"):
                        self.ids.add(row[side + "_id"])
                    self.texts.add(surface(row[side + "_text"]))
        self.reference_shingles = [shingles(t) for t in self.texts if len(t.split()) >= 5]

    @lru_cache(maxsize=32768)
    def text_match(self, text):
        normalized = surface(text)
        if normalized in self.texts:
            return "exact_text"
        if len(normalized.split()) >= 5:
            candidate = shingles(normalized)
            for reference in self.reference_shingles:
                # The size ratio bounds the highest possible Jaccard similarity.
                if min(len(candidate), len(reference)) < 0.8 * max(len(candidate), len(reference)):
                    continue
                if len(candidate & reference) >= 0.8 * len(candidate | reference):
                    return "similar_text"
        return ""

    def reason(self, row):
        if row["source_a"] in self.ids or row["source_b"] in self.ids:
            return "statement_id"
        return self.text_match(row["earlier_text"]) or self.text_match(row["later_text"])

    def screen(self, examples, output):
        kept, excluded = [], []
        for row in examples:
            reason = self.reason(row)
            if reason:
                excluded.append(dict(reason=reason, **row))
            else:
                kept.append(row)
        fields = ["reason", "label", "origin", "source_a", "source_b", "earlier_text", "later_text"]
        write(output.with_name(output.stem + "_excluded.csv"), excluded, fields=fields)
        remaining = sum(bool(self.reason(row)) for row in kept)
        counts = Counter(row["reason"] for row in excluded)
        write(output.with_name(output.stem + "_split_audit.csv"), [dict(
            candidates=len(examples), retained=len(kept), excluded=len(excluded),
            by_id=counts["statement_id"], by_exact_text=counts["exact_text"],
            by_similar_text=counts["similar_text"], remaining_matches=remaining,
            reference_files="pilot_reference.csv,test_reference.csv,controlled_pairs.csv",
            similarity="word trigram Jaccard >= 0.80, minimum 5 tokens",
        )])
        if remaining or len({r["label"] for r in kept}) != 3:
            raise ValueError("Check the split audit and retained class counts before training.")
        return kept
