"""Build quasi-sentences and preserve their parent passage offsets."""

import argparse
import re
import shutil
from pathlib import Path

import pysbd
from boundaries import ACTION_START, WORDS, spans
from common import DATA, ROOT, read, write

SEG = pysbd.Segmenter(language="en", clean=False, char_span=True)


def build(source):
    master = read(source)
    units = []
    for passage in master:
        text = passage["text"]
        pid = passage["passage_id"]
        method = passage["extraction"]
        meta = {field: passage[field] for field in ("document_id", "party", "year", "page")}
        seq = 0
        for sentence in SEG.segment(text):
            for a, b in spans(sentence.sent):
                start = sentence.start + a
                end = sentence.start + b
                while start < end and text[start].isspace():
                    start += 1
                while start < end and text[end - 1].isspace():
                    end -= 1
                value = text[start:end]
                if not value:
                    continue
                seq += 1
                count = len(WORDS.findall(value))
                flags = []
                if not 5 <= count <= 90:
                    flags.append("length")
                if value[:1].islower() or not re.search(r'[.!?][”’"]?$', value):
                    flags.append("possible_fragment")
                if ";" in value:
                    flags.append("check_clause")
                if (
                    value.istitle()
                    and count < 12
                    and not ACTION_START.match(value)
                ):
                    flags.append("possible_heading")
                if (
                    count < 3 or "possible_heading" in flags
                ) and not ACTION_START.match(value):
                    continue
                units.append(
                    dict(
                        statement_id=f"{pid}-q{seq:03}",
                        passage_id=pid,
                        **meta,
                        start_char=start,
                        end_char=end,
                        text=value,
                        word_count=count,
                        review_flags="|".join(flags),
                        extraction=method,
                    )
                )
    mapping = {r["passage_id"]: r["text"] for r in master}
    assert len(mapping) == len(master) and len(
        set(r["statement_id"] for r in units)
    ) == len(units)
    assert all(
        mapping[r["passage_id"]][r["start_char"] : r["end_char"]] == r["text"]
        for r in units
    )
    DATA.mkdir(parents=True, exist_ok=True)
    destination = DATA / "master_passages.csv"
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)
    write(DATA / "quasi_sentences.csv", units)
    print(f"Saved {len(units):,} quasi-sentences from {len(master):,} passages")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passages", type=Path, default=ROOT / "reference/data/master_passages.csv")
    build(parser.parse_args().passages)
