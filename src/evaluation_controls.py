"""Construct reproducible revision controls from source pledge statements."""
from collections import Counter
import hashlib
import re

from relations import STOP

from split_audit import surface


def stable_key(text):
    return hashlib.sha256(text.encode()).hexdigest()


# Training and evaluation use different wordings for the same types of change.
QUALIFIERS = {
    "beneficiary": (
        ("Eligibility will be limited to students.", "Eligibility will be limited to apprentices."),
        ("Only pensioners will qualify.", "Only unemployed workers will qualify."),
        r"\b(?:provid\w*|offer\w*|subsid\w*|grant\w*|assist\w*|support\w*)\b",
        r"\b(?:eligib\w*|qualif\w*|students?|apprentices?|pensioners?|unemployed|workers?)\b",
    ),
    "scope": (
        ("The commitment will apply only in rural districts.", "The commitment will apply only in urban districts."),
        ("This measure will be available exclusively in municipal areas.", "This measure will be available exclusively in village areas."),
        r"\b(?:provid\w*|establish\w*|expand\w*|launch\w*|build\w*|introduc\w*|implement\w*)\b",
        r"\b(?:villages?|rural|urban|districts?|municip\w*|regions?|states?|cities|nationwide)\b",
    ),
    "deadline": (
        ("We will complete this commitment by the middle of the term.", "We will complete this commitment by the end of the term."),
        ("Implementation will be completed before the next budget.", "Implementation will be completed before the next general election."),
        r"\b(?:implement\w*|build\w*|establish\w*|launch\w*|complet\w*|introduc\w*)\b",
        r"\b(?:before|within|deadline|by \d|years?|months?|ongoing|continue|continuing)\b",
    ),
    "obligation": (
        ("Compliance with this measure will be voluntary.", "Compliance with this measure will be compulsory."),
        ("We will let institutions choose whether to comply.", "We will require institutions to comply."),
        r"\b(?:standards?|guidelines?|register\w*|report\w*|disclos\w*|requirements?|regulat\w*|rules?)\b",
        r"\b(?:voluntary|mandatory|compulsory|enforce\w*|penalt\w*|require\w*|compliance)\b",
    ),
}


def control_pairs(rows, per_family, evaluation):
    """Use at most one original statement per pair and balance edit families by cap."""
    available = sorted(
        (row for row in rows if 5 <= int(row["word_count"]) <= 55),
        key=lambda row: stable_key(row["statement_id"]),
    )
    used_texts = set()
    for family, (train_slots, test_slots, include, exclude) in QUALIFIERS.items():
        before, after = test_slots if evaluation else train_slots
        count = 0
        for row in available:
            sid = row["statement_id"]
            text = row["text"].strip().rstrip(".;:")
            canonical = surface(text)
            if canonical in used_texts or re.search(r"[?!;:\n]|\b(?:only|except|exclusively)\b", text, re.I):
                continue
            if not re.search(include, text, re.I) or re.search(exclude, text, re.I):
                continue
            used_texts.add(canonical)
            count += 1
            base = text + ". "
            yield dict(pair_id=f"control-{family}-{sid}", earlier_id=sid, later_id=sid,
                       party=row["party"], year=row["year"], earlier_document=row["document_id"],
                       later_document=row["document_id"], earlier_page=row["page"], later_page=row["page"],
                       earlier_text=base + before, later_text=base + after, reference="revised",
                       category=family, earlier_change=before, later_change=after)
            if count >= per_family:
                break


def unchanged_controls(revisions):
    # Reuse each base within the evaluation split with alternate formatting of identical text.
    for row in revisions:
        later = row["earlier_text"].replace(" will ", " shall ", 1)
        yield dict(row, pair_id=row["pair_id"] + "-same", reference="same",
                   later_text=later, earlier_change="", later_change="")


def distinct_controls(revisions, topics, sources):
    """Pair disjoint source commitments within the same policy topic when possible."""
    action = re.compile(r"(?:we\s+(?:will|shall)\s+)?([a-z]+)\s+(.+)", re.I)
    used = set()
    for i, a in enumerate(revisions):
        a_id = a["earlier_id"]
        a_topic = topics.get(a_id)
        a_parts = action.fullmatch(sources[a_id]["text"].strip())
        if not a_topic or not a_parts or a_id in used:
            continue
        for b in revisions[i + 1:]:
            b_id = b["earlier_id"]
            if b_id in used or topics.get(b_id) != a_topic or (a["party"], a["year"]) != (b["party"], b["year"]):
                continue
            b_parts = action.fullmatch(sources[b_id]["text"].strip())
            if not b_parts or a_parts[1].lower() == b_parts[1].lower():
                continue
            lhs = set(surface(a_parts[2]).split()) - STOP
            rhs = set(surface(b_parts[2]).split()) - STOP
            if min(len(lhs), len(rhs)) < 3 or lhs & rhs:
                continue
            used.update((a_id, b_id))
            yield dict(a, pair_id=a["pair_id"] + "-different", later_id=b_id,
                       later_document=b["earlier_document"], later_page=b["earlier_page"],
                       earlier_text=a["earlier_text"], later_text=b["earlier_text"],
                       reference="different", category="distinct_actions_and_objects",
                       earlier_change="", later_change="")
            break
