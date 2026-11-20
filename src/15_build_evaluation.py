"""Freeze expanded natural pairs, text-repeat queries, and controlled revision panels."""
from collections import Counter, defaultdict
import importlib
import re

from common import DATA, OUTPUTS, key, read, write
from evaluation_controls import control_pairs, distinct_controls, stable_key, unchanged_controls
from relations import STOP, delta, evidence, stats, words
from split_audit import Holdout, surface
from weak_pairs import training_examples

DEST = OUTPUTS / "expanded_evaluation"
AUDIT_FIELDS = ["fold", "dataset", "party", "year", "reference", "category", "pairs"]
TRAIN_FIELDS = ["label", "earlier_text", "later_text", "origin", "source_a", "source_b"]
ACTION = re.compile(
    r"^(?:we\s+(?:will|shall)\s+)?"
    r"(build|construct|establish|provide|offer|increase|reduce|protect|introduce|launch|abolish|"
    r"enact|repeal|amend|implement|expand|extend|continue|reform|create|support)\s+(.+)",
    re.I,
)


def natural_reference(earlier, later, historical_change, earlier_confirmed, same_topic):
    first, second = earlier["text"], later["text"]
    if len(words(first)) >= 12 and surface(first) == surface(second):
        return "same", "normalized_repeat", "", ""
    change = historical_change(first, second)
    if change:
        kind, before, after = change
        return "revised", "single_" + kind, before, after
    similarity = stats(first, second)
    number_change = delta(first, second)
    if earlier_confirmed and same_topic and number_change and number_change[0].lastgroup != "date":
        a, b = number_change
        lhs = surface(first[:a.start()]).split()[-3:]
        rhs = surface(second[:b.start()]).split()[-3:]
        if len(lhs) == 3 and lhs == rhs and similarity >= .50:
            return "revised", "aligned_" + a.lastgroup, a.group(), b.group()
    if (earlier_confirmed and similarity >= .85 and len(words(first)) >= 8
            and re.findall(r"\d+", first) == re.findall(r"\d+", second)
            and not re.search(r"\b(?:not|never|only|except|excluding|instead)\b", first + " " + second, re.I)):
        return "same", "high_lexical_agreement", "", ""
    a, b = ACTION.match(first), ACTION.match(second)
    if a and b and a[1].casefold() != b[1].casefold():
        lhs, rhs = set(words(a[2])) - STOP, set(words(b[2])) - STOP
        if len(lhs) >= 3 and len(rhs) >= 3 and not lhs & rhs:
            return "different", "disjoint_action_objects", "", ""
    if earlier_confirmed and not same_topic and similarity < .10:
        return "different", "distinct_topic_and_vocabulary", "", ""
    return "uncertain", "no_decisive_rule", "", ""


def natural_candidates(pool, sources, topics, weak_pledges):
    historical_change = importlib.import_module("14_training_ablation").historical_change
    groups = defaultdict(list)
    for row in pool:
        a, b = row["earlier_id"], row["later_id"]
        if b not in weak_pledges:
            continue
        first, second = sources[a], sources[b]
        if not 5 <= int(first["word_count"]) <= 100 or not 5 <= int(second["word_count"]) <= 100:
            continue
        cosine = float(row["cosine"])
        same_topic = bool(topics.get(a)) and topics.get(a) == topics.get(b)
        if int(row["fusion_rank"]) == 1 and cosine >= 0.80:
            stratum = "high_similarity"
        elif same_topic and 0.60 <= cosine < 0.80:
            stratum = "same_topic_medium"
        elif same_topic and cosine < 0.60:
            stratum = "same_topic_low"
        else:
            stratum = "other_retrieved"
        reference, rule, before, after = natural_reference(first, second, historical_change, a in weak_pledges, same_topic)
        groups[row["party"], int(row["year"])].append(dict(
            row, earlier_text=first["text"], later_text=second["text"],
            earlier_document=first["document_id"], later_document=second["document_id"],
            earlier_page=first["page"], later_page=second["page"], category=stratum,
            reference=reference, rule=rule, earlier_change=before, later_change=after,
            earlier_topic=topics.get(a, ""), later_topic=topics.get(b, ""),
        ))
    return groups


def choose_pairs(rows, limit):
    chosen, seen = [], set()

    def take(candidates, cap):
        added = 0
        for row in candidates:
            later = surface(row["later_text"])
            if later in seen:
                continue
            seen.add(later)
            chosen.append(row)
            added += 1
            if added == cap:
                break

    ranked = sorted(rows, key=lambda row: stable_key(row["pair_id"]))
    # Ensure usable weak labels are represented, and retain the sampling stratum.
    for label in ("revised", "same", "different"):
        take((r for r in ranked if r["reference"] == label), max(1, limit // 5))
    remaining = limit - len(chosen)
    for stratum, fraction in (("high_similarity", .35), ("same_topic_medium", .40),
                              ("same_topic_low", .25)):
        quota = min(limit - len(chosen), max(1, round(remaining * fraction)))
        if quota:
            take((r for r in ranked if r["category"] == stratum), quota)
    if len(chosen) < limit:
        take(ranked, limit - len(chosen))
    return chosen


def exact_queries(sources):
    by_document = defaultdict(lambda: defaultdict(list))
    for row in sources.values():
        text = surface(row["text"])
        if len(text.split()) >= 12 and "possible_heading" not in row["review_flags"]:
            by_document[row["party"], int(row["year"])][text].append(row["statement_id"])
    queries = []
    for (party, year), later in sorted(by_document.items()):
        earlier = by_document.get((party, year - 5), {})
        for text in sorted(earlier.keys() & later.keys()):
            queries.append(dict(party=party, year=year, later_id=min(later[text]),
                                earlier_ids="|".join(sorted(earlier[text])), normalized_text=text))
    return queries


def setup_fold(name, natural, controls, retrieval, cutoff, held_party, sources, pool, weak_pledges):
    directory = DEST / name
    write(directory / "natural_pairs.csv", natural)
    write(directory / "controlled_pairs.csv", controls)
    write(directory / "retrieval_queries.csv", retrieval,
          fields=["party", "year", "later_id", "earlier_ids", "normalized_text"])
    holdout = Holdout(extra_pairs=[*natural, *controls], extra_queries=retrieval, sources=sources)
    eligible = {sid: row for sid, row in sources.items()
                if int(row["year"]) <= cutoff and row["party"] != held_party
                and sid not in holdout.ids and not holdout.text_match(row["text"])}
    write(directory / "eligible_statements.csv", [
        {field: row[field] for field in ("statement_id", "party", "year", "document_id")}
        for row in eligible.values()
    ])
    train_pool = [row for row in pool if row["earlier_id"] in eligible and row["later_id"] in eligible
                  and row["later_id"] in weak_pledges]
    examples = training_examples(train_pool, eligible, safe_quantities=True, quantity_limit=70)
    examples = [r for r in examples if r["origin"] != "historical_rule" or r["label"] != "revised"]
    bases = [row for sid, row in eligible.items() if sid in weak_pledges]
    revisions = list(control_pairs(bases, per_family=70, evaluation=False))
    for row in revisions + list(unchanged_controls(revisions)):
        examples.append(dict(label=row["reference"], earlier_text=row["earlier_text"],
                             later_text=row["later_text"], origin="controlled_" + row["category"],
                             source_a=row["earlier_id"], source_b=row["later_id"]))
    retained = holdout.screen(examples, directory / "training_pairs.csv")
    write(directory / "training_pairs.csv", retained, fields=TRAIN_FIELDS)
    write(directory / "training_mix.csv", [
        dict(label=label, origin=origin, pairs=count)
        for (label, origin), count in sorted(Counter((r["label"], r["origin"]) for r in retained).items())
    ])
    return [
        dict(fold=name, dataset=dataset, party=party, year=year, reference=reference, category=category, pairs=count)
        for dataset, pairs in (("natural", natural), ("controlled", controls))
        for (party, year, reference, category), count in sorted(Counter(
            (r["party"], int(r["year"]), r["reference"], r["category"]) for r in pairs).items())
    ]


def export_query_sample(pool, sources, weak_pledges):
    groups = defaultdict(list)
    for row in sources.values():
        if int(row["year"]) == 2024 and row["statement_id"] in weak_pledges:
            groups[row["party"]].append(row["statement_id"])
    selected = {sid for ids in groups.values() for sid in sorted(ids, key=stable_key)[:12]}
    candidates = [dict(r, relevance="", earlier_text=sources[r["earlier_id"]]["text"],
                       later_text=sources[r["later_id"]]["text"])
                  for r in pool if r["later_id"] in selected and int(r["fusion_rank"]) <= 5]
    write(DEST / "unlabelled_retrieval_sample.csv", candidates)


def export_revision_candidates(groups, weak_pledges):
    """Flag plausible single changes for inspection, with no extra reference labels."""
    candidates = []
    for _, pairs in sorted(groups.items()):
        flagged = []
        for row in pairs:
            same_topic = bool(row["earlier_topic"]) and row["earlier_topic"] == row["later_topic"]
            if (row["reference"] != "uncertain" or row["earlier_id"] not in weak_pledges
                    or not same_topic or float(row["cosine"]) < .60):
                continue
            earlier, later = row["earlier_text"], row["later_text"]
            if stats(earlier, later) < .25:
                continue
            span = evidence(earlier, later, "revised")
            if span and span[-1] != "possible_date_change":
                a, b, c, d, kind = span
                flagged.append(dict(row, evidence_type=kind, earlier_start=a, earlier_end=b,
                                    later_start=c, later_end=d, earlier_change=earlier[a:b],
                                    later_change=later[c:d]))
        used = set()
        for row in sorted(flagged, key=lambda r: (-float(r["cosine"]), r["pair_id"])):
            if row["later_id"] in used:
                continue
            used.add(row["later_id"])
            candidates.append(row)
            if len(used) == 12:
                break
    fields = (list(candidates[0]) if candidates else
              ["pair_id", "party", "year", "earlier_id", "later_id", "reference",
               "earlier_change", "later_change", "evidence_type"])
    write(DEST / "revision_candidates.csv", candidates, fields=fields)


def main():
    sources = {row["statement_id"]: row for row in read(DATA / "quasi_sentences.csv")}
    topics = {row["statement_id"]: row["topic_code"] for row in read(DATA / "statement_topics.csv")}
    weak_pledges = {row["statement_id"] for row in read(OUTPUTS / "pledges/pledge_scores.csv")
                    if row["weak_label"] == "pledge"}
    pool = importlib.import_module("06_retrieve_pledges").run(detected_only=False, output=None)
    export_query_sample(pool, sources, weak_pledges)
    groups = natural_candidates(pool, sources, topics, weak_pledges)
    export_revision_candidates(groups, weak_pledges)
    temporal = [row for (_, year), rows in sorted(groups.items()) if year == 2024
                for row in choose_pairs(rows, 55)]
    lopo = [row for _, rows in sorted(groups.items()) for row in choose_pairs(rows, 22)]
    queries = exact_queries(sources)
    folds = [("temporal", "", 2019), ("strict_temporal", "", 2014)]
    folds += [("lopo_" + party, party, 2019) for party in sorted({r["party"] for r in sources.values()})]
    fold_rows, counts = [], []
    for name, party, cutoff in folds:
        natural = [r for r in lopo if r["party"] == party] if party else temporal
        eligible = [r for sid, r in sources.items() if sid in weak_pledges
                    and (r["party"] == party if party else int(r["year"]) == 2024)]
        revisions = list(control_pairs(eligible, per_family=18 if party else 30, evaluation=True))
        controls = revisions + list(unchanged_controls(revisions)) + list(distinct_controls(revisions, topics, sources))
        retrieval = [r for r in queries if (r["party"] == party if party else r["year"] == 2024)]
        if not natural or not controls:
            raise ValueError(f"{name} has no natural or constructed evaluation rows.")
        counts.extend(setup_fold(name, natural, controls, retrieval, cutoff, party, sources, pool, weak_pledges))
        fold_rows.append(dict(fold=name, held_party=party, training_year_max=cutoff,
                              natural_pairs=len(natural), controlled_pairs=len(controls), exact_queries=len(retrieval)))
        print(f"{name}: {len(natural)} natural pairs, {len(controls)} controls, {len(retrieval)} exact queries", flush=True)
    write(DEST / "folds.csv", fold_rows)
    write(DEST / "dataset_counts.csv", counts, fields=AUDIT_FIELDS)


if __name__ == "__main__":
    main()
