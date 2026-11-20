"""Create weak relation labels from historical matches and controlled numeric edits."""

import random
import re
from collections import defaultdict

from relations import MEASURE, rule


def training_examples(pool, statements, safe_quantities=False):
    rng = random.Random(88)
    groups = defaultdict(list)
    for row in pool:
        if int(row["year"]) >= 2024:
            continue
        earlier, later = (
            statements[row[k]]["text"] for k in ("earlier_id", "later_id")
        )
        label = rule(earlier, later, float(row["cosine"]))
        if label != "unclear":
            groups[label].append(
                (earlier, later, "historical_rule", row["earlier_id"], row["later_id"])
            )
    training_examples = []
    seen = set()

    def add(label, earlier, later, origin, earlier_id, later_id):
        if (earlier, later) in seen:
            return
        seen.add((earlier, later))
        training_examples.append(
            dict(
                label=label,
                earlier_text=earlier,
                later_text=later,
                origin=origin,
                source_a=earlier_id,
                source_b=later_id,
            )
        )

    for label, group in groups.items():
        rng.shuffle(group)
        for example in group[:700]:
            add(label, *example)
    historical = [
        v
        for v in statements.values()
        if int(v["year"]) <= 2019 and 6 <= len(v["text"].split()) <= 65
    ]
    rng.shuffle(historical)
    for row in historical:
        original = row["text"]
        statement_id = row["statement_id"]
        measurements = list(MEASURE.finditer(original))
        if len(measurements) != 1:
            continue
        part = measurements[0].group()
        if safe_quantities and (measurements[0].lastgroup == "date" or re.search(r"\d\.\d", part)):
            continue
        number = re.search(r"\d[\d,]*", part)
        if not number:
            continue
        start = measurements[0].start() + number.start()
        end = measurements[0].start() + number.end()
        value = int(number.group().replace(",", ""))
        if value <= 0:
            continue
        change = (
            1
            if value <= 10 or measurements[0].lastgroup == "date"
            else max(1, value // 5)
        )
        changed = original[:start] + str(value + change) + original[end:]
        add(
            "revised",
            original,
            changed,
            "controlled_quantity_edit",
            statement_id,
            statement_id,
        )
        paraphrase = re.sub(r"\bwill\b", "shall", original, flags=re.I)
        if paraphrase == original:
            paraphrase = re.sub(
                r"\b(?:Rs\.?|rupees)\s*(?=\d)", "₹", original, flags=re.I
            )
        add(
            "same",
            original,
            paraphrase,
            "identity_or_surface_control",
            statement_id,
            statement_id,
        )
        if (
            sum(e["origin"] == "controlled_quantity_edit" for e in training_examples)
            >= 500
        ):
            break
    rng.shuffle(training_examples)
    return training_examples
