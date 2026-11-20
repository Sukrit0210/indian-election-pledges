"""Forecast the distribution of policy topics among each party's detected pledges."""

import json
from collections import Counter, defaultdict

import numpy as np
from common import DATA, OUTPUTS, read, write
from model_files import model_dir

OUT = OUTPUTS / "pledges"


def predict(distributions, party, year, alpha):
    latest = distributions[party, year - 5]
    previous = distributions.get((party, year - 10))
    if previous is None:
        return latest.copy()
    shifted = np.maximum(0, latest + alpha * (latest - previous))
    return shifted / shifted.sum()


def tv(x, y):
    return float(abs(x - y).sum() / 2)


def run():
    labels = json.loads((model_dir("topics") / "config.json").read_text())["id2label"]
    topics = [labels[str(i)] for i in range(len(labels))]
    idx = {t.split()[0]: i for i, t in enumerate(topics)}
    scores = read(OUT / "pledge_scores.csv")
    annotations = {r["statement_id"]: r for r in read(DATA / "statement_topics.csv")}
    counts = defaultdict(lambda: np.zeros(len(topics), dtype=int))
    totals = Counter()
    for r in scores:
        key = (r["party"], int(r["year"]))
        totals[key] += 1
        if r["prediction"] == "pledge":
            counts[key][idx[annotations[r["statement_id"]]["topic_code"]]] += 1
    ratios = {k: c / c.sum() for k, c in counts.items()}
    write(
        OUT / "observed_topic_shares.csv",
        [
            dict(
                party=k[0],
                year=k[1],
                statements=totals[k],
                detected_pledges=int(c.sum()),
                topic=topics[i],
                pledges=int(n),
                share=float(ratios[k][i]),
            )
            for k, c in sorted(counts.items())
            for i, n in enumerate(c)
        ],
    )
    # Only pre-2024 targets with two earlier elections can inform the choice of trend weight.
    tuning = [
        k
        for k in ratios
        if k[1] < 2024 and (k[0], k[1] - 5) in ratios and (k[0], k[1] - 10) in ratios
    ]
    grid = {
        a: np.mean([tv(predict(ratios, p, y, a), ratios[p, y]) for p, y in tuning])
        for a in [0.0, 0.25, 0.5, 1.0]
    }
    alpha = min(grid, key=grid.get)
    write(
        OUT / "historical_tuning.csv",
        [
            dict(
                alpha=a, eligible_transitions=len(tuning), mean_total_variation=float(v)
            )
            for a, v in grid.items()
        ],
    )
    summary = []
    forecasts = []
    for party, yr in sorted(ratios):
        if yr != 2024 or (party, 2019) not in ratios:
            continue
        for target in [2024, 2029]:
            truth = ratios.get((party, target))
            for name, a in [
                ("persistence", 0.0),
                ("half_trend", 0.5),
                ("tuned_trend", alpha),
            ]:
                prob = predict(ratios, party, target, a)
                top = np.argsort(-prob, kind="stable")[:3]
                for i, row in enumerate(prob):
                    forecasts.append(
                        dict(
                            party=party,
                            target_year=target,
                            forecast_origin=target - 5,
                            model=name,
                            topic=topics[i],
                            forecast_share=float(row),
                            observed_share="" if truth is None else float(truth[i]),
                        )
                    )
                summary.append(
                    dict(
                        party=party,
                        target_year=target,
                        model=name,
                        alpha=a,
                        total_variation="" if truth is None else tv(prob, truth),
                        common_top3=""
                        if truth is None
                        else len(set(top) & set(np.argsort(-truth, kind="stable")[:3])),
                        first=topics[top[0]],
                        second=topics[top[1]],
                        third=topics[top[2]],
                    )
                )
    write(OUT / "topic_forecasts.csv", forecasts)
    write(OUT / "forecast_summary.csv", summary)
    sensitivity = []
    for threshold in [0.4, 0.5, 0.6]:
        sets = defaultdict(Counter)
        for r in scores:
            if float(r["pledge_probability"]) >= threshold:
                sets[r["party"], r["year"]][
                    annotations[r["statement_id"]]["topic_label"]
                ] += 1
        for (party, year), group in sorted(sets.items()):
            top = group.most_common(1)[0][0]
            sensitivity.append(
                dict(
                    party=party,
                    year=year,
                    threshold=threshold,
                    detected_pledges=group.total(),
                    first_topic=top,
                    first_topic_share=group[top] / group.total(),
                )
            )
    write(OUT / "threshold_sensitivity.csv", sensitivity)

if __name__ == "__main__":
    run()
