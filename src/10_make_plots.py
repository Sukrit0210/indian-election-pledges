"""Build the six research figures from this run's CSV files."""

import os

from common import CACHE, DATA, OUT, OUTPUTS, REGISTRY, REVIEW

os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURES = OUTPUTS / "figures"
TEAL = "#156e66"
ORANGE = "#ad521e"
PARTIES = ["BJP", "INC", "CPIM", "DMK", "TMC", "AIADMK"]
YEARS = [2009, 2014, 2019, 2024]


def save(figure, name, data):
    FIGURES.mkdir(parents=True, exist_ok=True)
    data.to_csv(FIGURES / f"{name}.csv")
    for extension in ["png", "svg"]:
        figure.savefig(FIGURES / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def heatmap(table, name, title, xlabel, ylabel, missing=None):
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    values = table.to_numpy(dtype=float)
    image = ax.imshow(values, cmap="BuGn", aspect="auto", vmin=0)
    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set_yticks(range(len(table)), table.index)
    for i, row in enumerate(table.index):
        for j, column in enumerate(table.columns):
            value = values[i, j]
            if np.isnan(value):
                text = missing[(row, column)]
            else:
                text = f"{value:,.0f}"
            color = "white" if value > 0.60 * image.get_clim()[1] else "#183535"
            ax.text(j, i, text, va="center", ha="center", color=color)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    save(fig, name, table)


def corpus_coverage():
    registry = pd.read_csv(REGISTRY).set_index(["party", "year"])
    units = pd.read_csv(DATA / "quasi_sentences.csv")
    table = (
        units.groupby(["party", "year"])
        .size()
        .unstack()
        .reindex(index=PARTIES, columns=YEARS)
    )
    missing = {}
    for party in PARTIES:
        for year in YEARS:
            if pd.isna(table.loc[party, year]):
                missing[party, year] = (
                    "English PDF pending"
                    if (party, year) in registry.index
                    else "Outside registry"
                )
    heatmap(
        table,
        "01_coverage",
        "How much text does each manifesto contribute?",
        "Lok Sabha election",
        "Party",
        missing,
    )


def retrieval():
    filtered = (
        pd.read_csv(OUT / "retrieval_challenge.csv")
        .dropna(subset=["earlier_ids"])
        .set_index("later_id")
    )
    direct = (
        pd.read_csv(OUT / "retrieval_without_detector.csv")
        .set_index("later_id")
        .loc[filtered.index]
    )
    table = pd.DataFrame(index=["BM25", "Dense", "Hybrid"])
    methods = ["bm25", "dense", "fusion"]
    table["With pledge filter"] = [
        int(filtered[m + "_hit_at_5"].sum()) for m in methods
    ]
    table["Without pledge filter"] = [
        int(direct[m + "_first_reference_rank"].between(1, 5).sum()) for m in methods
    ]
    ax = table.plot.barh(color=[TEAL, ORANGE], figsize=(8, 4))
    ax.bar_label(ax.containers[0], padding=3)
    ax.bar_label(ax.containers[1], padding=3)
    ax.set(
        xlabel="Queries with a reference match in the first five results",
        xlim=(0, len(filtered) + 0.5),
        title=f"Retrieval on {len(filtered)} targeted queries with known matches",
    )
    ax.legend(loc="lower right")
    save(ax.figure, "03_retrieval", table)


def relations():
    decisions = pd.read_csv(OUT / "test_decisions.csv")
    labels = ["same", "revised", "different"]
    determinate = decisions[decisions.reference.isin(labels)]
    table = pd.crosstab(determinate.reference, determinate.cross_encoder)
    columns = labels + (["unclear"] if "unclear" in table.columns else [])
    table = table.reindex(index=labels, columns=columns, fill_value=0)
    heatmap(
        table,
        "04_relations",
        f"Relationship decisions on {len(determinate)} reviewed pairs",
        "Cross-encoder prediction",
        "Reference review",
    )


def revision_audit():
    matches = pd.read_csv(OUT / "best_matches.csv", keep_default_na=False)
    flags = matches[(matches.year == 2024) & (matches.decision == "revised")]
    audit = pd.read_csv(REVIEW / "flagged_2024_audit.csv", keep_default_na=False)
    historical = audit.set_index(["later_id", "earlier_id"])
    outcomes = []
    for row in flags.itertuples():
        pair = (row.later_id, row.earlier_id)
        if pair not in historical.index:
            outcomes.append("Awaiting review")
            continue
        reviewed = historical.loc[pair]
        if reviewed.reference_relation != "revised":
            outcomes.append(reviewed.reference_relation.capitalize())
        elif (reviewed.earlier_evidence, reviewed.later_evidence) != (
            row.earlier_evidence,
            row.later_evidence,
        ):
            outcomes.append("Revised, new highlights unreviewed")
        elif reviewed.evidence_supports_one_material_change == "yes":
            outcomes.append("Revised, evidence supports change")
        else:
            outcomes.append("Revised, evidence insufficient")
    table = pd.Series(outcomes, dtype=str).value_counts().rename("pairs").to_frame()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(table.index, table.pairs, color=TEAL)
    ax.bar_label(ax.containers[0], padding=3)
    ax.set(
        xlabel="Pairs",
        title=f"Reference review coverage of {len(flags)} revision flags in 2024",
    )
    save(fig, "05_revision_audit", table)


def forecast():
    forecasts = pd.read_csv(OUTPUTS / "pledges/forecast_summary.csv")
    backtest = forecasts[forecasts.target_year == 2024]
    table = backtest.pivot(index="party", columns="model", values="total_variation")
    ax = table[["persistence", "half_trend", "tuned_trend"]].plot.barh(
        color=[TEAL, ORANGE, "#84949a"], figsize=(8, 5)
    )
    ax.set(
        title="How close were the 2024 topic forecasts?",
        xlabel="Total variation distance (lower is better)",
        ylabel="Party",
    )
    ax.legend(["Previous shares", "Half trend", "Tuned trend"])
    save(ax.figure, "06_forecast", table)


if __name__ == "__main__":
    for plot in [
        corpus_coverage,
        retrieval,
        relations,
        revision_audit,
        forecast,
    ]:
        plot()
    print("Saved five plots and their underlying tables to", FIGURES)
