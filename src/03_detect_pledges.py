"""Train a binary classifier on English weak labels through 2019, and apply it to 2024."""

import re
from collections import Counter, defaultdict

import joblib
import numpy as np
from common import DATA, OUTPUTS, VECTORS, fingerprint, read, write
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, log_loss

OUT = OUTPUTS / "pledges"
ACT = r"(?:establish|introduce|create|raise|reduce|abolish|provide|ensure|expand|increase|implement|enact|set up|strengthen|build|launch|restore|review|amend|extend|repeal|allocate|fill|protect|recruit|construct|withdraw|scrap|double|reserve|open|enforce|bring|give|remove|guarantee|finance|invest|demand|press for|seek|urge|oppose|support|continue|ban)"
AGENT = r"(?:we|our (?:party|government)|(?:the )?(?:bjp|congress|dmk|aiadmk|tmc|cpi\s*\(m\)))"
OWN = re.compile(
    r"\b"
    + AGENT
    + r"\s+(?:will|shall|pledges? to|promises? to|undertakes? to|is committed to)\s+(?:also\s+|not\s+)?"
    + ACT
    + r"\b",
    re.I,
)
PASSIVE = re.compile(
    r"\b(?:will|shall)\s+(?:also\s+)?be\s+(?:established|provided|increased|abolished|implemented|introduced|set up|allocated|built|repealed|amended|expanded|recruited|filled)\b",
    re.I,
)
IMPERATIVE = re.compile(r"^" + ACT + r"\b", re.I)
PAST = re.compile(
    r"\b(?:have|has|had|was|were)\b.{0,45}\b(?:introduced|implemented|built|created|launched|achieved|established|failed|increased|reduced|provided|spent)\b",
    re.I,
)
DESCRIPTION = re.compile(
    r"\b(?:account(?:s|ed)? for|according to|at present|census of|there (?:is|are|were)|in (?:19|20)\d\d|under the previous government)\b",
    re.I,
)
FUTURE = re.compile(r"\b(?:will|shall|promise|pledge|commit|must|should)\b", re.I)
OPPONENT = re.compile(
    r"\b(?:opposition|their government|previous government|ruling party)\b.{0,40}\b(?:promis|will|shall)",
    re.I,
)


def weak_label(row, parent):
    text = row["text"]
    votes = []
    if "possible_heading" in row["review_flags"] or OPPONENT.search(text):
        return -1, "abstain_heading_or_attribution"
    actor = re.search(
        r"\b(?:the )?(bjp|congress|dmk|aiadmk|tmc|cpi\s*\(m\))\s+(?:will|shall|promises?|pledges?)\b",
        text,
        re.I,
    )
    if actor:
        named = actor[1].lower().replace(" ", "")
        own = {"inc": "congress", "cpim": "cpi(m)"}.get(
            row["party"].lower(), row["party"].lower()
        )
        if named != own:
            return -1, "abstain_other_party"
    if OWN.search(text):
        votes.append("party_action")
    if PASSIVE.search(text):
        votes.append("future_policy_action")
    if IMPERATIVE.search(text) and int(row["word_count"]) >= 3:
        votes.append("action_bullet")
    if re.search(
        r"\bwe (?:will|shall)\s*:\s*$", parent[: int(row["start_char"])], re.I
    ) and IMPERATIVE.search(text):
        votes.append("parent_commitment")
    if PAST.search(text):
        votes.append("past_record")
    if DESCRIPTION.search(text) and not FUTURE.search(text):
        votes.append("descriptive")
    pos = any(
        v
        in [
            "party_action",
            "future_policy_action",
            "action_bullet",
            "parent_commitment",
        ]
        for v in votes
    )
    neg = any(v in ["past_record", "descriptive"] for v in votes)
    return (int(pos) if pos != neg else -1), "|".join(votes) or "no_vote"


def fit_detector(rows, embeddings, weak_labels, train):
    texts = [row["text"] for row in rows]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    vectorizer.fit(texts[i] for i in train)
    features = hstack(
        [vectorizer.transform(texts), csr_matrix(embeddings)], format="csr"
    )
    model = LogisticRegression(class_weight="balanced", max_iter=600, random_state=19)
    model.fit(features[train], weak_labels[train])
    return vectorizer, model, model.predict_proba(features)[:, 1]


def run(vectors):
    rows = read(DATA / "quasi_sentences.csv")
    parent = {
        row["passage_id"]: row["text"] for row in read(DATA / "master_passages.csv")
    }
    cache = np.load(vectors)
    if str(cache["fingerprint"]) != fingerprint(rows):
        raise ValueError("Re-encode the current statements")
    embeddings = cache["emb"]
    labels = [weak_label(row, parent[row["passage_id"]]) for row in rows]
    weak_labels = np.array([label for label, _ in labels])
    years = np.array([int(row["year"]) for row in rows])
    train = np.flatnonzero((weak_labels >= 0) & (years <= 2019))
    vectorizer, model, probabilities = fit_detector(rows, embeddings, weak_labels, train)
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "classifier": model,
            "embedding_model": "all-MiniLM-L6-v2",
        },
        OUT / "pledge_classifier.joblib",
    )
    pred = []
    groups = defaultdict(Counter)
    for i, row in enumerate(rows):
        decision = "pledge" if probabilities[i] >= 0.5 else "non_pledge"
        weak = {-1: "abstain", 0: "non_pledge", 1: "pledge"}[int(weak_labels[i])]
        pred.append(
            dict(
                statement_id=row["statement_id"],
                passage_id=row["passage_id"],
                party=row["party"],
                year=row["year"],
                weak_label=weak,
                rules=labels[i][1],
                prediction=decision,
                pledge_probability=float(probabilities[i]),
                close_to_threshold=int(abs(probabilities[i] - 0.5) < 0.1),
                split="held_out" if years[i] == 2024 else "historical",
            )
        )
        counts = groups[(row["party"], row["year"])]
        counts["total"] += 1
        counts[weak] += 1
        counts["detected"] += int(probabilities[i] >= 0.5)
    write(OUT / "pledge_scores.csv", pred)
    write(
        OUT / "weak_coverage.csv",
        [
            dict(
                party=k[0],
                year=k[1],
                total=v["total"],
                weak_pledge=v["pledge"],
                weak_non_pledge=v["non_pledge"],
                abstain=v["abstain"],
                model_pledges=v["detected"],
            )
            for k, v in sorted(groups.items())
        ],
    )
    test = np.flatnonzero((weak_labels >= 0) & (years == 2024))
    summary = classification_report(
        weak_labels[test],
        probabilities[test] >= 0.5,
        target_names=["non_pledge", "pledge"],
        output_dict=True,
    )
    from json import dumps

    summary.update(
        train_weak_labels=len(train),
        held_out_weak_labels=len(test),
        held_out_log_loss=log_loss(weak_labels[test], probabilities[test]),
    )
    (OUT / "agreement_with_2024_rules.json").write_text(dumps(summary, indent=2))
    rng = np.random.default_rng(19)
    audit = []
    for key in sorted(groups):
        indices = [
            i for i, row in enumerate(rows) if (row["party"], row["year"]) == key
        ]
        for i in rng.choice(indices, size=min(12, len(indices)), replace=False):
            audit.append(
                dict(
                    **pred[i],
                    text=rows[i]["text"],
                    parent_text=parent[rows[i]["passage_id"]],
                    reviewer_label="",
                    notes="",
                )
            )
    write(OUT / "review_sample.csv", audit)


if __name__ == "__main__":
    run(VECTORS)
