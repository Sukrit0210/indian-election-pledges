"""Score the retrieved pairs and the fixed review panel using the saved checkpoint."""

import numpy as np
import torch
from common import CHECKPOINT, DATA, OUT, REVIEW, VECTORS, fingerprint, read, write
from pair_model import LABELS, batch_outputs
from relations import evidence, rule
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def score_pool(checkpoint):
    docs = {r["statement_id"]: r for r in read(DATA / "quasi_sentences.csv")}
    pool = read(OUT / "retrieval_pool.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, local_files_only=True
    ).to(device)
    for_predictions = [r for r in pool if int(r["fusion_rank"]) <= 5]
    top_five_ids = {r["pair_id"] for r in for_predictions}
    panel_rows = read(REVIEW / "test_reference.csv") + read(
        REVIEW / "pilot_reference.csv"
    )
    vectors = np.load(VECTORS)
    if str(vectors["fingerprint"]) != fingerprint(list(docs.values())):
        raise ValueError("Re-run step 02 on the current corpus")
    matrix = vectors["emb"]
    positions = {sid: i for i, sid in enumerate(docs)}
    seen = set(top_five_ids)
    for row in panel_rows:
        if row["pair_id"] in seen:
            continue
        a, b = row["earlier_id"], row["later_id"]
        cosine = float(matrix[positions[a]] @ matrix[positions[b]])
        for_predictions.append(
            dict(
                pair_id=row["pair_id"],
                party=row["party"],
                year=docs[b]["year"],
                later_id=b,
                earlier_id=a,
                bm25_rank="",
                dense_rank="",
                fusion_rank="",
                cosine=cosine,
                bm25="",
            )
        )
        seen.add(row["pair_id"])
    for_predictions.sort(
        key=lambda r: sum(len(docs[r[k]]["text"]) for k in ("earlier_id", "later_id"))
    )
    model.eval()
    if device == "cuda":
        model.half()
    evaluated = []
    with torch.inference_mode():
        for j in range(0, len(for_predictions), 48):
            batch = for_predictions[j : j + 48]
            texts = [
                (docs[r["earlier_id"]]["text"], docs[r["later_id"]]["text"])
                for r in batch
            ]
            probs = (
                batch_outputs(model, tokenizer, texts, device)
                .float()
                .softmax(-1)
                .cpu()
                .numpy()
            )
            for r, (a, b), probabilities in zip(batch, texts, probs):
                label = (
                    LABELS[probabilities.argmax()]
                    if max(probabilities) >= 0.65
                    else "unclear"
                )
                spans = evidence(a, b, label)
                highlights = dict(
                    earlier_start="",
                    earlier_end="",
                    later_start="",
                    later_end="",
                    earlier_evidence="",
                    later_evidence="",
                    evidence_kind="",
                )
                if spans:
                    i, k, l, m, kind = spans
                    highlights.update(
                        earlier_start=i,
                        earlier_end=k,
                        later_start=l,
                        later_end=m,
                        earlier_evidence=a[i:k],
                        later_evidence=b[l:m],
                        evidence_kind=kind,
                    )
                evaluated.append(
                    dict(
                        **r,
                        rule_label=rule(a, b, float(r["cosine"])),
                        model_label=label,
                        **{
                            "p_" + s: round(float(v), 6)
                            for s, v in zip(LABELS, probabilities)
                        },
                        **highlights,
                    )
                )
    panel_ids = {r["pair_id"] for r in panel_rows}
    write(
        OUT / "panel_predictions.csv",
        [r for r in evaluated if r["pair_id"] in panel_ids],
    )
    evaluated = [r for r in evaluated if r["pair_id"] in top_five_ids]
    evaluated.sort(key=lambda r: (r["later_id"], int(r["fusion_rank"])))
    write(OUT / "pair_predictions.csv", evaluated)
    best = []
    from itertools import groupby

    for later_id, group in groupby(evaluated, key=lambda r: r["later_id"]):
        candidates = list(group)
        matches = [r for r in candidates if r["model_label"] in ("same", "revised")]
        pick = (
            max(matches, key=lambda r: (1 - r["p_different"], -int(r["fusion_rank"])))
            if matches
            else candidates[0]
        )
        best.append(
            dict(
                later_id=later_id,
                party=pick["party"],
                year=pick["year"],
                earlier_id=pick["earlier_id"] if matches else "",
                decision=pick["model_label"] if matches else "no_confident_match",
                later_text=docs[later_id]["text"],
                earlier_text=docs[pick["earlier_id"]]["text"] if matches else "",
                earlier_evidence=pick["earlier_evidence"],
                later_evidence=pick["later_evidence"],
            )
        )
    write(OUT / "best_matches.csv", best)


if __name__ == "__main__":
    score_pool(CHECKPOINT)
