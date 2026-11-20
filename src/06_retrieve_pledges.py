"""Search the preceding same-party manifesto using BM25, MiniLM or reciprocal rank fusion."""

from collections import defaultdict

import numpy as np
from common import DATA, OUT, OUTPUTS, VECTORS, fingerprint, key, read, write
from sklearn.feature_extraction.text import CountVectorizer


def run():
    rows = read(DATA / "quasi_sentences.csv")
    lookup = {
        row["statement_id"]: row for row in read(OUTPUTS / "pledges/pledge_scores.csv")
    }
    with np.load(VECTORS) as vectors:
        if str(vectors["fingerprint"]) != fingerprint(rows):
            raise ValueError("Run step 02 again after changing the statements.")
        embeddings = vectors["emb"]
    documents = defaultdict(list)
    for index, row in enumerate(rows):
        documents[row["party"], int(row["year"])].append(index)
    records = []
    for (party, year), indices in sorted(documents.items()):
        earlier_indices = documents.get((party, year - 5), [])
        if not earlier_indices:
            continue
        later_indices = [
            later_index
            for later_index in indices
            if lookup[rows[later_index]["statement_id"]]["prediction"] == "pledge"
        ]
        if not later_indices:
            continue
        # Include earlier non-detections, so an earlier detector error need not hide the right match.
        vectorizer = CountVectorizer(stop_words="english", token_pattern=r"(?u)\b\w+\b")
        tf = vectorizer.fit_transform(
            [rows[position]["text"] for position in earlier_indices]
        ).astype(float)
        document_frequency = np.asarray((tf > 0).sum(0)).ravel()
        idf = np.log1p(
            (len(earlier_indices) - document_frequency + 0.5)
            / (document_frequency + 0.5)
        )
        lengths = np.asarray(tf.sum(1)).ravel()
        term_weights = tf.tocoo()
        denom = term_weights.data + 1.2 * (
            0.25 + 0.75 * lengths[term_weights.row] / max(1, lengths.mean())
        )
        term_weights.data = term_weights.data * 2.2 / denom * idf[term_weights.col]
        queries = vectorizer.transform(
            [rows[later_index]["text"] for later_index in later_indices]
        )
        queries.data[:] = 1
        keyword_scores = (queries @ term_weights.tocsr().T).toarray()
        dense = embeddings[later_indices] @ embeddings[earlier_indices].T
        for query_index, later_index in enumerate(later_indices):
            bm25_rank = (-keyword_scores[query_index]).argsort(
                kind="stable"
            ).argsort() + 1
            dense_rank = (-dense[query_index]).argsort(kind="stable").argsort() + 1
            fusion = 1 / (30 + dense_rank) + np.where(
                keyword_scores[query_index] > 0, 1 / (30 + bm25_rank), 0.0
            )
            fusion_rank = (-fusion).argsort(kind="stable").argsort() + 1
            for position in np.flatnonzero(
                (bm25_rank <= 5) | (dense_rank <= 5) | (fusion_rank <= 5)
            ):
                earlier_row = rows[earlier_indices[position]]
                later_row = rows[later_index]
                records.append(
                    dict(
                        pair_id=key(
                            later_row["statement_id"], earlier_row["statement_id"]
                        ),
                        party=party,
                        year=year,
                        later_id=later_row["statement_id"],
                        earlier_id=earlier_row["statement_id"],
                        bm25_rank=int(bm25_rank[position]),
                        dense_rank=int(dense_rank[position]),
                        fusion_rank=int(fusion_rank[position]),
                        cosine=round(float(dense[query_index, position]), 6),
                        bm25=round(float(keyword_scores[query_index, position]), 5),
                    )
                )
    write(OUT / "retrieval_pool.csv", records)
    print(len(records), "pooled pairs")


if __name__ == "__main__":
    run()
