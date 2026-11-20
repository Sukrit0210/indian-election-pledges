"""Create one normalized MiniLM vector per quasi-sentence."""

import argparse

import numpy as np
from common import DATA, VECTORS, fingerprint, read
from torch_inference import infer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    rows = read(DATA / "quasi_sentences.csv")
    vectors = infer([row["text"] for row in rows])
    VECTORS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        VECTORS, emb=vectors, fingerprint=fingerprint(rows)
    )
    print(f"Saved {len(vectors)} vectors to {VECTORS}")


if __name__ == "__main__":
    main()
