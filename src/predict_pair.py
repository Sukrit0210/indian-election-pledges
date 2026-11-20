"""Use the supplied checkpoint on one earlier/later English pair, without retraining."""

import argparse
import json
from pathlib import Path

import torch
from common import CHECKPOINT
from relations import evidence
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def predict(earlier, later, checkpoint):
    if not earlier.strip() or not later.strip():
        raise ValueError("Both passages are required")
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            checkpoint, local_files_only=True
        )
        .to(device)
        .eval()
    )
    args = tokenizer(
        earlier, later, truncation=True, max_length=256, return_tensors="pt"
    ).to(device)
    with torch.inference_mode():
        probs = model(**args).logits.float().softmax(-1)[0].tolist()
    top = max(range(len(probs)), key=probs.__getitem__)
    label = model.config.id2label[top] if probs[top] >= 0.65 else "unclear"
    span = evidence(earlier, later, label)
    rationale = None
    if span:
        i, j, k, l, method = span
        rationale = dict(
            earlier=earlier[i:j],
            later=later[k:l],
            earlier_offsets=[i, j],
            later_offsets=[k, l],
            method=method,
        )
    return dict(
        decision=label,
        probabilities=dict(
            zip([model.config.id2label[i] for i in range(len(probs))], probs)
        ),
        proposed_evidence=rationale,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--earlier", required=True)
    p.add_argument("--later", required=True)
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = p.parse_args()
    print(
        json.dumps(
            predict(args.earlier, args.later, args.checkpoint),
            indent=2,
            ensure_ascii=False,
        )
    )
