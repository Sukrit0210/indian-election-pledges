"""Fine-tune a MiniLM cross-encoder on historical weak labels in PyTorch."""

import random
from collections import Counter
import re
from sklearn.metrics import f1_score

import numpy as np
import torch
from common import CHECKPOINT, DATA, OUT, REVIEW, read, write
from model_files import model_dir
from pair_model import LABELS, batch_outputs
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from weak_pairs import training_examples


def run(base, checkpoint, examples=None, output=OUT / "training_pairs.csv"):
    torch.manual_seed(88)
    np.random.seed(88)
    torch.set_num_threads(4)
    if examples is None:
        statements = {r["statement_id"]: r for r in read(DATA / "quasi_sentences.csv")}
        pool = read(OUT / "retrieval_pool.csv")
        examples = training_examples(pool, statements)
    training_pairs = list(examples)
    write(output, training_pairs)
    print(
        Counter((pair["origin"], pair["label"]) for pair in training_pairs), flush=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=3, ignore_mismatched_sizes=True, local_files_only=True
    )
    # Train the new three-class head, the pooler, and the final three encoder layers.
    for parameter in model.base_model.parameters():
        parameter.requires_grad = False
    for block in model.base_model.encoder.layer[-3:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    if getattr(model.base_model, "pooler", None):
        for parameter in model.base_model.pooler.parameters():
            parameter.requires_grad = True
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=4e-5,
    )
    class_counts = Counter(pair["label"] for pair in training_pairs)
    weights = torch.tensor(
        [len(training_pairs) / (3 * class_counts[label]) for label in LABELS],
        device=device,
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    epochs = 5
    for epoch in range(epochs):
        epoch_loss = 0.0
        random.Random(88 + epoch).shuffle(training_pairs)
        for start in range(0, len(training_pairs), 16):
            sample = training_pairs[start : start + 16]
            logits = batch_outputs(
                model,
                tokenizer,
                [(pair["earlier_text"], pair["later_text"]) for pair in sample],
                device,
            )
            loss = loss_fn(
                logits,
                torch.tensor(
                    [LABELS.index(pair["label"]) for pair in sample], device=device
                ),
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(sample)
        print(
            f"Epoch {epoch + 1}/{epochs}, mean batch loss {epoch_loss / len(training_pairs):.4f}",
            flush=True,
        )
    model.config.id2label = dict(enumerate(LABELS))
    model.config.label2id = {s: i for i, s in enumerate(LABELS)}
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    print("Saved fine-tuned model to", checkpoint)


if __name__ == "__main__":
    run(model_dir("comparison"), CHECKPOINT)
