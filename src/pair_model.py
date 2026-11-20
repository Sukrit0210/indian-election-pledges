LABELS = ["different", "same", "revised"]


def batch_outputs(model, tokenizer, batch, device):
    encoded = tokenizer(
        [pair[0] for pair in batch],
        [pair[1] for pair in batch],
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    ).to(device)
    return model(**encoded).logits
