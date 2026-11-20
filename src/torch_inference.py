"""Run a frozen PyTorch transformer. Long inputs use all 256-token windows."""

import numpy as np
import torch
from model_files import model_dir
from transformers import AutoModel, AutoTokenizer


def infer(texts):
    folder = model_dir("minilm")
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    model = AutoModel.from_pretrained(folder, local_files_only=True).to(device).eval()
    windows = []
    for row, text in enumerate(texts):
        encoded = tokenizer(
            text, truncation=True, max_length=256, return_overflowing_tokens=True
        )
        windows.extend((row, ids) for ids in encoded["input_ids"])
    windows.sort(key=lambda item: len(item[1]))
    width = model.config.hidden_size
    sums = np.zeros((len(texts), width), dtype=np.float64)
    counts = np.zeros((len(texts), 1), dtype=np.float64)
    batch_size = 32 if device.type == "cuda" else 8
    print(f"MiniLM: {len(texts)} texts on {device}", flush=True)
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            batch = windows[start : start + batch_size]
            inputs = tokenizer.pad(
                [{"input_ids": ids} for _, ids in batch], return_tensors="pt"
            ).to(device)
            output = model(**inputs)
            mask = inputs.attention_mask.unsqueeze(-1)
            values = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            for (row, ids), value in zip(batch, values.cpu().numpy()):
                weight = max(1, len(ids) - 2)
                sums[row] += value * weight
                counts[row] += weight
    result = (sums / counts).astype(np.float32)
    result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-9)
    return result
