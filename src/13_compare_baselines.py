"""Score frozen relation baselines against the exploratory panels and controls."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re

import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from common import CACHE, OUTPUTS, REVIEW, ROOT, read, write

NLI_MODEL = ("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", "0e2603d5d3d3ef9b2910814b34eebe1a2101da65")
LLM_MODEL = ("Qwen/Qwen2.5-1.5B-Instruct", "989aa79")
LABELS = ("same", "revised", "different")
DEFINITIONS = {
    "same": "Both texts make the same commitment, with no material change.",
    "revised": "Both texts concern the same commitment and the later text materially changes its obligation.",
    "different": "The texts concern different commitments or at least one text does not make a commitment.",
}
DEST = OUTPUTS / "baseline_comparison"


def load_cases():
    cases = []
    for split in ("pilot", "test"):
        seen = set()
        for row in read(REVIEW / f"{split}_reference.csv"):
            key = (row["party"], row["earlier_text"].lower().strip(), row["later_text"].lower().strip())
            if key in seen:
                continue
            seen.add(key)
            cases.append(dict(split=split, category="reference_panel", pair_id=row["pair_id"],
                              reference=row["relation"], earlier_text=row["earlier_text"],
                              later_text=row["later_text"]))
    cases.extend(dict(split="controls", **r) for r in read(REVIEW / "controlled_pairs.csv"))
    return cases


def model_inputs(cases):
    return [f"Earlier statement: {r['earlier_text']}\nLater statement: {r['later_text']}" for r in cases]


def nli(cases, device):
    repo, revision = NLI_MODEL
    args = dict(revision=revision, cache_dir=CACHE / "hf")
    tokenizer = AutoTokenizer.from_pretrained(repo, **args)
    model = AutoModelForSequenceClassification.from_pretrained(repo, **args).to(device).eval()
    entailment = next(i for i, label in model.config.id2label.items() if label.lower() == "entailment")
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(cases), 8):
            premises = model_inputs(cases[start:start + 8])
            inputs = tokenizer([p for p in premises for _ in LABELS], list(DEFINITIONS.values()) * len(premises),
                               padding=True, truncation="only_first", max_length=512, return_tensors="pt").to(device)
            scores = model(**inputs).logits.softmax(-1)[:, entailment].reshape(len(premises), 3)
            predictions.extend(LABELS[i] for i in scores.argmax(1).tolist())
    return predictions


def llm(cases, device):
    repo, revision = LLM_MODEL
    args = dict(revision=revision, cache_dir=CACHE / "hf")
    tokenizer = AutoTokenizer.from_pretrained(repo, **args)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=dtype, **args).to(device).eval()
    generation = deepcopy(model.generation_config)
    generation.update(
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        max_new_tokens=8,
        pad_token_id=tokenizer.eos_token_id,
    )
    instructions = "\n".join(f"{label}: {description}" for label, description in DEFINITIONS.items())
    prompt = "Compare two manifesto statements. Answer with exactly one label.\n" + instructions
    predictions = []
    with torch.inference_mode():
        for pair in model_inputs(cases):
            ids = tokenizer.apply_chat_template(
                [{"role": "system", "content": prompt}, {"role": "user", "content": pair}],
                tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
            ).to(device)
            reply = model.generate(**ids, generation_config=generation)
            answer = tokenizer.decode(reply[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
            match = re.fullmatch(r'\s*(same|revised|different)[.\s]*', answer)
            predictions.append(match.group(1) if match else "unclear")
    return predictions


def trained(cases, path, device, threshold):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True).to(device).eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(cases), 16):
            batch = cases[start:start + 16]
            inputs = tokenizer([r["earlier_text"] for r in batch], [r["later_text"] for r in batch],
                               padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            probs = model(**inputs).logits.softmax(-1)
            predictions.extend(
                model.config.id2label[int(p.argmax())] if float(p.max()) >= threshold else "unclear"
                for p in probs
            )
    return predictions


def add_rows(rows, cases, method, answers):
    rows.extend(dict(split=r["split"], category=r["category"], pair_id=r["pair_id"],
                     reference=r["reference"], method=method, prediction=p)
                for r, p in zip(cases, answers, strict=True))


def summarize(predictions):
    metrics, confusion = [], []
    groups = {}
    for row in predictions:
        if row["reference"] in LABELS:
            groups.setdefault((row["split"], row["method"]), []).append(row)
    for (split, method), scored in sorted(groups.items()):
        truth = [r["reference"] for r in scored]
        guess = [r["prediction"] for r in scored]
        matrix = Counter(zip(truth, guess))
        revised = truth.count("revised")
        pred_revised = guess.count("revised")
        correct_rev = matrix["revised", "revised"]
        metrics.append(dict(split=split, method=method, evaluated=len(scored),
            covered=len(scored) - guess.count("unclear"), revised_support=revised,
            predicted_revised=pred_revised, correct_revised=correct_rev,
            revised_precision=correct_rev / pred_revised if pred_revised else "",
            revised_recall=correct_rev / revised if revised else "",
            macro_f1=f1_score(truth, guess, labels=LABELS, average="macro", zero_division=0)))
        confusion.extend(dict(split=split, method=method, reference=a, prediction=b, count=matrix[a, b])
                         for a in LABELS for b in (*LABELS, "unclear"))
    return metrics, confusion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("nli", "llm", "both", "archived"), default="both")
    parser.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH")
    args = parser.parse_args()
    cases = load_cases()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    add_rows(rows, cases, "always_different", ["different"] * len(cases))
    archive = {r["pair_id"]: r for r in read(ROOT / "reference/comparison/panel_predictions.csv")}
    pilot = [r for r in cases if r["split"] != "controls"]
    for name, column in [("weak_rule", "rule_label"), ("released_minilm", "model_label")]:
        add_rows(rows, pilot, name, [archive[r["pair_id"]][column] for r in pilot])
    for name, scorer in [("nli", nli), ("llm", llm)]:
        if args.method in (name, "both"):
            add_rows(rows, cases, name, scorer(cases, device))
    for spec in args.checkpoint:
        name, sep, checkpoint = spec.partition("=")
        if not sep or not name or not checkpoint:
            parser.error("--checkpoint must be NAME=PATH")
        add_rows(rows, cases, name, trained(cases, Path(checkpoint), device, 0.65))
    metrics, confusion = summarize(rows)
    write(DEST / "predictions.csv", rows)
    write(DEST / "metrics.csv", metrics)
    write(DEST / "confusion_counts.csv", confusion)
    control_groups = {}
    for row in rows:
        if row["split"] == "controls":
            control_groups.setdefault((row["category"], row["method"]), []).append(row)
    write(DEST / "control_results.csv", [
        dict(category=category, method=method, cases=len(group),
             correct=sum(r["prediction"] == r["reference"] for r in group),
             abstentions=sum(r["prediction"] == "unclear" for r in group))
        for (category, method), group in sorted(control_groups.items())
    ])
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "models.json").write_text(json.dumps({"nli": NLI_MODEL, "llm": LLM_MODEL}, indent=2) + "\n")


if __name__ == "__main__":
    main()
