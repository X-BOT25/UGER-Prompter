#!/usr/bin/env python3
"""Train position-first SFT with LoRA.

The model learns to output only {"reading":[one_value]} for bracketed targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_dataset(path: Path, limit: int = 0) -> Dataset:
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        prompt = row.get("prompt")
        completion = row.get("completion")
        if not isinstance(prompt, str) or not isinstance(completion, str):
            continue
        rows.append({"prompt": prompt, "completion": completion})
        if limit and len(rows) >= limit:
            break
    return Dataset.from_list(rows)


def tokenize_function(examples: dict[str, list], tokenizer):
    prompts = examples["prompt"]
    completions = examples["completion"]
    texts = [p + c for p, c in zip(prompts, completions)]
    model_inputs = tokenizer(texts, truncation=True, max_length=512)
    labels = model_inputs["input_ids"].copy()
    prompt_tokens = tokenizer(prompts, truncation=True, max_length=512, add_special_tokens=False)
    for i, prompt_ids in enumerate(prompt_tokens["input_ids"]):
        labels[i][:len(prompt_ids)] = [-100] * len(prompt_ids)
    model_inputs["labels"] = labels
    return model_inputs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    args = ap.parse_args()

    dataset = load_dataset(args.data, limit=args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    tokenized = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
    )

    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    metadata = {
        "base_model": args.model,
        "output": str(args.output),
        "dataset": str(args.data),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "learning_rate": args.learning_rate,
        },
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
        },
    }
    (args.output / "sft_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
