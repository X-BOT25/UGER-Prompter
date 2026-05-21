#!/usr/bin/env python3
"""Predict position-first rows with adapter.

The adapter is trained to output only {"reading":[...]}.
Explanation is generated later by export script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def extract_json(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    while start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
                    return obj if isinstance(obj, dict) else None
        start = text.find("{", start + 1)
    return None


def validate(obj: dict[str, Any] | None, row: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    reading = obj.get("reading")
    candidates = {str(item) for item in row.get("candidate_readings", []) if str(item)}
    if not isinstance(reading, list) or len(reading) != 1:
        return None
    normalized = [str(item) for item in reading]
    if any(item not in candidates for item in normalized):
        return None
    return {"reading": normalized}


def batched(rows: list[dict[str, Any]], size: int):
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def predict(
    rows: list[dict[str, Any]],
    *,
    model_path: str,
    adapter_path: str,
    out: Path,
    device: str,
    batch_size: int,
    max_new_tokens: int,
) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if device not in {"cpu", "none"}:
        kwargs["device_map"] = device
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model = PeftModel.from_pretrained(base, adapter_path) if adapter_path else base
    model.eval()

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        done = 0
        for batch in batched(rows, batch_size):
            prompts = [str(row.get("prompt", "")) for row in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True)
            if device not in {"cpu", "none"}:
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prompt_width = inputs["input_ids"].shape[1]
            for row, seq in zip(batch, generated):
                raw = tokenizer.decode(seq[prompt_width:], skip_special_tokens=True)
                prediction = validate(extract_json(raw), row)
                rec = {
                    "id": row.get("id"),
                    "original_row_id": row.get("original_row_id"),
                    "original_row_index": row.get("original_row_index"),
                    "input": row.get("input"),
                    "source_rows": row.get("source_rows"),
                    "task_type": row.get("task_type"),
                    "target_char": row.get("target_char"),
                    "gold_reading": row.get("gold_reading"),
                    "prediction": prediction,
                }
                if prediction is None:
                    rec["raw_prediction"] = raw
                f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
                done += 1
            if done % 100 == 0:
                print(json.dumps({"predicted": done, "total": len(rows)}, ensure_ascii=False), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()
    rows = list(iter_jsonl(args.gold))
    predict(
        rows,
        model_path=args.model,
        adapter_path=args.adapter,
        out=args.out,
        device=args.device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps({"gold": str(args.gold), "out": str(args.out), "rows": len(rows)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
