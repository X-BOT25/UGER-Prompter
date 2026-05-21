#!/usr/bin/env python3
"""Score position-first candidates with cross-entropy likelihood.

Produces prediction, confidence, normalized entropy, and correctness for
experience pool building. Uses per-candidate completion likelihood.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                row = json.loads(line)
                row["_line_no"] = line_no
                yield row


def completion(reading: str) -> str:
    return json.dumps({"reading": [reading]}, ensure_ascii=False, separators=(",", ":"))


def softmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    m = max(scores)
    exps = [math.exp(max(-80.0, min(80.0, x - m))) for x in scores]
    z = sum(exps) or 1.0
    return [x / z for x in exps]


def entropy_norm(probs: list[float]) -> float:
    if len(probs) <= 1:
        return 0.0
    h = -sum(p * math.log(max(p, 1e-12)) for p in probs)
    return h / math.log(len(probs))


def score_rows(
    rows: list[dict[str, Any]],
    *,
    model_path: str,
    adapter_path: str,
    out: Path,
    device: str,
    batch_size: int,
    max_len: int,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

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
        for row_start in range(0, len(rows), batch_size):
            row_batch = rows[row_start : row_start + batch_size]
            pair_meta: list[tuple[int, str, int]] = []
            texts: list[str] = []
            prompt_lens: list[int] = []
            for local_idx, row in enumerate(row_batch):
                prompt = str(row.get("prompt") or "")
                prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                for cand in row.get("candidate_readings") or []:
                    cand = str(cand)
                    comp = completion(cand) + tokenizer.eos_token
                    comp_ids = tokenizer(comp, add_special_tokens=False)["input_ids"]
                    pair_meta.append((local_idx, cand, len(comp_ids)))
                    texts.append(prompt + comp)
                    prompt_lens.append(len(prompt_ids))
            if not texts:
                continue
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len, add_special_tokens=False)
            labels = inputs["input_ids"].clone()
            for i, prompt_len in enumerate(prompt_lens):
                labels[i, : min(prompt_len, labels.shape[1])] = -100
                labels[i, inputs["attention_mask"][i] == 0] = -100
            if device not in {"cpu", "none"}:
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                labels = labels.to(model.device)
            with torch.no_grad():
                logits = model(**inputs).logits
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="none",
                    ignore_index=-100,
                ).view(shift_labels.shape)
                token_counts = (shift_labels != -100).sum(dim=1).clamp_min(1)
                nll = loss.sum(dim=1)
                avg_nll = nll / token_counts
            per_row: list[list[dict[str, Any]]] = [[] for _ in row_batch]
            for pair_i, (local_idx, cand, token_count) in enumerate(pair_meta):
                score = -float(avg_nll[pair_i].detach().cpu())
                per_row[local_idx].append({
                    "reading": cand,
                    "score": score,
                    "avg_nll": float(avg_nll[pair_i].detach().cpu()),
                    "tokens": int(token_count),
                })
            for row, cand_scores in zip(row_batch, per_row):
                probs = softmax([x["score"] for x in cand_scores])
                for item, prob in zip(cand_scores, probs):
                    item["prob"] = prob
                best = max(cand_scores, key=lambda x: x["prob"]) if cand_scores else {"reading": None, "prob": 0.0}
                gold = (row.get("gold_reading") or [None])[0]
                h_norm = entropy_norm(probs)
                out_row = {
                    "schema": "uger_candidate_scored_v1",
                    "source": "train_position_first",
                    "source_line": row.get("_line_no"),
                    "source_rl_line": row.get("source_rl_line"),
                    "task_type": row.get("task_type"),
                    "target_char": row.get("target_char"),
                    "candidate_readings": row.get("candidate_readings"),
                    "gold_reading": [gold],
                    "pred_reading": [best["reading"]],
                    "p_conf": best["prob"],
                    "h_norm": h_norm,
                    "correct": best["reading"] == gold,
                    "candidate_scores": cand_scores,
                    "input": row.get("input"),
                }
                f.write(json.dumps(out_row, ensure_ascii=False, separators=(",", ":")) + "\n")
                done += 1
            if done % 1000 == 0:
                print(json.dumps({"scored": done, "total": len(rows)}, ensure_ascii=False), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    rows = list(iter_jsonl(args.data))
    if args.limit:
        rows = rows[: args.limit]
    score_rows(
        rows,
        model_path=args.model,
        adapter_path=args.adapter,
        out=args.out,
        device=args.device,
        batch_size=args.batch_size,
        max_len=args.max_len,
    )
    print(json.dumps({"data": str(args.data), "out": str(args.out), "rows": len(rows)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
