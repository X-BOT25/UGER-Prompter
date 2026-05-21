#!/usr/bin/env python3
"""Train UGER reading-only with GRPO.

Reward is based on JSON format, candidate legality, gold reading accuracy, and
harm-aware memory decisions from train/dev buckets. Reliability thresholds are
metadata for the upstream gate, not direct reward targets.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                yield line_no, json.loads(line)


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Mapping) and isinstance(completion.get("content"), str):
        return str(completion["content"])
    if isinstance(completion, Sequence) and not isinstance(completion, (str, bytes)):
        return "".join(completion_to_text(item) for item in completion)
    return str(completion)


def extract_json(text: str) -> tuple[dict[str, Any] | None, bool]:
    text = completion_to_text(text).strip()
    try:
        obj = json.loads(text)
        return (obj if isinstance(obj, dict) else None), True
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        try:
            obj = json.loads(text[start : end + 1])
            return (obj if isinstance(obj, dict) else None), False
        except json.JSONDecodeError:
            pass
    return None, False


def normalize_pinyin(value: Any) -> str:
    return str(value or "").strip().lower().replace("u:", "v").replace("眉", "v")


def json_list(value: Any) -> list[Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def pinyin_list(value: Any) -> list[str]:
    return [normalize_pinyin(item) for item in json_list(value) if normalize_pinyin(item)]


def reading_list(value: Any) -> list[str]:
    items = pinyin_list(value)
    if items:
        return items
    normalized = normalize_pinyin(value)
    return [normalized] if normalized else []


def coerce_float(value: Any, default: float = 1.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def score_reading_completion(
    completion: Any,
    *,
    candidate_readings: Any,
    gold_reading: Any,
) -> float:
    raw = completion_to_text(completion).strip()
    parsed, strict_json = extract_json(raw)
    extra_text_penalty = 0.0
    if "\n" in raw:
        extra_text_penalty += 0.3
    if "```" in raw:
        extra_text_penalty += 0.6
    extra_text_penalty += min(0.8, max(0, len(raw) - 80) / 120.0)
    if not parsed:
        return -6.0 - extra_text_penalty
    candidates = set(pinyin_list(candidate_readings))
    gold = pinyin_list(gold_reading)
    reading = pinyin_list(parsed.get("reading"))
    canonical = set(parsed.keys()) == {"reading"} and len(reading) == 1
    candidate_ok = bool(reading) and (not candidates or reading[0] in candidates)
    exact = bool(gold) and reading == gold
    reward = 0.0
    reward += 1.0 if strict_json else -0.8
    reward += 1.0 if canonical else -1.0
    reward += 1.5 if candidate_ok else -5.0
    reward += 10.0 if exact else -4.0
    reward -= extra_text_penalty
    return reward


def parsed_reading(completion: Any) -> list[str]:
    parsed, _ = extract_json(completion_to_text(completion))
    if not parsed:
        return []
    return pinyin_list(parsed.get("reading"))


def first_json_item(value: Any) -> str:
    items = pinyin_list(value)
    if items:
        return items[0]
    return normalize_pinyin(value)


def action_reward(
    completion: Any,
    *,
    case_type: Any,
    pred_first: Any,
    pred_second: Any,
    gold_reading: Any,
    mixed_gold: Any,
    same_gold: Any,
) -> float:
    case = str(case_type or "")
    reading = parsed_reading(completion)
    out = reading[0] if len(reading) == 1 else ""
    first = first_json_item(pred_first)
    second = first_json_item(pred_second)
    gold = first_json_item(gold_reading)
    changed_from_first = bool(out and first and out != first)
    keeps_first = bool(out and first and out == first)
    uses_second = bool(out and second and out == second)
    base_correct = bool(first and gold and first == gold)
    rl_correct = bool(out and gold and out == gold)
    reward = 0.0

    # Candidate-ranker guided rows
    if case == "ranker_preserve":
        if keeps_first and rl_correct:
            reward += 1.4
        elif changed_from_first:
            reward -= 2.8
    elif case == "ranker_repair":
        if changed_from_first and rl_correct:
            reward += 2.8
        elif keeps_first:
            reward -= 1.2
        elif changed_from_first and not rl_correct:
            reward -= 1.4

    # Preserve/repair decomposition
    if base_correct:
        if keeps_first:
            reward += 1.0 if case == "direction_easy_control" else 0.8
        elif not rl_correct:
            reward -= 2.2
        else:
            reward -= 0.2
    else:
        if changed_from_first and rl_correct:
            reward += 2.0
        elif keeps_first:
            reward -= 0.4
        elif changed_from_first and not rl_correct:
            reward -= 0.8

    # Direction harm/help
    if case in {"harm", "direction_harm"} and changed_from_first and not rl_correct:
        reward -= 1.2 if case == "direction_harm" else 0.8
    if case in {"help", "direction_help"} and uses_second:
        reward += 1.8 if case == "direction_help" else 1.5
    if case == "direction_both_wrong" and changed_from_first:
        reward -= 0.5
    if bool(mixed_gold) and reading and len(set(reading)) == 1 and len(reading) > 1:
        reward -= 1.0
    if bool(same_gold) and len(set(reading)) > 1:
        reward -= 1.0
    return reward


def make_reward_func():
    def reward_func(
        completions: list[Any],
        candidate_readings: list[Any] | None = None,
        gold_reading: list[Any] | None = None,
        sample_weight: list[Any] | None = None,
        case_type: list[Any] | None = None,
        best_action: list[Any] | None = None,
        pred_first: list[Any] | None = None,
        pred_second: list[Any] | None = None,
        mixed_gold: list[Any] | None = None,
        same_gold: list[Any] | None = None,
        **_: Any,
    ) -> list[float]:
        candidate_readings = candidate_readings or [None] * len(completions)
        gold_reading = gold_reading or [None] * len(completions)
        sample_weight = sample_weight or [1.0] * len(completions)
        case_type = case_type or [""] * len(completions)
        pred_first = pred_first or [None] * len(completions)
        pred_second = pred_second or [None] * len(completions)
        mixed_gold = mixed_gold or [False] * len(completions)
        same_gold = same_gold or [False] * len(completions)
        _ = best_action
        rewards = []
        for i, comp in enumerate(completions):
            base = score_reading_completion(comp, candidate_readings=candidate_readings[i], gold_reading=gold_reading[i])
            base += action_reward(
                comp,
                case_type=case_type[i],
                pred_first=pred_first[i],
                pred_second=pred_second[i],
                gold_reading=gold_reading[i],
                mixed_gold=mixed_gold[i],
                same_gold=same_gold[i],
            )
            weight = coerce_float(sample_weight[i], 1.0)
            rewards.append(base * weight)
        return rewards
    return reward_func


def load_dataset(path: Path, *, limit: int = 0) -> tuple[Dataset, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    task_hist: Counter[str] = Counter()
    hint_rows = 0
    prompt_chars = 0
    for _, row in iter_jsonl(path):
        prompt = row.get("prompt")
        cands = row.get("candidate_readings")
        gold = reading_list(row.get("gold_reading"))
        if not isinstance(prompt, str) or not prompt.strip() or not isinstance(cands, list) or not gold:
            continue
        pred_first = reading_list(row.get("pred_first") or row.get("pred_reading_first") or row.get("base_pred"))
        pred_second = reading_list(row.get("pred_second") or row.get("rl_pred_or_second_pred"))
        item = {
            "prompt": prompt,
            "candidate_readings": json.dumps(cands, ensure_ascii=False, separators=(",", ":")),
            "gold_reading": json.dumps(gold, ensure_ascii=False, separators=(",", ":")),
            "task_type": str(row.get("task_type") or ""),
            "target_char": str(row.get("target_char") or ""),
            "case_type": str(row.get("case_type") or ""),
            "best_action": str(row.get("best_action") or ""),
            "sample_weight": coerce_float(row.get("sample_weight"), 1.0),
            "pred_first": json.dumps(pred_first, ensure_ascii=False, separators=(",", ":")),
            "pred_second": json.dumps(pred_second, ensure_ascii=False, separators=(",", ":")),
            "mixed_gold": bool(row.get("mixed_gold") or row.get("gold_is_mixed")),
            "same_gold": bool(row.get("same_gold") or row.get("gold_is_same")),
        }
        first_pass = row.get("first_pass")
        if isinstance(first_pass, dict):
            item["first_pass"] = json.dumps(first_pass, ensure_ascii=False, separators=(",", ":"))
        trigger = row.get("memory_trigger")
        if isinstance(trigger, dict):
            item["memory_trigger"] = json.dumps(trigger, ensure_ascii=False, separators=(",", ":"))
            hint_rows += int(bool(trigger.get("hint_found")))
        rows.append(item)
        task_hist[item["task_type"]] += 1
        prompt_chars += len(prompt)
        if limit and len(rows) >= limit:
            break
    if not rows:
        raise RuntimeError(f"empty GRPO dataset: {path}")
    return Dataset.from_list(rows), {
        "rows": len(rows),
        "task_hist": dict(task_hist),
        "case_type_hist": dict(Counter(item.get("case_type", "") for item in rows if item.get("case_type"))),
        "best_action_hist": dict(Counter(item.get("best_action", "") for item in rows if item.get("best_action"))),
        "sample_weight_hist": dict(Counter(str(item.get("sample_weight", 1.0)) for item in rows)),
        "hint_rows": hint_rows,
        "avg_prompt_chars": prompt_chars / len(rows),
        "reward": "format + candidate legality + gold reading exact match + harm-aware memory action terms, weighted by sample_weight",
        "harm_aware_reward": {
            "ranker_preserve_keep_correct": 1.4,
            "ranker_preserve_changed": -2.8,
            "ranker_repair_changed_correct": 2.8,
            "ranker_repair_keep_first": -1.2,
            "ranker_repair_changed_wrong": -1.4,
            "base_correct_keep_first": 0.8,
            "direction_easy_control_keep_first": 1.0,
            "base_correct_changed_wrong": -2.2,
            "base_correct_changed_still_correct": -0.2,
            "base_wrong_changed_correct": 2.0,
            "base_wrong_keep_first": -0.4,
            "base_wrong_changed_wrong": -0.8,
            "harm_changed_wrong": -0.8,
            "direction_harm_changed_wrong": -1.2,
            "help_uses_second": 1.5,
            "direction_help_uses_second": 1.8,
            "direction_both_wrong_changed_from_first": -0.5,
            "mixed_gold_pred_all_same": -1.0,
            "same_gold_pred_forced_mixed": -1.0,
        },
        "threshold_is_reward": False,
    }


def cast_trainables_to_float(model) -> dict[str, int]:
    tensors = 0
    params = 0
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
            tensors += 1
            params += param.numel()
    return {"trainable_tensors_cast_to_float": tensors, "trainable_params": params}


def load_model(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=args.torch_dtype,
        device_map=None if args.device_map.lower() in {"", "none", "null"} else args.device_map,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(
        model,
        args.adapter,
        is_trainable=True,
        autocast_adapter_dtype=False,
    )
    cast_meta = cast_trainables_to_float(model)
    return model, tokenizer, cast_meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--torch-dtype", default="auto")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=3e-7)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--generation-batch-size", type=int, default=0)
    ap.add_argument("--max-completion-length", type=int, default=32)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--torch-empty-cache-steps", type=int, default=0)
    ap.add_argument("--resume-from-checkpoint", default="")
    args = ap.parse_args()

    dataset, dataset_meta = load_dataset(args.data, limit=args.limit)
    model, tokenizer, cast_meta = load_model(args)
    print(
        json.dumps(
            {
                "dataset": dataset_meta,
                "model": {
                    "base_model": args.model,
                    "adapter": args.adapter,
                    "adapter_mode": "continue_training_reading_only_adapter",
                    "device_map": args.device_map,
                    "learning_rate": args.learning_rate,
                    "num_generations": args.num_generations,
                    "max_completion_length": args.max_completion_length,
                    **cast_meta,
                },
                "output_schema": {"keys": ["reading"], "example": {"reading": ["de5"]}},
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    train_args = GRPOConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_generations=args.num_generations,
        generation_batch_size=args.generation_batch_size or None,
        max_completion_length=args.max_completion_length,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        torch_empty_cache_steps=args.torch_empty_cache_steps or None,
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=make_reward_func(),
        args=train_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(args.out)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.out)
        metadata = {
            "base_model": args.model,
            "input_adapter": args.adapter,
            "output_adapter": args.out,
            "dataset": dataset_meta,
            "training": {
                "epochs": args.epochs,
                "max_steps": args.max_steps,
                "batch": args.batch,
                "grad_accum": args.grad_accum,
                "learning_rate": args.learning_rate,
                "num_generations": args.num_generations,
                "generation_batch_size": args.generation_batch_size or None,
                "max_completion_length": args.max_completion_length,
            },
            "reward": {
                "format_json_reading": True,
                "candidate_legality": True,
                "gold_exact_match": True,
                "harm_aware_memory_action": {
                    "ranker_preserve_keep_correct": 1.4,
                    "ranker_preserve_changed": -2.8,
                    "ranker_repair_changed_correct": 2.8,
                    "ranker_repair_keep_first": -1.2,
                    "ranker_repair_changed_wrong": -1.4,
                    "base_correct_keep_first": 0.8,
                    "direction_easy_control_keep_first": 1.0,
                    "base_correct_changed_wrong": -2.2,
                    "base_correct_changed_still_correct": -0.2,
                    "base_wrong_changed_correct": 2.0,
                    "base_wrong_keep_first": -0.4,
                    "base_wrong_changed_wrong": -0.8,
                    "harm_changed_wrong": -0.8,
                    "direction_harm_changed_wrong": -1.2,
                    "help_uses_second": 1.5,
                    "direction_help_uses_second": 1.8,
                    "direction_both_wrong_changed_from_first": -0.5,
                    "mixed_gold_pred_all_same": -1.0,
                    "same_gold_pred_forced_mixed": -1.0,
                },
                "threshold_is_reward": False,
            },
        }
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "reading_only_grpo_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Saved reading-only GRPO adapter to {args.out}", flush=True)


if __name__ == "__main__":
    main()
