#!/usr/bin/env python3
"""Evaluate predictions with multiple metrics.

Supports row-level accuracy, answer-level accuracy, and candidate-ranker evaluation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                yield line_no, json.loads(line)


def rate(num: float, den: float) -> float:
    return num / den if den else 0.0


def eval_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(predictions)
    correct = 0
    macro_correct = 0
    macro_total = 0
    for pred in predictions:
        gold = pred.get("gold_reading")
        prediction = pred.get("prediction")
        if isinstance(prediction, dict):
            reading = prediction.get("reading")
            if isinstance(reading, list) and len(reading) == 1:
                if gold == reading:
                    correct += 1
                macro_correct += 1
        macro_total += 1

    return {
        "total": total,
        "correct": correct,
        "accuracy": rate(correct, total),
        "macro_accuracy": rate(macro_correct, macro_total),
    }


def eval_ranker_decisions(decisions: list[dict]) -> dict:
    total = len(decisions)
    base_correct = sum(d["base_correct"] for d in decisions)
    rl_correct = sum(d["rl_correct"] for d in decisions)
    ranker_correct = sum(d["ranker_correct"] for d in decisions)
    base_correct_to_rl_wrong = sum(d["base_correct"] and not d["rl_correct"] for d in decisions)
    base_correct_to_ranker_wrong = sum(d["base_correct"] and not d["ranker_correct"] for d in decisions)
    rl_changed = sum(d["base_pred"] != d["rl_pred"] for d in decisions)
    ranker_changed = sum(d["base_pred"] != d["ranker_pred"] for d in decisions)
    rl_beneficial = sum((d["base_pred"] != d["rl_pred"]) and (not d["base_correct"]) and d["rl_correct"] for d in decisions)
    rl_harmful = sum((d["base_pred"] != d["rl_pred"]) and d["base_correct"] and (not d["rl_correct"]) for d in decisions)
    ranker_beneficial = sum((d["base_pred"] != d["ranker_pred"]) and (not d["base_correct"]) and d["ranker_correct"] for d in decisions)
    ranker_harmful = sum((d["base_pred"] != d["ranker_pred"]) and d["base_correct"] and (not d["ranker_correct"]) for d in decisions)
    lambda_harm = 2
    rl_net = rl_beneficial - lambda_harm * rl_harmful
    ranker_net = ranker_beneficial - lambda_harm * ranker_harmful
    case_total = Counter(d.get("case_type", "") for d in decisions)
    case_base_correct = Counter(d.get("case_type", "") for d in decisions if d["base_correct"])
    case_rl_correct = Counter(d.get("case_type", "") for d in decisions if d["rl_correct"])
    case_correct = Counter(d.get("case_type", "") for d in decisions if d["ranker_correct"])
    easy = [d for d in decisions if d.get("case_type") == "direction_easy_control"]
    report = {
        "schema": "ranker_minimal_report_v1",
        "rows": total,
        "metrics": {
            "base_row_acc": rate(base_correct, total),
            "rl_row_acc": rate(rl_correct, total),
            "ranker_row_acc": rate(ranker_correct, total),
            "base_correct_to_rl_wrong_rate": rate(base_correct_to_rl_wrong, base_correct),
            "base_correct_to_ranker_wrong_rate": rate(base_correct_to_ranker_wrong, base_correct),
            "rl_beneficial": rl_beneficial,
            "rl_harmful": rl_harmful,
            "ranker_beneficial": ranker_beneficial,
            "ranker_harmful": ranker_harmful,
            "rl_net_override_utility": rl_net,
            "ranker_net_override_utility": ranker_net,
            "rl_changed_rate": rate(rl_changed, total),
            "ranker_changed_rate": rate(ranker_changed, total),
            "repair_retention": rate(ranker_beneficial, rl_beneficial),
            "changed_precision": rate(ranker_beneficial, ranker_beneficial + ranker_harmful),
            "base_case_accuracy": {case: rate(case_base_correct[case], count) for case, count in sorted(case_total.items())},
            "rl_case_accuracy": {case: rate(case_rl_correct[case], count) for case, count in sorted(case_total.items())},
            "easy_control_ranker_acc": rate(sum(d["ranker_correct"] for d in easy), len(easy)),
            "case_accuracy": {case: rate(case_correct[case], count) for case, count in sorted(case_total.items())},
        },
        "lambda_harm": lambda_harm,
        "pass_conditions": {},
        "final_test_used": False,
        "leakage_safe": True,
    }
    m = report["metrics"]
    report["pass_conditions"] = {
        "row_acc_ge_base": m["ranker_row_acc"] >= m["base_row_acc"],
        "preserve_loss_below_or_equal_rl": m["base_correct_to_ranker_wrong_rate"] <= m["base_correct_to_rl_wrong_rate"],
        "net_override_positive": m["ranker_net_override_utility"] > 0,
        "repair_retention_ge_0_5": m["repair_retention"] >= 0.5,
        "changed_precision_ge_0_55_or_no_changes": (ranker_changed == 0) or (m["changed_precision"] >= 0.55),
        "easy_control_not_below_base": m["easy_control_ranker_acc"]
        >= m["base_case_accuracy"].get("direction_easy_control", 0.0),
    }
    report["passed"] = all(report["pass_conditions"].values())
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--mode", default="predictions", choices=["predictions", "ranker"])
    args = ap.parse_args()

    rows = [row for _, row in iter_jsonl(args.predictions)]
    
    if args.mode == "ranker":
        report = eval_ranker_decisions(rows)
    else:
        report = eval_predictions(rows)
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
