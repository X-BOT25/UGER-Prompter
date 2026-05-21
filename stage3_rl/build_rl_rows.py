#!/usr/bin/env python3
"""Build UGER RL rows with dynamic memory-gated hints.

Gate:
    score = alpha * P_conf + beta * (1 - H_norm)
    use hint iff score < tau_memory

The gate is a fixed routing decision for memory retrieval. It is written to
metadata for auditability, but it is not a reward and is not updated by RL.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_hint_templates(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        char = str(row.get("char") or "")
        hint = str(row.get("hint") or "").strip()
        if char and hint:
            out[char] = row
    return out


def extract_reading(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if isinstance(value, list) and value:
        return str(value[0])
    if isinstance(value, str):
        return value
    return ""


def gate_score(p_conf: float, h_norm: float, alpha: float, beta: float) -> float:
    return alpha * p_conf + beta * (1.0 - h_norm)


def should_use_hint(row: dict[str, Any], *, alpha: float, beta: float, tau_memory: float) -> tuple[bool, float]:
    p_conf = float(row.get("p_conf") or 0.0)
    h_norm = float(row.get("h_norm") or 0.0)
    score = gate_score(p_conf, h_norm, alpha, beta)
    return score < tau_memory, score


def strip_existing_hint(prompt: str) -> str:
    marker = "\n[Hint]\n"
    if marker in prompt:
        return prompt.split(marker, 1)[0].rstrip()
    return prompt.rstrip()


def inject_hint(prompt: str, hint: str) -> str:
    return strip_existing_hint(prompt) + "\n" + hint.strip()


def build_prompt(row: dict[str, Any], train_by_line: dict[int, dict[str, Any]], hint: str | None) -> str | None:
    line_no = int(row.get("source_line") or 0)
    train = train_by_line.get(line_no)
    if not train:
        return None
    prompt = str(train.get("prompt") or "")
    if not prompt:
        return None
    if hint:
        prompt = inject_hint(prompt, hint)
    else:
        prompt = strip_existing_hint(prompt)
    return prompt


def completion(reading: str) -> str:
    return json.dumps({"reading": [reading]}, ensure_ascii=False, separators=(",", ":"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True, type=Path, help="Original train_position_first.jsonl with prompts.")
    ap.add_argument("--scored", required=True, type=Path, help="Train candidate scoring with p_conf/h_norm.")
    ap.add_argument("--hint-templates", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--summary-out", required=True, type=Path)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--beta", type=float, default=0.4)
    ap.add_argument("--tau-memory", type=float, default=0.55)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not math.isclose(args.alpha + args.beta, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("alpha + beta should equal 1.0 for the normalized confidence gate")

    train_by_line: dict[int, dict[str, Any]] = {}
    for line_no, row in enumerate(iter_jsonl(args.train), 1):
        train_by_line[line_no] = row
    templates = load_hint_templates(args.hint_templates)

    rows: list[dict[str, Any]] = []
    task_hist: Counter[str] = Counter()
    gate_checked = 0
    gate_triggered = 0
    hint_found = 0
    hint_missing = 0
    no_hint_prompt = 0
    wrong_rows = 0
    correct_rows = 0

    for scored in iter_jsonl(args.scored):
        if args.limit and len(rows) >= args.limit:
            break
        char = str(scored.get("target_char") or "")
        gold = extract_reading(scored, "gold_reading")
        if not char or not gold:
            continue
        use_hint, score = should_use_hint(scored, alpha=args.alpha, beta=args.beta, tau_memory=args.tau_memory)
        gate_checked += 1
        hint_text = None
        template_id = None
        if use_hint:
            gate_triggered += 1
            template = templates.get(char)
            if template:
                hint_text = str(template.get("hint") or "").strip()
                template_id = template.get("char")
                hint_found += 1
            else:
                hint_missing += 1
        else:
            no_hint_prompt += 1
        prompt = build_prompt(scored, train_by_line, hint_text)
        if not prompt:
            continue
        correct = bool(scored.get("correct"))
        wrong_rows += int(not correct)
        correct_rows += int(correct)
        task_type = str(scored.get("task_type") or "position")
        rows.append({
            "schema": "uger_rl_row_v1",
            "task_type": task_type,
            "target_char": char,
            "candidate_readings": scored.get("candidate_readings") or [],
            "prompt": prompt,
            "completion": completion(gold),
            "gold_reading": [gold],
            "pred_reading": scored.get("pred_reading"),
            "p_conf": scored.get("p_conf"),
            "h_norm": scored.get("h_norm"),
            "memory_gate": {
                "alpha": args.alpha,
                "beta": args.beta,
                "tau_memory": args.tau_memory,
                "score": score,
                "triggered": use_hint,
                "hint_found": bool(hint_text),
                "template_char": template_id,
                "note": "fixed routing gate only; not a reward term",
            },
        })
        task_hist[task_type] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "schema": "uger_rl_summary_v1",
        "rows": len(rows),
        "task_hist": dict(task_hist),
        "gate_formula": "alpha * P_conf + beta * (1 - H_norm) < tau_memory",
        "alpha": args.alpha,
        "beta": args.beta,
        "tau_memory": args.tau_memory,
        "gate_checked": gate_checked,
        "gate_triggered": gate_triggered,
        "hint_found": hint_found,
        "hint_missing": hint_missing,
        "no_hint_prompt": no_hint_prompt,
        "correct_rows": correct_rows,
        "wrong_rows": wrong_rows,
        "hint_source": str(args.hint_templates),
        "threshold_is_reward": False,
        "leakage_safe": True,
    }
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
