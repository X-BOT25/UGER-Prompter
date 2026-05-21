#!/usr/bin/env python3
"""Build UGER experience pool from scored position rows.

Selection:
- include all wrong train predictions;
- include correct-but-high-entropy rows with H_norm >= threshold.

The pool keeps retrieval/ranking fields internally, but prompt-facing hints are
light natural-language rules. It deliberately does not copy source sentences or
position metadata into the experience entries.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def plain_from_marked(text: str) -> str:
    return text.replace("[", "").replace("]", "")


def bracket_spans(marked: str) -> list[tuple[str, int, int, str]]:
    plain_parts: list[str] = []
    spans: list[tuple[str, int, int, str]] = []
    i = 0
    plain_pos = 0
    while i < len(marked):
        if marked[i] == "[":
            j = marked.find("]", i + 1)
            if j < 0:
                break
            char = marked[i + 1 : j]
            start = plain_pos
            plain_parts.append(char)
            plain_pos += len(char)
            spans.append((char, start, start + len(char), ""))
            i = j + 1
        else:
            plain_parts.append(marked[i])
            plain_pos += 1
            i += 1
    plain = "".join(plain_parts)
    return [(char, start, end, plain) for char, start, end, _ in spans]


def context_window(marked: str, width: int = 4) -> tuple[str, str]:
    spans = bracket_spans(marked)
    if not spans:
        return "", plain_from_marked(marked)
    char, start, end, plain = spans[0]
    left = plain[max(0, start - width) : start]
    right = plain[end : min(len(plain), end + width)]
    return f"{left}{char}{right}", plain


def compact(value: Any, limit: int) -> str:
    return " ".join(str(value or "").strip().split())[:limit]


def semantic_label(char: str, reading: str, window: str) -> str:
    if char == "为":
        if reading == "wei4":
            return "原因/目的结构"
        if reading == "wei2":
            return "认定/作为/成为结构"
    if char == "的" and reading == "de5":
        return "结构助词"
    if char == "地" and reading == "de5":
        return "状语助词"
    if char == "得" and reading == "de5":
        return "补语助词"
    if char == "了":
        if reading == "le5":
            return "完成/变化助词"
        if reading == "liao3":
            return "清楚/结束义项"
    return "按语义功能判断"


def hint_pair(char: str, gold: str, wrong: str, pos_label: str, neg_label: str) -> tuple[str, str]:
    positive = f""{char}"在{pos_label}中多读 {gold}。"
    negative = f"不要和{neg_label}中的 {wrong} 混淆；多个相同目标字也要逐位置判断。"
    return positive, negative


def quality_score(correct: bool, h_norm: float, p_conf: float) -> float:
    if correct:
        return max(0.5, min(0.99, 0.55 + 0.4 * h_norm + 0.05 * p_conf))
    return max(0.55, min(0.99, 0.6 + 0.25 * (1.0 - p_conf) + 0.15 * h_norm))


def build_entry(
    idx: int,
    row: dict[str, Any],
    *,
    high_entropy_threshold: float,
) -> dict[str, Any] | None:
    char = str(row.get("target_char") or "")
    gold = (row.get("gold_reading") or [""])[0]
    pred = (row.get("pred_reading") or [""])[0]
    cands = [str(x) for x in row.get("candidate_readings") or []]
    h_norm = float(row.get("h_norm") or 0.0)
    p_conf = float(row.get("p_conf") or 0.0)
    correct = bool(row.get("correct"))
    if correct and h_norm < high_entropy_threshold:
        return None
    if not char or not gold or gold not in cands:
        return None
    wrong = pred if pred and pred != gold else next((x for x in cands if x != gold), "")
    if not wrong:
        return None
    window, plain = context_window(str(row.get("input") or ""))
    pos_label = semantic_label(char, gold, window)
    neg_label = semantic_label(char, wrong, window)
    pos_hint, neg_hint = hint_pair(char, gold, wrong, pos_label, neg_label)
    pool_type = "correct_high_entropy" if correct else "wrong_prediction"
    if str(row.get("task_type")) == "position_from_same_char_mixed":
        pool_type = "mixed_assignment_" + pool_type
    retrieval_text = f"{char} {gold} {wrong} {pos_label} {neg_label} {pool_type}"
    embedding_text = f"{char}在{pos_label}中读{gold}，易与{neg_label}读{wrong}混淆。"
    return {
        "id": f"exp_{idx:06d}",
        "pool_type": pool_type,
        "char": char,
        "gold_reading": gold,
        "wrong_reading": wrong,
        "candidate_readings": cands,
        "semantic_label": pos_label,
        "confusable_semantic": neg_label,
        "error_pattern": ("correct_high_entropy" if correct else f"{wrong}_to_{gold}"),
        "hint_positive": pos_hint,
        "hint_negative": neg_hint,
        "hint": pos_hint + neg_hint,
        "retrieval_text": retrieval_text,
        "embedding_text": embedding_text,
        "h_norm": h_norm,
        "p_conf": p_conf,
        "quality_score": quality_score(correct, h_norm, p_conf),
        "source": "train_scored",
        "selection_rule": "wrong_or_correct_high_entropy",
        "leakage_safe": True,
    }


def contains_forbidden_context(entry: dict[str, Any]) -> bool:
    text = json.dumps(entry, ensure_ascii=False)
    forbidden_keys = ("input", "sent", "sentence", "source_line", "source_rows", "sentence_id", "start", "end", "prompt", "completion")
    if any(f'"{key}"' in text for key in forbidden_keys):
        return True
    return bool(re.search(r"[，。；！？].*\[.\].*[，。；！？]", text))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scored", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--summary-out", required=True, type=Path)
    ap.add_argument("--min-h-norm", type=float, default=0.55)
    ap.add_argument("--max-per-char", type=int, default=80)
    args = ap.parse_args()

    entries: list[dict[str, Any]] = []
    per_char: Counter[str] = Counter()
    input_rows = 0
    wrong_rows = 0
    correct_high_entropy_rows = 0
    rejected_for_context = 0
    for row in iter_jsonl(args.scored):
        input_rows += 1
        correct = bool(row.get("correct"))
        h_norm = float(row.get("h_norm") or 0.0)
        if not correct:
            wrong_rows += 1
        elif h_norm >= args.min_h_norm:
            correct_high_entropy_rows += 1
        entry = build_entry(len(entries) + 1, row, high_entropy_threshold=args.min_h_norm)
        if not entry:
            continue
        char = entry["char"]
        if per_char[char] >= args.max_per_char:
            continue
        if contains_forbidden_context(entry):
            rejected_for_context += 1
            continue
        per_char[char] += 1
        entries.append(entry)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "schema": "uger_experience_pool_summary_v1",
        "input_rows": input_rows,
        "wrong_rows": wrong_rows,
        "correct_high_entropy_rows": correct_high_entropy_rows,
        "min_h_norm": args.min_h_norm,
        "entries": len(entries),
        "chars": len(per_char),
        "pool_type_counts": dict(Counter(e["pool_type"] for e in entries)),
        "rejected_for_context": rejected_for_context,
        "contains_source_sentence": False,
        "contains_position_offsets": False,
        "leakage_safe": True,
    }
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
