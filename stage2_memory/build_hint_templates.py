#!/usr/bin/env python3
"""Build lightweight UGER hint templates from the experience pool.

The output is one template per character. It is intended for prompt injection
after the memory gate fires, and contains only abstract reading conditions, not
source sentences or training offsets.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def clean_condition(text: str) -> str:
    text = " ".join(str(text or "").strip().split())
    replacements = {
        "原因/目的结构": "表示原因/目的，如"因为、为了、为何"",
        "认定/作为/成为结构": "表示认定/归类/作为，如"认为、作为、被认为"",
        "完成/变化助词": "作完成、变化或语气助词",
        "清楚/结束义项": "表示清楚、了解或结束",
        "结构助词": "作结构助词",
        "状语助词": "作状语助词，修饰后面的动作",
        "补语助词": "作补语助词，连接动作与结果/程度",
    }
    if text in replacements:
        return replacements[text]
    if text.startswith("固定搭配"") and text.endswith("""):
        phrase = text.removeprefix("固定搭配"")
        return f"出现在固定搭配"{phrase}"中"
    return text or "根据当前位置语义判断"


def letter(idx: int) -> str:
    return chr(ord("A") + idx)


def build_templates(rows: list[dict[str, Any]], *, max_conditions_per_reading: int, min_count: int) -> list[dict[str, Any]]:
    by_char: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_char[str(row.get("char") or "")].append(row)

    templates: list[dict[str, Any]] = []
    for char, items in sorted(by_char.items()):
        if not char:
            continue
        candidates: list[str] = []
        reading_conditions: dict[str, Counter[str]] = defaultdict(Counter)
        pool_counts = Counter(str(item.get("pool_type") or "") for item in items)
        for item in items:
            for cand in item.get("candidate_readings") or []:
                cand = str(cand)
                if cand and cand not in candidates:
                    candidates.append(cand)
            reading = str(item.get("gold_reading") or "")
            condition = item.get("collocation") or item.get("semantic_label") or ""
            if item.get("collocation"):
                condition = f"固定搭配"{condition}""
            if reading and condition:
                reading_conditions[reading][clean_condition(condition)] += 1
        lines: list[str] = []
        conditions: list[dict[str, Any]] = []
        idx = 0
        for reading in candidates:
            cond_counts = reading_conditions.get(reading, Counter())
            selected = [(cond, count) for cond, count in cond_counts.most_common(max_conditions_per_reading) if count >= min_count]
            if not selected and cond_counts:
                selected = cond_counts.most_common(1)
            for cond, count in selected:
                tag = letter(idx)
                line = f"{tag}. 若{cond}，读 {reading}。"
                lines.append(line)
                conditions.append({"label": tag, "reading": reading, "condition": cond, "support_count": count})
                idx += 1
        if not lines:
            continue
        if len(candidates) > 1:
            lines.append("多个 [] 按出现顺序分别判断。")
        hint = "[Hint]\n" + f"请根据"{char}"的语义条件选择读音：\n" + "\n".join(lines)
        templates.append({
            "schema": "uger_hint_template_v1",
            "char": char,
            "candidate_readings": candidates,
            "conditions": conditions,
            "hint": hint,
            "retrieval_text": f"{char} " + " ".join(candidates) + " " + " ".join(c["condition"] for c in conditions),
            "source": "experience_pool_h07",
            "pool_entry_count": len(items),
            "pool_type_counts": dict(pool_counts),
            "leakage_safe": True,
        })
    return templates


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--out-jsonl", required=True, type=Path)
    ap.add_argument("--summary-out", required=True, type=Path)
    ap.add_argument("--max-conditions-per-reading", type=int, default=2)
    ap.add_argument("--min-count", type=int, default=2)
    args = ap.parse_args()

    rows = list(iter_jsonl(args.pool))
    templates = build_templates(rows, max_conditions_per_reading=args.max_conditions_per_reading, min_count=args.min_count)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(templates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for template in templates:
            f.write(json.dumps(template, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "schema": "uger_hint_template_summary_v1",
        "pool_entries": len(rows),
        "templates": len(templates),
        "max_conditions_per_reading": args.max_conditions_per_reading,
        "min_count": args.min_count,
        "leakage_safe": True,
    }
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
