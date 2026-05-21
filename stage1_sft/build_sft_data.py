#!/usr/bin/env python3
"""Build position-first SFT rows.

Every training row contains exactly one bracketed target and asks the model to
emit only {"reading":[one_value]}. Multi-position data is expanded into
independent position rows and can be merged at inference time.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_candidates(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for row in iter_jsonl(path):
        char = row.get("char") or row.get("字")
        raw = row.get("union_pinyins") or row.get("pinyins") or row.get("readings") or []
        if not isinstance(char, str) or len(char) != 1 or not isinstance(raw, list):
            continue
        vals: list[str] = []
        for item in raw:
            if isinstance(item, str) and item.strip():
                reading = item.strip().lower().replace("u:", "v")
                if reading not in vals:
                    vals.append(reading)
        if vals:
            out[char] = vals
    return out


def safe_text(text: str) -> str:
    return text.replace("[", "【").replace("]", "】")


def mark_one(sentence: str, start: int, end: int) -> str:
    return safe_text(sentence[:start]) + "[" + safe_text(sentence[start:end]) + "]" + safe_text(sentence[end:])


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


def compact(value: Any, limit: int) -> str:
    return " ".join(str(value or "").strip().split())[:limit]


def kg_lines(char: str, candidates: list[str]) -> list[str]:
    lines = ["K: 候选读音及知识摘要："]
    for reading in candidates:
        lines.append(f"- {reading}: 根据上下文判断")
    return lines


def prompt(marked_input: str, char: str, candidates: list[str]) -> str:
    return "\n".join([
        "Task: 判断方括号[]标记的目标字读音。",
        f"Input: {marked_input}",
        f"C: {char}",
        f"Candidates: {json.dumps(candidates, ensure_ascii=False)}",
        *kg_lines(char, candidates),
        'Output JSON only: {"reading":["一个候选读音"]}',
    ])


def completion(reading: str) -> str:
    return json.dumps({"reading": [reading]}, ensure_ascii=False, separators=(",", ":"))


def build_row(sentence: str, start: int, end: int, char: str, gold: str, candidates: list[str]) -> dict[str, Any]:
    marked = mark_one(sentence, start, end)
    return {
        "schema": "position_first_sft_v1",
        "task_type": "position",
        "target_char": char,
        "candidate_readings": candidates,
        "input": marked,
        "gold_reading": [gold],
        "prompt": prompt(marked, char, candidates),
        "completion": completion(gold),
    }


def build_same_mixed_rows(
    row: dict[str, Any],
    line_no: int,
    candidate_map: dict[str, list[str]],
) -> list[dict[str, Any]]:
    marked = str(row.get("sent") or "")
    answers = row.get("answer")
    if not marked or not isinstance(answers, list):
        return []
    spans = bracket_spans(marked)
    if len(spans) != len(answers):
        return []
    rows: list[dict[str, Any]] = []
    for pos, ((char, start, end, plain), answer) in enumerate(zip(spans, answers), 1):
        if not char or len(char) != 1:
            continue
        reading = str(answer).strip().lower().replace("u:", "v")
        candidates = list(candidate_map.get(char, []))
        if reading not in candidates:
            candidates.append(reading)
        one_input = mark_one(plain, start, end)
        rows.append({
            "schema": "position_first_sft_v1",
            "task_type": "position_from_same_char_mixed",
            "source": "train_same_char_mixed",
            "source_rows": [f"train_same_char_mixed:{line_no}#{pos}"],
            "target_char": char,
            "candidate_readings": candidates,
            "input": one_input,
            "gold_reading": [reading],
            "group_input": marked,
            "group_position": pos,
            "prompt": prompt(one_input, char, candidates),
            "completion": completion(reading),
        })
    return rows


def validate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    forbidden = ("internal_candidate", "pinyin_sequence")
    for i, row in enumerate(rows, 1):
        text = str(row.get("prompt", "")) + str(row.get("completion", ""))
        if row.get("input", "").count("[") != 1:
            errors.append({"line": i, "error": "not_single_bracket"})
        if any(token in text for token in forbidden):
            errors.append({"line": i, "error": "old_interface_token"})
        try:
            comp = json.loads(row["completion"])
        except Exception:
            errors.append({"line": i, "error": "bad_completion"})
            continue
        if set(comp) != {"reading"} or len(comp.get("reading") or []) != 1:
            errors.append({"line": i, "error": "bad_completion_shape"})
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--candidates", required=True, type=Path)
    ap.add_argument("--same-char-mixed", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--summary-out", required=True, type=Path)
    ap.add_argument("--same-mixed-repeat", type=int, default=2)
    args = ap.parse_args()

    candidate_map = load_candidates(args.candidates)
    out_rows: list[dict[str, Any]] = []

    # Load main training data
    for line_no, row in enumerate(iter_jsonl(args.input), 1):
        sentence = row.get("sentence", "")
        start = row.get("start", 0)
        end = row.get("end", 0)
        char = row.get("target_char", "")
        gold = row.get("gold_pinyin", "").strip().lower().replace("u:", "v")
        candidates = candidate_map.get(char, [])
        if gold in candidates:
            out_rows.append(build_row(sentence, start, end, char, gold, candidates))

    # Load same-char mixed data
    mixed_source = 0
    for line_no, row in enumerate(iter_jsonl(args.same_char_mixed), 1):
        rows = build_same_mixed_rows(row, line_no, candidate_map)
        if rows:
            mixed_source += 1
        for _ in range(max(1, args.same_mixed_repeat)):
            out_rows.extend(dict(item) for item in rows)

    errors = validate(out_rows)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    if errors:
        args.summary_out.write_text(json.dumps({"error_count": len(errors), "errors": errors[:100]}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(f"format validation failed: {len(errors)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    counts = Counter(str(row.get("task_type")) for row in out_rows)
    summary = {
        "schema": "position_first_sft_summary_v1",
        "rows": len(out_rows),
        "task_type_counts": dict(counts),
        "same_mixed_source_rows": mixed_source,
        "same_mixed_repeat": max(1, args.same_mixed_repeat),
        "model_outputs": ["reading"],
    }
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
