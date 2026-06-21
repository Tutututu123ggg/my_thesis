from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def to_int(value: Any, default: int = -1) -> int:
    try:
        s = str(value or "").strip()
        if not s:
            return default
        return int(float(s))
    except Exception:
        return default


def first_existing(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def get_word_count(row: dict[str, Any]) -> int:
    return to_int(first_existing(row, ["word_count", "words", "num_words"]), default=-1)


def get_heading_count(row: dict[str, Any]) -> int:
    return to_int(first_existing(row, ["heading_count", "num_headings"]), default=-1)


def classify_length(row: dict[str, Any], min_words: int, soft_words: int) -> tuple[str, str]:
    word_count = get_word_count(row)
    heading_count = get_heading_count(row)

    if word_count < 0:
        return "KEEP_UNKNOWN_LENGTH", "Không có word_count, giữ lại để tránh loại nhầm."

    if word_count < min_words:
        return "AUTO_REJECT_SHORT", f"Bài quá ngắn: word_count={word_count} < {min_words}."

    if word_count < soft_words:
        return "SHORT_KEEP_FOR_GPT", f"Bài hơi ngắn: word_count={word_count}, giữ để GPT content-level xét."

    if heading_count == 0:
        return "KEEP_NO_HEADING_FLAG", f"word_count={word_count}, nhưng heading_count=0, giữ và flag để kiểm tra."

    return "KEEP_FOR_GPT", f"word_count={word_count}, đủ dài để lọc content-level."


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", default="data/debug/gpt_filter/length_filter_common_v1")
    parser.add_argument("--min-words", type=int, default=350)
    parser.add_argument("--soft-words", type=int, default=600)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    extra_fields = ["length_decision", "length_reason", "length_crawl_keep"]
    output_fields = list(fieldnames)

    for field in extra_fields:
        if field not in output_fields:
            output_fields.append(field)

    kept_rows = []
    rejected_rows = []
    all_rows = []

    stats = {
        "total": 0,
        "kept": 0,
        "auto_rejected_short": 0,
        "keep_unknown_length": 0,
        "short_keep_for_gpt": 0,
        "keep_no_heading_flag": 0,
        "keep_for_gpt": 0,
        "min_words": args.min_words,
        "soft_words": args.soft_words,
    }

    for row in rows:
        stats["total"] += 1

        decision, reason = classify_length(
            row=row,
            min_words=args.min_words,
            soft_words=args.soft_words,
        )

        out = dict(row)
        out["length_decision"] = decision
        out["length_reason"] = reason
        out["length_crawl_keep"] = "0" if decision == "AUTO_REJECT_SHORT" else "1"

        all_rows.append(out)

        if decision == "AUTO_REJECT_SHORT":
            rejected_rows.append(out)
            stats["auto_rejected_short"] += 1
        else:
            kept_rows.append(out)
            stats["kept"] += 1

            if decision == "KEEP_UNKNOWN_LENGTH":
                stats["keep_unknown_length"] += 1
            elif decision == "SHORT_KEEP_FOR_GPT":
                stats["short_keep_for_gpt"] += 1
            elif decision == "KEEP_NO_HEADING_FLAG":
                stats["keep_no_heading_flag"] += 1
            elif decision == "KEEP_FOR_GPT":
                stats["keep_for_gpt"] += 1

    all_csv = output_dir / "content_table_with_length_filter.csv"
    kept_csv = output_dir / "content_table_kept_for_gpt.csv"
    rejected_csv = output_dir / "content_table_auto_rejected_short.csv"
    report_json = output_dir / "length_filter_report.json"

    write_csv(all_csv, all_rows, output_fields)
    write_csv(kept_csv, kept_rows, output_fields)
    write_csv(rejected_csv, rejected_rows, output_fields)

    report_json.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("DONE")
    print(f"input: {input_csv}")
    print(f"total: {stats['total']}")
    print(f"kept for GPT: {stats['kept']}")
    print(f"auto rejected short: {stats['auto_rejected_short']}")
    print(f"short keep for GPT: {stats['short_keep_for_gpt']}")
    print(f"unknown length kept: {stats['keep_unknown_length']}")
    print(f"all csv: {all_csv}")
    print(f"kept csv: {kept_csv}")
    print(f"rejected csv: {rejected_csv}")
    print(f"report: {report_json}")


if __name__ == "__main__":
    main()