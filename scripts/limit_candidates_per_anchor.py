from __future__ import annotations

import argparse
import csv
from collections import defaultdict, deque
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def first_existing(row: dict, keys: list[str]) -> str:
    for k in keys:
        v = str(row.get(k, "") or "").strip()
        if v:
            return v
    return ""


def canonical_url(url: str) -> str:
    url = str(url or "").strip()
    if not url:
        return ""
    p = urlsplit(url)
    path = p.path.rstrip("/")
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), path, "", ""))


def get_anchor(row: dict) -> str:
    return first_existing(row, ["anchor_entity", "anchor", "query", "keyword", "topic"]) or "__unknown__"


def get_source(row: dict) -> str:
    return first_existing(row, ["source", "domain"]) or "__unknown__"


def get_score(row: dict) -> int:
    try:
        return int(float(row.get("score", 0) or 0))
    except Exception:
        return 0


def source_balanced_top(rows: list[dict], max_items: int) -> list[dict]:
    """
    Chọn top N trong một anchor, ưu tiên score cao nhưng vẫn cân bằng source.
    Tránh trường hợp top 20 toàn Vinmec/Tâm Anh.
    """
    by_source: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        by_source[get_source(row)].append(row)

    for src in by_source:
        by_source[src].sort(key=lambda r: (-get_score(r), str(r.get("title", "")), str(r.get("url", ""))))

    queues = {
        src: deque(items)
        for src, items in sorted(by_source.items())
    }

    picked = []
    seen_url = set()

    while len(picked) < max_items and any(queues.values()):
        progressed = False

        for src in list(queues.keys()):
            q = queues[src]
            while q:
                row = q.popleft()
                url = canonical_url(first_existing(row, ["url", "article_url", "link", "href"]))
                if not url or url in seen_url:
                    continue

                seen_url.add(url)
                picked.append(row)
                progressed = True
                break

            if len(picked) >= max_items:
                break

        if not progressed:
            break

    return picked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--max-per-anchor", type=int, default=20)
    args = parser.parse_args()

    in_path = Path(args.input_csv)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    by_anchor: dict[str, list[dict]] = defaultdict(list)

    # Global URL dedupe trước, nhưng vẫn giữ bản score cao hơn nếu URL lặp.
    best_by_url: dict[str, dict] = {}
    for row in rows:
        url = canonical_url(first_existing(row, ["url", "article_url", "link", "href"]))
        if not url:
            continue

        prev = best_by_url.get(url)
        if prev is None or get_score(row) > get_score(prev):
            best_by_url[url] = row

    for row in best_by_url.values():
        by_anchor[get_anchor(row)].append(row)

    kept = []
    report_rows = []

    for anchor, items in sorted(by_anchor.items()):
        selected = source_balanced_top(items, args.max_per_anchor)
        kept.extend(selected)

        report_rows.append({
            "anchor_entity": anchor,
            "input_count": len(items),
            "kept_count": len(selected),
            "sources": ";".join(sorted({get_source(r) for r in selected})),
        })

    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    report_path = out_path.with_name(out_path.stem + "_limit_report.csv")
    with report_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["anchor_entity", "input_count", "kept_count", "sources"])
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"input rows: {len(rows)}")
    print(f"unique urls before anchor limit: {len(best_by_url)}")
    print(f"anchors: {len(by_anchor)}")
    print(f"kept rows: {len(kept)}")
    print(f"output: {out_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()