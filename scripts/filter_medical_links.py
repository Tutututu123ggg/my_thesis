from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def domain_key(source: str) -> str:
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter discovered medical links into approved crawl seed + coverage report.")
    parser.add_argument("--input", default="data/crawl/candidate_links.jsonl")
    parser.add_argument("--output-seed", default="data/crawl/approved_seed_urls.json")
    parser.add_argument("--coverage", default="data/crawl/coverage_report.csv")
    parser.add_argument("--min-score", type=int, default=60)
    parser.add_argument("--max-per-anchor-source", type=int, default=3)
    parser.add_argument("--min-domains-per-anchor", type=int, default=2)
    args = parser.parse_args()

    rows = [r for r in read_jsonl(Path(args.input)) if int(r.get("score", 0)) >= args.min_score]
    rows.sort(key=lambda r: int(r.get("score", 0)), reverse=True)

    # De-duplicate same URL globally, keeping the highest-scored anchor assignment.
    used_urls: set[str] = set()
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        url = r.get("url", "")
        if not url or url in used_urls:
            continue
        key = (r.get("anchor_entity", ""), r.get("source", ""))
        if len(grouped[key]) >= args.max_per_anchor_source:
            continue
        grouped[key].append(r)
        used_urls.add(url)

    by_anchor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for items in grouped.values():
        for item in items:
            by_anchor[item.get("anchor_entity", "")].append(item)

    seed_clusters: dict[str, dict[str, Any]] = {}
    coverage_rows: list[dict[str, Any]] = []
    for anchor, items in sorted(by_anchor.items()):
        domains = sorted({domain_key(i.get("source", "")) for i in items if i.get("source")})
        status = "approved" if len(domains) >= args.min_domains_per_anchor else "needs_more_sources"
        coverage_rows.append({
            "anchor_entity": anchor,
            "cluster": items[0].get("cluster", "") if items else "",
            "organ_system": items[0].get("organ_system", "") if items else "",
            "domain_count": len(domains),
            "sources": ";".join(domains),
            "article_count": len(items),
            "status": status,
        })
        if status != "approved":
            continue
        cluster_name = items[0].get("cluster", "uncategorized") or "uncategorized"
        cluster = seed_clusters.setdefault(cluster_name, {
            "cluster": cluster_name,
            "organ_system": items[0].get("organ_system", ""),
            "anchors": [],
        })
        cluster["anchors"].append({
            "anchor_entity": anchor,
            "domain_count": len(domains),
            "sources": domains,
            "urls": [
                {
                    "source": item.get("source", ""),
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "score": item.get("score", 0),
                    "purpose_guess": item.get("purpose_guess", []),
                    "status": "approved",
                }
                for item in sorted(items, key=lambda x: (x.get("source", ""), -int(x.get("score", 0))))
            ],
        })

    output_seed = Path(args.output_seed)
    output_seed.parent.mkdir(parents=True, exist_ok=True)
    seed_data = {"clusters": list(seed_clusters.values())}
    output_seed.write_text(json.dumps(seed_data, ensure_ascii=False, indent=2), encoding="utf-8")

    cov_path = Path(args.coverage)
    cov_path.parent.mkdir(parents=True, exist_ok=True)
    with cov_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["anchor_entity", "cluster", "organ_system", "domain_count", "sources", "article_count", "status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(coverage_rows)

    print(f"[DONE] approved clusters={len(seed_clusters)} anchors={sum(len(c['anchors']) for c in seed_clusters.values())}")
    print(f"[SEED]     {output_seed}")
    print(f"[COVERAGE] {cov_path}")


if __name__ == "__main__":
    main()
