"""
Select and validate Tam Anh disease URLs for overnight hybrid ingestion.

Purpose
-------
The overnight ingestion should not waste tokens on false-positive URLs such as:
- viem-da-day        -> viêm dạ dày, not viêm da
- cham-phat-trien   -> chậm phát triển, not chàm
- nhip-tim-cham     -> nhịp tim chậm, not chàm

This script can either:
1. validate an existing selected_urls.json, or
2. discover URLs from Tam Anh /benh pages and build a cleaner candidate set.

It does NOT call OpenAI, Neo4j, or Qdrant.
It is safe to run before spending extraction tokens.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

BASE = "https://tamanhhospital.vn"
BENH_INDEX_URLS = [
    "https://tamanhhospital.vn/benh/",
    *[f"https://tamanhhospital.vn/benh/page/{i}/" for i in range(2, 45)],
]

# Hard false positives caused by Vietnamese accent loss in slugs.
# Keep this list conservative and explicit.
BAD_SLUG_PATTERNS = [
    # viêm dạ / đại / dây / đa ... not viêm da
    r"^viem-da-day($|-)",
    r"^viem-da-day-ruot$",
    r"^viem-da-day-ta-trang$",
    r"^viem-da-day-than-kinh$",
    r"^viem-dai-trang($|-)",
    r"^viem-dau-ti-num-vu$",
    r"^viem-day-than-kinh($|-)",
    r"^viem-da-khop$",      # viêm đa khớp
    r"^viem-da-xoang$",     # viêm đa xoang
    # chậm/chẩm, not chàm
    r"^cham-kinh$",
    r"^cham-phat-trien($|-)",
    r"^tre-cham($|-)",
    r"^nhip-cham-xoang$",
    r"^nhip-tim-cham($|-)",
    r"^dau-day-than-kinh-cham$",
]

# Coherent cluster for the initial 40-60 article sample:
# dermatology + allergy/immunology + respiratory allergy + closely related skin conditions.
CORE_INCLUDE_PATTERNS = [
    r"^viem-da-di-ung$",
    r"^viem-da-co-dia$",
    r"^viem-da-co-la-gi$",
    r"^viem-da-tiep-xuc$",
    r"^viem-da-tiet-ba$",
    r"^viem-da-.*",            # after BAD filters, this mostly means skin dermatitis
    r"^benh-da-lieu$",
    r"^benh-cham$",
    r"^cham-dong-tien$",
    r"^cham-.*",               # after BAD filters, mostly eczema variants
    r"^benh-vay-nen$",
    r"^vay-nen.*",
    r"^viem-khop-vay-nen$",
    r"^mun($|-).*",
    r"^benh-nam-da$",
    r"^nam-da($|-).*",
    r"^di-ung($|-).*",
    r"^viem-mui-di-ung($|-).*",
    r"^viem-xoang-mui-di-ung$",
    r"^hen-suyen$",
    r"^hen-phe-quan-di-ung$",
    r"^benh-phoi-tac-nghen-man-tinh-copd$",
]

# Priority groups make selected URLs more connected for graph retrieval demos.
PRIORITY_PATTERNS = [
    r"di-ung",
    r"viem-da",
    r"cham",
    r"vay-nen",
    r"mun",
    r"nam-da",
    r"hen",
    r"viem-mui",
]


@dataclass
class UrlDecision:
    url: str
    slug: str
    keep: bool
    score: int
    reasons: list[str]


def normalize_url(url: str) -> str:
    url = url.strip()
    if not url:
        return ""
    if url.startswith("/"):
        url = urljoin(BASE, url)
    parsed = urlparse(url)
    if parsed.netloc and "tamanhhospital.vn" not in parsed.netloc:
        return ""
    # Normalize to no trailing slash for stable comparison.
    return f"{parsed.scheme or 'https'}://{parsed.netloc or 'tamanhhospital.vn'}{parsed.path}".rstrip("/")


def slug_from_url(url: str) -> str:
    path = urlparse(url).path.strip("/")
    parts = path.split("/")
    if len(parts) >= 2 and parts[-2] == "benh":
        return parts[-1]
    if parts and parts[0] == "benh":
        return parts[-1]
    return parts[-1] if parts else ""


def matches_any(slug: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, slug) for pattern in patterns)


def decide_url(url: str) -> UrlDecision:
    url = normalize_url(url)
    slug = slug_from_url(url)
    reasons: list[str] = []
    score = 0

    if not url or not slug:
        return UrlDecision(url=url, slug=slug, keep=False, score=-999, reasons=["invalid_url"])

    if "/benh/" not in urlparse(url).path:
        return UrlDecision(url=url, slug=slug, keep=False, score=-999, reasons=["not_benh_article"])

    if matches_any(slug, BAD_SLUG_PATTERNS):
        return UrlDecision(url=url, slug=slug, keep=False, score=-100, reasons=["explicit_false_positive_slug"])

    if matches_any(slug, CORE_INCLUDE_PATTERNS):
        score += 50
        reasons.append("core_cluster_match")
    else:
        # Not necessarily wrong, but not in the initial coherent sample.
        score -= 10
        reasons.append("outside_core_cluster")

    for idx, pattern in enumerate(PRIORITY_PATTERNS):
        if re.search(pattern, slug):
            score += max(1, 12 - idx)
            reasons.append(f"priority:{pattern}")

    # Prefer disease article slugs that are not overly generic.
    if len(slug.split("-")) >= 2:
        score += 2

    keep = score >= 40
    return UrlDecision(url=url, slug=slug, keep=keep, score=score, reasons=reasons)


def fetch_html(url: str, timeout: int = 20) -> str:
    req = Request(
        url,
        headers={
            "User-Agent": "medical-vietnamese-hybrid-rag/0.1 (academic thesis URL selector)",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def discover_urls(max_pages: int, sleep_seconds: float = 0.25) -> list[str]:
    urls: set[str] = set()
    pages = BENH_INDEX_URLS[:max_pages]

    for i, page_url in enumerate(pages, start=1):
        print(f"[DISCOVER] {i}/{len(pages)} {page_url}", flush=True)
        try:
            html = fetch_html(page_url)
        except Exception as exc:
            print(f"  [WARN] failed: {type(exc).__name__}: {exc}", flush=True)
            continue

        # Match hrefs under /benh/<slug>. Avoid query/hash.
        for href in re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE):
            full = normalize_url(href)
            if not full:
                continue
            path = urlparse(full).path.strip("/")
            if re.fullmatch(r"benh/[a-z0-9-]+", path):
                urls.add(full)

        time.sleep(sleep_seconds)

    return sorted(urls)


def load_source_urls(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict):
        if "selected_urls" in data:
            return [str(x) for x in data["selected_urls"]]
        if "urls" in data:
            return [str(x) for x in data["urls"]]
    raise ValueError(f"Unsupported URL file format: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-file", type=Path, default=None, help="Validate an existing selected_urls.json instead of discovering.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/tmp/url_selection_v45"))
    parser.add_argument("--target", type=int, default=60)
    parser.add_argument("--max-pages", type=int, default=30)
    parser.add_argument("--discover", action="store_true", help="Discover URLs from Tam Anh index pages.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.source_file:
        print(f"[LOAD] source_file={args.source_file}", flush=True)
        raw_urls = load_source_urls(args.source_file)
    else:
        print(f"[DISCOVER] max_pages={args.max_pages}", flush=True)
        raw_urls = discover_urls(max_pages=args.max_pages)

    decisions = [decide_url(url) for url in raw_urls]
    kept = [d for d in decisions if d.keep]
    kept = sorted(kept, key=lambda d: (-d.score, d.slug))
    selected = kept[: args.target]
    excluded = [d for d in decisions if not d.keep]

    selected_urls = [d.url for d in selected]

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_file": str(args.source_file) if args.source_file else None,
        "raw_count": len(raw_urls),
        "kept_count": len(kept),
        "selected_count": len(selected_urls),
        "target": args.target,
        "selected_urls": selected_urls,
        "selected": [asdict(d) for d in selected],
        "excluded_false_positive_examples": [asdict(d) for d in excluded if "explicit_false_positive_slug" in d.reasons][:80],
        "excluded_other_examples": [asdict(d) for d in excluded if "explicit_false_positive_slug" not in d.reasons][:80],
    }

    (args.output_dir / "selected_urls_strict.json").write_text(
        json.dumps({
            "created_at": report["created_at"],
            "target_articles": args.target,
            "raw_count": len(raw_urls),
            "selected_count": len(selected_urls),
            "selected_urls": selected_urls,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "selected_urls_strict.txt").write_text("\n".join(selected_urls), encoding="utf-8")
    (args.output_dir / "url_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n========== URL SELECTION VALIDATION ==========")
    print(json.dumps({
        "raw_count": len(raw_urls),
        "kept_count": len(kept),
        "selected_count": len(selected_urls),
        "target": args.target,
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False, indent=2))

    print("\n[SELECTED SAMPLE]")
    for i, d in enumerate(selected[:20], start=1):
        print(f"{i:02d}. score={d.score:>3} | {d.slug} | {d.url}")

    false_pos = [d for d in excluded if "explicit_false_positive_slug" in d.reasons]
    if false_pos:
        print("\n[EXCLUDED FALSE POSITIVE SAMPLE]")
        for i, d in enumerate(false_pos[:30], start=1):
            print(f"{i:02d}. {d.slug} | {d.url}")

    if len(selected_urls) < args.target:
        print("\n[WARN] selected_count < target. Increase --max-pages or relax filters.", file=sys.stderr)


if __name__ == "__main__":
    main()
