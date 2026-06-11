"""
Select exactly 60 coherent Tam Anh disease URLs for overnight ingestion.

Compared with the strict selector, this version is tiered:
- Tier 1: tight dermatology/allergy/respiratory-allergy cluster.
- Tier 2: nearby respiratory/ENT/infectious/inflammatory/metabolic comorbidities.
- Hard false positives are still blocked, e.g. viem-da-day (viêm dạ dày),
  tre-cham-* (trẻ chậm...), nhip-tim-cham (nhịp tim chậm).

It does NOT call OpenAI, Neo4j, or Qdrant.
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
    "https://tamanhhospital.vn/benh-hoc-a-z/",
    *[f"https://tamanhhospital.vn/benh/page/{i}/" for i in range(2, 80)],
]

BAD_SLUG_PATTERNS = [
    # viêm dạ / đại / dây / đa ... not viêm da
    r"^viem-da-day($|-)",
    r"^viem-da-day-ruot$",
    r"^viem-da-day-ta-trang$",
    r"^viem-da-day-than-kinh$",
    r"^viem-dai-trang($|-)",
    r"^viem-dau-ti-num-vu$",
    r"^viem-day-than-kinh($|-)",
    r"^viem-da-khop$",
    r"^viem-da-xoang$",
    # chậm/chẩm, not chàm
    r"^cham-kinh$",
    r"^cham-phat-trien($|-)",
    r"^tre-cham($|-)",
    r"^nhip-cham-xoang$",
    r"^nhip-tim-cham($|-)",
    r"^dau-day-than-kinh-cham$",
]

# Highest priority: coherent for dermatology/allergy graph demo.
TIER1_PATTERNS = [
    r"^viem-da-di-ung$",
    r"^viem-da-co-dia$",
    r"^viem-da-co-la-gi$",
    r"^viem-da-tiep-xuc$",
    r"^viem-da-tiet-ba$",
    r"^viem-da-.*",  # after bad filters, mostly dermatitis/skin inflammation
    r"^benh-da-lieu$",
    r"^benh-cham$",
    r"^cham-dong-tien$",
    r"^cham-.*",
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

# Fillers to reach 60 while staying clinically connected.
TIER2_PATTERNS = [
    # respiratory / ENT inflammation, often related to allergy/immune context
    r"^viem-xoang($|-).*",
    r"^viem-mui($|-).*",
    r"^viem-hong($|-).*",
    r"^viem-amidan($|-).*",
    r"^viem-phe-quan($|-).*",
    r"^viem-phoi($|-).*",
    r"^lao-phoi$",
    r"^copd$",
    r"^benh-phoi($|-).*",
    # infectious/inflammatory skin-adjacent or immune-relevant
    r"^zona($|-).*",
    r"^zona-than-kinh$",
    r"^noi-me-day($|-).*",
    r"^ngua($|-).*",
    r"^phat-ban($|-).*",
    r"^thuy-dau$",
    r"^soi($|-).*",
    # metabolic/comorbidity group useful for broader medical graph
    r"^dai-thao-duong($|-).*",
    r"^tieu-duong($|-).*",
    r"^beo-phi$",
    r"^roi-loan-mo-mau$",
    r"^tang-huyet-ap$",
    r"^gan-nhiem-mo$",
    r"^viem-gan($|-).*",
]

KEYWORD_PRIORITY = [
    "di-ung", "viem-da", "cham", "vay-nen", "mun", "nam-da", "hen", "viem-mui",
    "viem-xoang", "viem-hong", "viem-amidan", "viem-phe-quan", "viem-phoi", "copd",
    "zona", "noi-me-day", "ngua", "phat-ban", "dai-thao-duong", "beo-phi",
    "roi-loan-mo-mau", "tang-huyet-ap", "gan-nhiem-mo", "viem-gan",
]

# Known good seeds make the first part stable even if discovery ordering changes.
CURATED_SEEDS = [
    "https://tamanhhospital.vn/benh/viem-da-di-ung",
    "https://tamanhhospital.vn/benh/hen-phe-quan-di-ung",
    "https://tamanhhospital.vn/benh/hen-suyen",
    "https://tamanhhospital.vn/benh/viem-mui-di-ung",
    "https://tamanhhospital.vn/benh/viem-mui-di-ung-boi-nhiem",
    "https://tamanhhospital.vn/benh/di-ung-hoa-chat",
    "https://tamanhhospital.vn/benh/di-ung-mat",
    "https://tamanhhospital.vn/benh/di-ung-thoi-tiet",
    "https://tamanhhospital.vn/benh/viem-xoang-mui-di-ung",
    "https://tamanhhospital.vn/benh/viem-da-co-dia",
    "https://tamanhhospital.vn/benh/viem-da-co-la-gi",
    "https://tamanhhospital.vn/benh/viem-da-tiep-xuc",
    "https://tamanhhospital.vn/benh/viem-da-tiet-ba",
    "https://tamanhhospital.vn/benh/benh-cham",
    "https://tamanhhospital.vn/benh/cham-dong-tien",
    "https://tamanhhospital.vn/benh/benh-vay-nen",
    "https://tamanhhospital.vn/benh/vay-nen-dao-nguoc",
    "https://tamanhhospital.vn/benh/vay-nen-the-giot",
    "https://tamanhhospital.vn/benh/vay-nen-the-mang",
    "https://tamanhhospital.vn/benh/viem-khop-vay-nen",
    "https://tamanhhospital.vn/benh/mun",
    "https://tamanhhospital.vn/benh/mun-an",
    "https://tamanhhospital.vn/benh/mun-boc",
    "https://tamanhhospital.vn/benh/mun-cam",
    "https://tamanhhospital.vn/benh/mun-coc",
    "https://tamanhhospital.vn/benh/mun-dau-den",
    "https://tamanhhospital.vn/benh/mun-nhot",
    "https://tamanhhospital.vn/benh/mun-noi-tiet",
    "https://tamanhhospital.vn/benh/mun-thit",
    "https://tamanhhospital.vn/benh/mun-trung-ca",
    "https://tamanhhospital.vn/benh/benh-nam-da",
    "https://tamanhhospital.vn/benh/nam-da",
    "https://tamanhhospital.vn/benh/nam-da-biu",
    "https://tamanhhospital.vn/benh/benh-phoi-tac-nghen-man-tinh-copd",
    "https://tamanhhospital.vn/benh/benh-da-lieu",
]


@dataclass
class UrlDecision:
    url: str
    slug: str
    keep: bool
    tier: int
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
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc or "tamanhhospital.vn"
    return f"{scheme}://{netloc}{parsed.path}".rstrip("/")


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


def unique_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        url = normalize_url(item)
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(url)
    return out


def decide_url(url: str) -> UrlDecision:
    url = normalize_url(url)
    slug = slug_from_url(url)
    reasons: list[str] = []

    if not url or not slug:
        return UrlDecision(url, slug, False, 99, -999, ["invalid_url"])
    if "/benh/" not in urlparse(url).path:
        return UrlDecision(url, slug, False, 99, -999, ["not_benh_article"])
    if matches_any(slug, BAD_SLUG_PATTERNS):
        return UrlDecision(url, slug, False, 99, -100, ["explicit_false_positive_slug"])

    score = 0
    tier = 99
    if matches_any(slug, TIER1_PATTERNS):
        tier = 1
        score += 100
        reasons.append("tier1_core")
    elif matches_any(slug, TIER2_PATTERNS):
        tier = 2
        score += 55
        reasons.append("tier2_related")
    else:
        return UrlDecision(url, slug, False, 99, -10, ["outside_tier1_tier2"])

    for idx, kw in enumerate(KEYWORD_PRIORITY):
        if kw in slug:
            score += max(1, 30 - idx)
            reasons.append(f"kw:{kw}")

    # Prefer specific disease pages, not overly generic index-like pages.
    if len(slug.split("-")) >= 2:
        score += 2
    if slug in {"mun", "nam-da", "benh-da-lieu", "benh-phoi"}:
        score -= 5
        reasons.append("generic_but_allowed")

    return UrlDecision(url, slug, True, tier, score, reasons)


def fetch_html(url: str, timeout: int = 20) -> str:
    req = Request(url, headers={"User-Agent": "medical-vietnamese-hybrid-rag/0.1 URL selector"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def discover_urls(max_pages: int, sleep_seconds: float = 0.2) -> list[str]:
    urls: set[str] = set(CURATED_SEEDS)
    pages = BENH_INDEX_URLS[:max_pages]
    for i, page_url in enumerate(pages, start=1):
        print(f"[DISCOVER] {i}/{len(pages)} {page_url}", flush=True)
        try:
            html = fetch_html(page_url)
        except Exception as exc:
            print(f"  [WARN] failed: {type(exc).__name__}: {exc}", flush=True)
            continue
        for href in re.findall(r"href=[\"']([^\"']+)[\"']", html, flags=re.IGNORECASE):
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
    parser.add_argument("--source-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/tmp/url_selection_v45_expand60"))
    parser.add_argument("--target", type=int, default=60)
    parser.add_argument("--max-pages", type=int, default=70)
    parser.add_argument("--discover", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.source_file:
        raw_urls = unique_keep_order(load_source_urls(args.source_file) + CURATED_SEEDS)
    else:
        raw_urls = discover_urls(max_pages=args.max_pages)

    decisions = [decide_url(url) for url in raw_urls]
    kept = [d for d in decisions if d.keep]
    kept = sorted(kept, key=lambda d: (d.tier, -d.score, d.slug))
    selected = kept[: args.target]
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
        "kept_unselected": [asdict(d) for d in kept[args.target:]],
        "excluded_false_positive_examples": [asdict(d) for d in decisions if "explicit_false_positive_slug" in d.reasons][:100],
        "excluded_other_examples": [asdict(d) for d in decisions if not d.keep and "explicit_false_positive_slug" not in d.reasons][:100],
    }

    (args.output_dir / "selected_urls_60.json").write_text(
        json.dumps({
            "created_at": report["created_at"],
            "target_articles": args.target,
            "raw_count": len(raw_urls),
            "kept_count": len(kept),
            "selected_count": len(selected_urls),
            "selected_urls": selected_urls,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "selected_urls_60.txt").write_text("\n".join(selected_urls), encoding="utf-8")
    (args.output_dir / "url_selection_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n========== URL SELECTION EXPAND60 ==========")
    print(json.dumps({
        "raw_count": len(raw_urls),
        "kept_count": len(kept),
        "selected_count": len(selected_urls),
        "target": args.target,
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False, indent=2))

    print("\n[SELECTED SAMPLE]")
    for i, d in enumerate(selected[:80], start=1):
        print(f"{i:02d}. tier={d.tier} score={d.score:>3} | {d.slug} | {d.url}")

    false_pos = [d for d in decisions if "explicit_false_positive_slug" in d.reasons]
    if false_pos:
        print("\n[EXCLUDED FALSE POSITIVE SAMPLE]")
        for i, d in enumerate(false_pos[:30], start=1):
            print(f"{i:02d}. {d.slug} | {d.url}")

    if len(selected_urls) < args.target:
        print("\n[WARN] selected_count < target. Increase --max-pages or add more Tier2 patterns.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
