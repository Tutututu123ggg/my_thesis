from __future__ import annotations

import argparse
import csv
import html
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlencode, urlparse, urlunparse

try:
    import requests
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'requests'. Install with: pip install requests") from exc


@dataclass(frozen=True)
class SourceConfig:
    name: str
    domains: tuple[str, ...]
    allowed_path_contains: tuple[str, ...]
    search_site: str
    bad_url_terms: tuple[str, ...]


SOURCE_CONFIGS: dict[str, SourceConfig] = {
    "tamanh": SourceConfig(
        name="tamanh",
        domains=("tamanhhospital.vn",),
        allowed_path_contains=("/",),
        search_site="tamanhhospital.vn",
        bad_url_terms=("/tag/", "/author/", "dat-lich", "bang-gia", "uu-dai", "tuyen-dung"),
    ),
    "vinmec": SourceConfig(
        name="vinmec",
        domains=("vinmec.com",),
        allowed_path_contains=("/vie/bai-viet/", "/vie/benh/"),
        search_site="vinmec.com/vie",
        bad_url_terms=(
            "/en/", "/eng/", "chinh-sach", "bao-ve-du-lieu", "quyen-rieng-tu",
            "thong-bao", "lich-su", "thanh-tich", "vinsmart", "v-app", "vingroup",
            "khuyen-mai", "uu-dai", "tuyen-dung", "dich-vu-y-te", "he-thong-y-te",
        ),
    ),
    "medlatec": SourceConfig(
        name="medlatec",
        domains=("medlatec.vn", "medlatec.com"),
        allowed_path_contains=("/tin-tuc/", "/benh/", "/suc-khoe/"),
        search_site="medlatec.vn",
        bad_url_terms=("tuyen-dung", "khuyen-mai", "uu-dai", "bang-gia", "dat-lich", "goi-kham"),
    ),
    "msd_manual_vi": SourceConfig(
        name="msd_manual_vi",
        domains=("msdmanuals.com",),
        allowed_path_contains=("/vi/",),
        search_site="msdmanuals.com/vi",
        bad_url_terms=("/en/", "/professional/news", "resourcespages", "about"),
    ),
}

PURPOSE_TERMS = {
    "overview": ("tổng quan", "là gì", "bệnh gì", "overview"),
    "symptoms": ("triệu chứng", "dấu hiệu", "biểu hiện", "symptoms"),
    "causes": ("nguyên nhân", "yếu tố nguy cơ", "causes", "risk"),
    "diagnosis": ("chẩn đoán", "xét nghiệm", "diagnosis", "test"),
    "treatment": ("điều trị", "thuốc", "phác đồ", "treatment"),
    "prevention": ("phòng ngừa", "dự phòng", "prevention"),
    "differential": ("phân biệt", "khác nhau", "nhầm lẫn", "differential"),
    "complications": ("biến chứng", "complication"),
}

GLOBAL_BAD_TERMS = (
    "chính sách", "bảo vệ dữ liệu", "quyền riêng tư", "thông báo", "lịch sử hình thành",
    "thành tích", "vinsmart", "v-app", "vingroup", "ưu đãi", "khuyến mãi", "tuyển dụng",
    "bảng giá", "đặt lịch", "dịch vụ", "hệ thống y tế", "app", "myvinmec",
)

USER_AGENT = "medical-vietnamese-hybrid-rag-link-discovery/0.1 (academic thesis; contact: ductu3003@gmail.com)"


def norm_text(text: str | None) -> str:
    if not text:
        return ""
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_url(url: str) -> str:
    url = html.unescape(url).strip()
    if url.startswith("/url?") or "bing.com/ck/a" in url:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        for key in ("q", "url", "u"):
            if key in qs and qs[key]:
                url = qs[key][0]
                break
    url = unquote(url)
    parsed = urlparse(url)
    if not parsed.scheme.startswith("http"):
        return ""
    query_items = []
    for k, vals in parse_qs(parsed.query, keep_blank_values=True).items():
        if k.lower().startswith("utm_") or k.lower() in {"fbclid", "gclid"}:
            continue
        for v in vals:
            query_items.append((k, v))
    query = urlencode(query_items, doseq=True)
    return urlunparse((parsed.scheme, parsed.netloc.lower(), parsed.path.rstrip("/"), "", query, ""))


def source_matches(url: str, cfg: SourceConfig) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    return any(host == d or host.endswith("." + d) for d in cfg.domains)


def path_allowed(url: str, cfg: SourceConfig) -> bool:
    path = urlparse(url).path.lower()
    return any(part.lower() in path for part in cfg.allowed_path_contains)


def is_bad_url(url: str, cfg: SourceConfig) -> bool:
    low = url.lower()
    return any(term in low for term in cfg.bad_url_terms)


def bing_search(query: str, limit: int, sleep: float) -> list[str]:
    params = {"q": query, "count": str(max(10, limit)), "setlang": "vi-VN"}
    url = "https://www.bing.com/search?" + urlencode(params)
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "vi,en;q=0.8"}
    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
    except Exception as exc:
        print(f"[WARN] Bing search failed for query={query!r}: {type(exc).__name__}: {exc}")
        return []
    time.sleep(sleep)
    html_text = response.text
    urls: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r'<a[^>]+href=["\']([^"\']+)["\']', html_text, flags=re.I):
        candidate = clean_url(match.group(1))
        if not candidate:
            continue
        host = urlparse(candidate).netloc.lower()
        if any(skip in host for skip in ("bing.com", "microsoft.com", "go.microsoft.com")):
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        urls.append(candidate)
        if len(urls) >= limit:
            break
    return urls


def fetch_metadata(url: str, sleep: float) -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "vi,en;q=0.8"}
    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
    except Exception as exc:
        return {"fetch_error": f"{type(exc).__name__}: {exc}", "title": "", "description": ""}
    time.sleep(sleep)
    text = response.text[:500_000]
    title = ""
    desc = ""
    og_title = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)', text, flags=re.I)
    if og_title:
        title = html.unescape(og_title.group(1)).strip()
    if not title:
        m = re.search(r"<title[^>]*>(.*?)</title>", text, flags=re.I | re.S)
        if m:
            title = html.unescape(re.sub(r"\s+", " ", m.group(1))).strip()
    mdesc = re.search(r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)', text, flags=re.I)
    if not mdesc:
        mdesc = re.search(r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)', text, flags=re.I)
    if mdesc:
        desc = html.unescape(mdesc.group(1)).strip()
    return {"title": title, "description": desc, "fetch_error": ""}


def purpose_guess(text: str) -> list[str]:
    low = norm_text(text)
    found: list[str] = []
    for purpose, terms in PURPOSE_TERMS.items():
        for term in terms:
            if norm_text(term) in low:
                found.append(purpose)
                break
    return found


def score_candidate(url: str, source: str, anchor: str, aliases: list[str], title: str, desc: str) -> tuple[int, list[str], list[str]]:
    cfg = SOURCE_CONFIGS[source]
    score = 0
    reasons: list[str] = []
    text = f"{title} {desc} {url}"
    text_norm = norm_text(text)
    title_norm = norm_text(title)
    desc_norm = norm_text(desc)
    path_norm = norm_text(urlparse(url).path)

    if source_matches(url, cfg):
        score += 20
        reasons.append("domain_match")
    else:
        score -= 100
        reasons.append("wrong_domain")

    if path_allowed(url, cfg):
        score += 15
        reasons.append("allowed_path")
    else:
        score -= 20
        reasons.append("weak_path")

    if is_bad_url(url, cfg):
        score -= 80
        reasons.append("bad_url_term")

    anchor_terms = [anchor, *aliases]
    matched_anchor = False
    for term in anchor_terms:
        nterm = norm_text(term)
        if not nterm:
            continue
        if nterm in title_norm:
            score += 45
            reasons.append(f"anchor_in_title:{term}")
            matched_anchor = True
            break
        if nterm in desc_norm:
            score += 30
            reasons.append(f"anchor_in_description:{term}")
            matched_anchor = True
            break
        if nterm in path_norm:
            score += 18
            reasons.append(f"anchor_in_path:{term}")
            matched_anchor = True
            break

    if not matched_anchor:
        # Fuzzy token overlap for titles that use shorter wording.
        anchor_tokens = set(norm_text(anchor).split())
        title_tokens = set(title_norm.split())
        overlap = len(anchor_tokens & title_tokens)
        if anchor_tokens and overlap / max(1, len(anchor_tokens)) >= 0.5:
            score += 18
            reasons.append("partial_anchor_title_overlap")
        else:
            score -= 10
            reasons.append("no_anchor_match")

    purposes = purpose_guess(text)
    if purposes:
        score += min(20, 6 * len(purposes))
        reasons.append("purpose_terms:" + ",".join(purposes[:4]))

    bad_text = [term for term in GLOBAL_BAD_TERMS if norm_text(term) in text_norm]
    if bad_text:
        score -= 60
        reasons.append("bad_text_terms:" + ",".join(bad_text[:4]))

    return score, reasons, purposes


def iter_topic_anchors(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for cluster in data.get("clusters", []):
        for item in cluster.get("anchors", []):
            records.append({
                "cluster": cluster.get("cluster", ""),
                "organ_system": cluster.get("organ_system", ""),
                "anchor_entity": item.get("anchor_entity", ""),
                "aliases": item.get("aliases", []),
            })
    return records


def build_queries(source: str, anchor: str, aliases: list[str]) -> list[str]:
    cfg = SOURCE_CONFIGS[source]
    terms = [anchor, *aliases[:2]]
    quoted = " OR ".join(f'"{t}"' for t in terms if t)
    base = f"site:{cfg.search_site} ({quoted})"
    # One broad query + a few intent queries. Bing may ignore parentheses, scoring handles filtering.
    return [
        base,
        f'{base} "triệu chứng" OR "chẩn đoán"',
        f'{base} "điều trị" OR "nguyên nhân"',
    ]


def write_outputs(records: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    records = sorted(records, key=lambda r: (r["score"], r.get("title", "")), reverse=True)
    with output.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    csv_path = output.with_suffix(".csv")
    fieldnames = [
        "score", "source", "cluster", "organ_system", "anchor_entity", "url", "title",
        "description", "purpose_guess", "reasons", "query", "fetch_error",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = dict(r)
            row["purpose_guess"] = ";".join(row.get("purpose_guess", []))
            row["reasons"] = ";".join(row.get("reasons", []))
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"[DONE] wrote {len(records)} candidates")
    print(f"[JSONL] {output}")
    print(f"[CSV]   {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover and score candidate medical article links by source/topic.")
    parser.add_argument("--topics-file", default="data/crawl/topic_seed_dense_v1.json")
    parser.add_argument("--sources", nargs="+", default=["tamanh", "vinmec", "medlatec", "msd_manual_vi"])
    parser.add_argument("--output", default="data/crawl/candidate_links.jsonl")
    parser.add_argument("--max-results-per-query", type=int, default=8)
    parser.add_argument("--max-fetch-per-anchor-source", type=int, default=12)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--no-fetch", action="store_true", help="Do not fetch pages for title/description; score URL only.")
    parser.add_argument("--queries-only", action="store_true", help="Only write search queries to a .queries.txt file.")
    args = parser.parse_args()

    topic_path = Path(args.topics_file)
    anchors = iter_topic_anchors(topic_path)
    out_path = Path(args.output)
    records: list[dict[str, Any]] = []
    queries_out: list[str] = []
    seen_keys: set[tuple[str, str, str]] = set()

    for source in args.sources:
        if source not in SOURCE_CONFIGS:
            raise SystemExit(f"Unsupported source={source}. Supported: {sorted(SOURCE_CONFIGS)}")
        for anchor_rec in anchors:
            anchor = anchor_rec["anchor_entity"]
            aliases = list(anchor_rec.get("aliases", []))
            queries = build_queries(source, anchor, aliases)
            source_anchor_candidates: list[tuple[str, str]] = []
            for query in queries:
                queries_out.append(query)
                if args.queries_only:
                    continue
                print(f"[SEARCH] source={source} anchor={anchor} query={query}")
                urls = bing_search(query, limit=args.max_results_per_query, sleep=args.sleep)
                for url in urls:
                    cfg = SOURCE_CONFIGS[source]
                    if not source_matches(url, cfg):
                        continue
                    if is_bad_url(url, cfg):
                        continue
                    source_anchor_candidates.append((url, query))

            if args.queries_only:
                continue

            # Deduplicate and limit metadata fetch per anchor/source.
            deduped: list[tuple[str, str]] = []
            seen_url: set[str] = set()
            for url, query in source_anchor_candidates:
                if url in seen_url:
                    continue
                seen_url.add(url)
                deduped.append((url, query))
                if len(deduped) >= args.max_fetch_per_anchor_source:
                    break

            for url, query in deduped:
                key = (source, anchor, url)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                meta = {"title": "", "description": "", "fetch_error": ""}
                if not args.no_fetch:
                    meta = fetch_metadata(url, sleep=args.sleep)
                score, reasons, purposes = score_candidate(
                    url=url,
                    source=source,
                    anchor=anchor,
                    aliases=aliases,
                    title=meta.get("title", ""),
                    desc=meta.get("description", ""),
                )
                records.append({
                    **anchor_rec,
                    "source": source,
                    "url": url,
                    "title": meta.get("title", ""),
                    "description": meta.get("description", ""),
                    "fetch_error": meta.get("fetch_error", ""),
                    "score": score,
                    "reasons": reasons,
                    "purpose_guess": purposes,
                    "query": query,
                })

    if args.queries_only:
        qpath = out_path.with_suffix(".queries.txt")
        qpath.parent.mkdir(parents=True, exist_ok=True)
        qpath.write_text("\n".join(queries_out), encoding="utf-8")
        print(f"[DONE] wrote queries: {qpath}")
        return

    write_outputs(records, out_path)


if __name__ == "__main__":
    main()
