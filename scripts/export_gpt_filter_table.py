from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFC", str(s or ""))
    return re.sub(r"\s+", " ", s).strip()


def norm_key(s: str) -> str:
    s = norm_text(s).lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def canonical_url(url: str) -> str:
    url = norm_text(url)
    if not url:
        return ""
    p = urlsplit(url)
    path = p.path.rstrip("/")
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), path, "", ""))


def first_existing(row: dict, keys: list[str]) -> str:
    for k in keys:
        if k in row and str(row[k]).strip():
            return str(row[k]).strip()
    return ""


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    meta = {}
    end_idx = None
    for i in range(1, min(len(lines), 80)):
        if lines[i].strip() == "---":
            end_idx = i
            break

        m = re.match(r"^([A-Za-z0-9_\-]+)\s*:\s*(.*)$", lines[i])
        if m:
            key = m.group(1).strip().lower()
            val = m.group(2).strip().strip('"').strip("'")
            meta[key] = val

    if end_idx is None:
        return meta, text

    body = "\n".join(lines[end_idx + 1 :])
    return meta, body


def extract_title(meta: dict[str, str], body: str, path: Path) -> str:
    for k in ["title", "article_title", "name"]:
        if meta.get(k):
            return norm_text(meta[k])

    for line in body.splitlines()[:80]:
        line = line.strip()
        if line.startswith("# "):
            return norm_text(line[2:])
        if line.lower().startswith("title:"):
            return norm_text(line.split(":", 1)[1])

    return path.stem


def extract_headings(body: str, limit: int = 20) -> list[str]:
    out = []
    for line in body.splitlines():
        line = line.strip()
        if re.match(r"^#{1,4}\s+", line):
            h = re.sub(r"^#{1,4}\s+", "", line).strip()
            if h and h not in out:
                out.append(h)
        if len(out) >= limit:
            break
    return out


def markdown_to_plain(body: str) -> str:
    body = re.sub(r"```.*?```", " ", body, flags=re.S)
    body = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", body)
    body = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", body)
    body = re.sub(r"^#{1,6}\s+", "", body, flags=re.M)
    body = re.sub(r"[*_`>#\-]+", " ", body)
    return norm_text(body)


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text, flags=re.UNICODE))


def load_candidates(path: Path) -> dict[str, dict]:
    if not path or not path.exists():
        return {}

    out = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = first_existing(row, ["url", "article_url", "link", "href"])
            cu = canonical_url(url)
            if not cu:
                continue

            anchor = first_existing(row, [
                "anchor_entity",
                "anchor",
                "query",
                "keyword",
                "topic",
                "matched_anchor",
            ])
            source = first_existing(row, ["source", "domain"])
            title = first_existing(row, ["title", "article_title"])

            # một URL có thể xuất hiện nhiều anchor, gộp anchor lại
            if cu not in out:
                out[cu] = {
                    "source": source,
                    "candidate_title": title,
                    "anchors": set(),
                    "raw_url": url,
                }
            if anchor:
                out[cu]["anchors"].add(anchor)

    for item in out.values():
        item["anchors"] = sorted(item["anchors"])

    return out


def load_topic_keywords(path: Path) -> tuple[list[str], dict[str, str]]:
    if not path or not path.exists():
        return [], {}

    data = json.loads(path.read_text(encoding="utf-8"))
    kws = []
    kw_to_cluster = {}

    for cluster in data.get("clusters", []):
        cluster_name = cluster.get("cluster", "")
        for a in cluster.get("anchors", []):
            terms = [a.get("anchor_entity", "")] + list(a.get("aliases", []) or [])
            for term in terms:
                term = norm_text(term)
                if not term:
                    continue
                kws.append(term)
                kw_to_cluster[norm_key(term)] = cluster_name

    # unique keep order
    seen = set()
    unique = []
    for k in kws:
        nk = norm_key(k)
        if nk and nk not in seen:
            seen.add(nk)
            unique.append(k)

    return unique, kw_to_cluster


def find_keywords(text: str, keywords: list[str], limit: int = 12) -> list[str]:
    t = norm_key(text)
    hits = []
    seen = set()

    for kw in keywords:
        nk = norm_key(kw)
        if not nk or nk in seen:
            continue
        if re.search(rf"(^|\s){re.escape(nk)}($|\s)", t):
            hits.append(kw)
            seen.add(nk)
        if len(hits) >= limit:
            break

    return hits


def guess_topic(matched_keywords: list[str], kw_to_cluster: dict[str, str]) -> str:
    counts = {}
    for kw in matched_keywords:
        cluster = kw_to_cluster.get(norm_key(kw), "")
        if cluster:
            counts[cluster] = counts.get(cluster, 0) + 1
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]


def short_hash(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:n]


def build_rows_from_markdown(
    markdown_dir: Path,
    candidates_by_url: dict[str, dict],
    keywords: list[str],
    kw_to_cluster: dict[str, str],
    excerpt_chars: int,
) -> list[dict]:
    rows = []

    files = sorted(markdown_dir.rglob("*.md"))
    for path in files:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        meta, body = parse_frontmatter(raw)

        url = (
            meta.get("url")
            or meta.get("source_url")
            or meta.get("article_url")
            or meta.get("canonical_url")
            or ""
        )
        cu = canonical_url(url)

        candidate = candidates_by_url.get(cu, {}) if cu else {}

        title = extract_title(meta, body, path)
        source = meta.get("source") or candidate.get("source", "")
        headings = extract_headings(body)
        plain = markdown_to_plain(body)
        excerpt = plain[:excerpt_chars]

        matched = find_keywords(
            "\n".join([title, " ".join(headings), plain[:5000]]),
            keywords,
            limit=12,
        )
        topic_hint = guess_topic(matched, kw_to_cluster)

        anchors = candidate.get("anchors", [])
        anchor_text = "; ".join(anchors) if anchors else ""

        rows.append({
            "id": "",
            "decision": "",
            "quality_score": "",
            "source": source,
            "url": url,
            "canonical_url": cu,
            "title": title,
            "candidate_anchors": anchor_text,
            "matched_keywords": "; ".join(matched),
            "topic_hint": topic_hint,
            "word_count": word_count(plain),
            "heading_count": len(headings),
            "headings": " | ".join(headings[:20]),
            "excerpt": excerpt,
            "md_path": str(path),
            "title_key": norm_key(title),
            "content_hash": short_hash(plain[:3000]),
        })

    return rows


def build_rows_from_candidates_only(
    candidates_by_url: dict[str, dict],
    keywords: list[str],
    kw_to_cluster: dict[str, str],
) -> list[dict]:
    rows = []
    for cu, item in sorted(candidates_by_url.items()):
        title = item.get("candidate_title", "")
        anchors = item.get("anchors", [])
        text = title + " " + " ".join(anchors)
        matched = find_keywords(text, keywords, limit=12)
        topic_hint = guess_topic(matched, kw_to_cluster)

        rows.append({
            "id": "",
            "decision": "",
            "quality_score": "",
            "source": item.get("source", ""),
            "url": item.get("raw_url", cu),
            "canonical_url": cu,
            "title": title,
            "candidate_anchors": "; ".join(anchors),
            "matched_keywords": "; ".join(matched),
            "topic_hint": topic_hint,
            "word_count": 0,
            "heading_count": 0,
            "headings": "",
            "excerpt": "",
            "md_path": "",
            "title_key": norm_key(title),
            "content_hash": "",
        })
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return

    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_batches(rows: list[dict], batch_dir: Path, batch_size: int) -> None:
    batch_dir.mkdir(parents=True, exist_ok=True)

    for old in batch_dir.glob("batch_*.md"):
        old.unlink()

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        batch_no = start // batch_size + 1
        path = batch_dir / f"batch_{batch_no:03d}.md"

        lines = []
        lines.append("# GPT corpus filtering batch")
        lines.append("")
        lines.append("Hãy phân loại từng bài thành APPROVE / REJECT / REVIEW theo prompt lọc corpus đã cung cấp.")
        lines.append("Output bảng: id | decision | quality_score | topic_group | duplicate_group | reason")
        lines.append("")

        for row in batch:
            lines.append(f"## ID {row['id']}")
            lines.append(f"- source: {row['source']}")
            lines.append(f"- url: {row['url']}")
            lines.append(f"- title: {row['title']}")
            lines.append(f"- candidate_anchors: {row['candidate_anchors']}")
            lines.append(f"- matched_keywords: {row['matched_keywords']}")
            lines.append(f"- topic_hint: {row['topic_hint']}")
            lines.append(f"- word_count: {row['word_count']}")
            lines.append(f"- headings: {row['headings']}")
            lines.append("")
            lines.append("excerpt:")
            lines.append("```text")
            lines.append(row["excerpt"])
            lines.append("```")
            lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates-csv", required=True)
    parser.add_argument("--markdown-dir", default="")
    parser.add_argument("--topics-file", default="data/crawl/topic_seed_dermatology_v1.json")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--excerpt-chars", type=int, default=1400)
    args = parser.parse_args()

    candidates = load_candidates(Path(args.candidates_csv))
    keywords, kw_to_cluster = load_topic_keywords(Path(args.topics_file))

    md_dir = Path(args.markdown_dir) if args.markdown_dir else None
    if md_dir and md_dir.exists() and list(md_dir.rglob("*.md")):
        rows = build_rows_from_markdown(
            markdown_dir=md_dir,
            candidates_by_url=candidates,
            keywords=keywords,
            kw_to_cluster=kw_to_cluster,
            excerpt_chars=args.excerpt_chars,
        )
    else:
        rows = build_rows_from_candidates_only(
            candidates_by_url=candidates,
            keywords=keywords,
            kw_to_cluster=kw_to_cluster,
        )

    # sort: bài dài + đúng topic lên trước, source ổn định
    rows.sort(key=lambda r: (r["source"], -int(r["word_count"] or 0), r["title"]))

    for i, row in enumerate(rows, start=1):
        row["id"] = f"A{i:05d}"

    write_csv(rows, Path(args.output_csv))
    write_jsonl(rows, Path(args.output_jsonl))
    write_batches(rows, Path(args.batch_dir), args.batch_size)

    print(f"rows: {len(rows)}")
    print(f"csv: {args.output_csv}")
    print(f"jsonl: {args.output_jsonl}")
    print(f"batch_dir: {args.batch_dir}")


if __name__ == "__main__":
    main()