from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterator

from openai import OpenAI


ROOT_DIR = Path(__file__).resolve().parents[1]


def load_project_env() -> None:
    """
    Load OPENAI_API_KEY từ .env ở root project.
    Có python-dotenv thì dùng, không có thì fallback đọc thủ công.
    """
    env_path = ROOT_DIR / ".env"

    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
        return
    except ModuleNotFoundError:
        pass

    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


load_project_env()


SYSTEM_PROMPT = """
Bạn là trợ lý lọc content-level cho corpus Vietnamese Medical Hybrid RAG.

Phạm vi corpus:
- Da liễu mở rộng: bệnh da, tóc, móng, da đầu, dị ứng da, bệnh có ban/phát ban/tổn thương ngoài da.

Bạn nhận article preview gồm:
id, source, url, title, candidate_anchors, matched_keywords, topic_hint,
word_count, heading_count, headings, excerpt.

Bạn chỉ được dùng thông tin trong preview.
Không được bịa thêm nội dung ngoài preview.

Nhiệm vụ:
Phân loại từng article thành APPROVE / REJECT / REVIEW.

APPROVE nếu:
- Trọng tâm bài thuộc da liễu mở rộng.
- Nội dung có tri thức y khoa đủ dùng cho RAG: nguyên nhân, triệu chứng, yếu tố nguy cơ, chẩn đoán, điều trị, chăm sóc, phòng ngừa, biến chứng hoặc phân biệt bệnh.
- Là article thật, không phải category/tag/search/landing page.
- Có khả năng extract được entity/relation y khoa hữu ích.

REJECT nếu:
- Trọng tâm nằm ngoài scope: hô hấp, tim mạch, tiêu hóa, sản khoa, thần kinh, cơ xương khớp, dinh dưỡng, vaccine/chủng ngừa chung... và biểu hiện da không phải trọng tâm.
- Chủ yếu là quảng bá dịch vụ, đặt lịch, bảng giá, giới thiệu bác sĩ/khoa/phòng khám, tin tức bệnh viện.
- Làm đẹp/thẩm mỹ thuần túy, skincare chung, không có tri thức bệnh học.
- Quá chung chung, crawl lỗi, hoặc không đủ thông tin để chunk/extract.
- Gần trùng với bài khác và là bản kém hơn.

REVIEW nếu:
- Có vẻ liên quan nhưng preview chưa đủ chắc.
- Bệnh toàn thân nhưng có biểu hiện da quan trọng.
- Bài vaccine/dị ứng/virus/trẻ em có thể liên quan da nhưng chưa rõ trọng tâm.
- Có dấu hiệu duplicate nhưng chưa chắc.
- Không đủ chắc để APPROVE hoặc REJECT.

Quality score:
5 = rất tốt, đúng scope, giàu tri thức y khoa.
4 = tốt, đúng scope.
3 = dùng được, không quá sâu.
2 = yếu hoặc hơi lệch, thường REVIEW/REJECT.
1 = ngoài scope/rác/không đủ tri thức.
0 = lỗi crawl/không phải article.

Topic group:
Chọn một trong:
viem_da_cham, mun_nang_long, nam_da, di_ung_may_day, vay_sung_sac_to,
virus_ban_da, nhiem_khuan_da, toc_mong_da_dau, sac_to_da,
trieu_chung_da, other_skin_related, out_of_scope

Nguyên tắc quan trọng:
- Không reject quá tay. Nếu chưa chắc, chọn REVIEW.
- Không bịa nội dung ngoài title/headings/excerpt.
- Triệu chứng da như ngứa, phát ban, nổi mẩn, da khô, mụn nước vẫn có thể APPROVE nếu bài có nội dung y khoa.
- Với sởi, rubella, thủy đậu, tay chân miệng, zona, herpes: APPROVE nếu nói về bệnh/triệu chứng/chăm sóc/biến chứng; REVIEW hoặc REJECT nếu chỉ nói lịch tiêm/xét nghiệm.
- Với lupus, dị ứng thuốc, bệnh toàn thân: REVIEW nếu preview có biểu hiện da; REJECT nếu trọng tâm không phải da.
- Với vaccine/chủng ngừa: REJECT nếu chỉ là lịch tiêm, giá tiêm, nơi tiêm; REVIEW nếu có phần bệnh/triệu chứng da rõ.
- Với bài làm đẹp: REJECT nếu chỉ skincare/thẩm mỹ; APPROVE/REVIEW nếu có bệnh/tình trạng y khoa thật như nám, mụn, tăng sắc tố.

Output:
- Trả về JSON hợp lệ, không markdown.
- Mỗi input id phải có đúng một item output.
- Không bỏ sót id.
- Schema:
{
  "items": [
    {
      "id": "A00001",
      "decision": "APPROVE|REJECT|REVIEW",
      "quality_score": 0,
      "topic_group": "string",
      "duplicate_group": "string",
      "reason": "lý do ngắn bằng tiếng Việt"
    }
  ]
}
"""


VALID_DECISIONS = {"APPROVE", "REJECT", "REVIEW"}

VALID_TOPIC_GROUPS = {
    "viem_da_cham",
    "mun_nang_long",
    "nam_da",
    "di_ung_may_day",
    "vay_sung_sac_to",
    "virus_ban_da",
    "nhiem_khuan_da",
    "toc_mong_da_dau",
    "sac_to_da",
    "trieu_chung_da",
    "other_skin_related",
    "out_of_scope",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def first_existing(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def make_id(row_number: int) -> str:
    return f"C{row_number:05d}"


def safe_int(value: Any, default: int = -1) -> int:
    try:
        s = str(value or "").strip()
        if not s:
            return default
        return int(float(s))
    except Exception:
        return default


def truncate_text(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def compact_row(row: dict[str, Any], excerpt_chars: int, headings_chars: int) -> dict[str, Any]:
    return {
        "id": str(row.get("id", "")).strip(),
        "source": first_existing(row, ["source", "domain"]),
        "url": first_existing(row, ["url", "article_url", "link", "href"]),
        "title": first_existing(row, ["title", "article_title", "name"]),
        "candidate_anchors": first_existing(row, ["candidate_anchors", "anchors", "anchor_entity", "anchor"]),
        "matched_keywords": first_existing(row, ["matched_keywords", "keywords"]),
        "topic_hint": first_existing(row, ["topic_hint", "cluster", "topic_group"]),
        "word_count": first_existing(row, ["word_count", "words", "num_words"]),
        "heading_count": first_existing(row, ["heading_count", "num_headings"]),
        "headings": truncate_text(first_existing(row, ["headings", "heading_text"]), headings_chars),
        "excerpt": truncate_text(first_existing(row, ["excerpt", "preview", "text_preview"]), excerpt_chars),
    }


def input_rows_stream(input_csv: Path) -> tuple[Iterator[dict[str, Any]], list[str]]:
    f = input_csv.open("r", encoding="utf-8-sig", newline="")
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames or []

    has_id = "id" in fieldnames
    output_fieldnames = list(fieldnames)

    if not has_id:
        output_fieldnames = ["id"] + output_fieldnames

    def gen() -> Iterator[dict[str, Any]]:
        try:
            for i, row in enumerate(reader, start=1):
                if not has_id or not str(row.get("id", "")).strip():
                    row["id"] = make_id(i)
                yield row
        finally:
            f.close()

    return gen(), output_fieldnames


def read_processed_ids(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()

    done: set[str] = set()

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rid = str(row.get("id", "")).strip()
            if rid:
                done.add(rid)

    return done


def extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"(\{.*\})", text, flags=re.S)
    if not match:
        raise ValueError("Cannot find JSON object in model response.")

    return json.loads(match.group(1))


def normalize_decision(value: str) -> str:
    value = str(value or "").strip().upper()
    if value in VALID_DECISIONS:
        return value
    return "REVIEW"


def normalize_quality_score(value: Any) -> int:
    try:
        score = int(float(value))
    except Exception:
        return 2

    return max(0, min(5, score))


def normalize_topic_group(value: str, decision: str) -> str:
    value = str(value or "").strip()

    if value in VALID_TOPIC_GROUPS:
        return value

    if decision == "REJECT":
        return "out_of_scope"

    return "other_skin_related"


def call_llm(
    client: OpenAI,
    model: str,
    batch_rows: list[dict[str, Any]],
    temperature: float,
    max_retries: int,
) -> list[dict[str, Any]]:
    wanted_ids = [str(r["id"]) for r in batch_rows]

    payload = {
        "task": "Classify content-level relevance for dermatology corpus using article previews. Return JSON only.",
        "must_include_ids": wanted_ids,
        "items": batch_rows,
    }

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
            )

            content = response.choices[0].message.content or ""
            data = extract_json(content)
            items = data.get("items", [])

            if not isinstance(items, list):
                raise ValueError("Model output has no valid items list.")

            out = []
            seen = set()

            for item in items:
                rid = str(item.get("id", "")).strip()

                if rid not in wanted_ids or rid in seen:
                    continue

                decision = normalize_decision(item.get("decision", ""))
                score = normalize_quality_score(item.get("quality_score", 2))
                topic_group = normalize_topic_group(item.get("topic_group", ""), decision)

                seen.add(rid)
                out.append({
                    "id": rid,
                    "content_decision": decision,
                    "quality_score": score,
                    "topic_group": topic_group,
                    "duplicate_group": str(item.get("duplicate_group", "") or "").strip(),
                    "content_reason": str(item.get("reason", "") or "").strip(),
                })

            missing = [rid for rid in wanted_ids if rid not in seen]

            for rid in missing:
                out.append({
                    "id": rid,
                    "content_decision": "REVIEW",
                    "quality_score": 2,
                    "topic_group": "other_skin_related",
                    "duplicate_group": "",
                    "content_reason": "Model output thiếu ID này, đưa vào REVIEW để tránh loại nhầm.",
                })

            return out

        except Exception as exc:
            last_error = exc
            wait = min(30, 2 ** attempt)
            print(f"[WARN] LLM failed attempt {attempt}/{max_retries}: {exc}")
            time.sleep(wait)

    raise RuntimeError(f"LLM failed after {max_retries} retries: {last_error}")


def open_append_writer(path: Path, fieldnames: list[str]) -> tuple[Any, csv.DictWriter]:
    ensure_dir(path.parent)

    file_exists = path.exists() and path.stat().st_size > 0
    f = path.open("a", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")

    if not file_exists:
        writer.writeheader()
        f.flush()
        os.fsync(f.fileno())

    return f, writer


def flush_files(files: dict[str, Any]) -> None:
    for f in files.values():
        f.flush()
        os.fsync(f.fileno())


def next_raw_batch_no(raw_dir: Path) -> int:
    if not raw_dir.exists():
        return 1

    max_no = 0

    for path in raw_dir.glob("content_filter_batch_*.json"):
        match = re.search(r"content_filter_batch_(\d+)\.json$", path.name)
        if match:
            max_no = max(max_no, int(match.group(1)))

    return max_no + 1


def write_raw_batch(raw_dir: Path, batch_no: int, obj: dict[str, Any]) -> None:
    ensure_dir(raw_dir)

    tmp = raw_dir / f"content_filter_batch_{batch_no:04d}.json.tmp"
    final = raw_dir / f"content_filter_batch_{batch_no:04d}.json"

    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(final)


def final_keep_value(decision: str, quality_score: int, min_approve_score: int) -> str:
    if decision == "APPROVE" and quality_score >= min_approve_score:
        return "1"
    return "0"


def write_direct_row(
    row: dict[str, Any],
    decision: str,
    quality_score: int,
    topic_group: str,
    reason: str,
    output_writers: dict[str, csv.DictWriter],
    output_files: dict[str, Any],
    min_approve_score: int,
) -> str:
    out_row = dict(row)
    out_row["content_decision"] = decision
    out_row["quality_score"] = quality_score
    out_row["topic_group"] = topic_group
    out_row["duplicate_group"] = ""
    out_row["content_reason"] = reason
    out_row["final_keep"] = final_keep_value(decision, quality_score, min_approve_score)

    output_writers["all"].writerow(out_row)

    if decision == "APPROVE" and out_row["final_keep"] == "1":
        output_writers["approved"].writerow(out_row)
    elif decision == "REJECT":
        output_writers["rejected"].writerow(out_row)
    else:
        output_writers["review"].writerow(out_row)

    flush_files(output_files)
    return decision


def process_batch(
    client: OpenAI,
    model: str,
    batch_no: int,
    original_rows: list[dict[str, Any]],
    compact_rows: list[dict[str, Any]],
    output_writers: dict[str, csv.DictWriter],
    output_files: dict[str, Any],
    raw_dir: Path,
    temperature: float,
    max_retries: int,
    min_approve_score: int,
) -> tuple[int, int, int]:
    print(f"[CALL] batch {batch_no:04d}: {compact_rows[0]['id']}..{compact_rows[-1]['id']}")

    decisions = call_llm(
        client=client,
        model=model,
        batch_rows=compact_rows,
        temperature=temperature,
        max_retries=max_retries,
    )

    decision_by_id = {d["id"]: d for d in decisions}

    write_raw_batch(raw_dir, batch_no, {
        "batch_no": batch_no,
        "model": model,
        "ids": [r["id"] for r in compact_rows],
        "inputs": compact_rows,
        "decisions": decisions,
    })

    n_approve = 0
    n_review = 0
    n_reject = 0

    for row in original_rows:
        rid = str(row.get("id", "")).strip()

        decision = decision_by_id.get(rid) or {
            "content_decision": "REVIEW",
            "quality_score": 2,
            "topic_group": "other_skin_related",
            "duplicate_group": "",
            "content_reason": "Không có decision, đưa vào REVIEW để tránh loại nhầm.",
        }

        content_decision = decision["content_decision"]
        quality_score = int(decision["quality_score"])

        out_row = dict(row)
        out_row["content_decision"] = content_decision
        out_row["quality_score"] = quality_score
        out_row["topic_group"] = decision["topic_group"]
        out_row["duplicate_group"] = decision["duplicate_group"]
        out_row["content_reason"] = decision["content_reason"]
        out_row["final_keep"] = final_keep_value(content_decision, quality_score, min_approve_score)

        output_writers["all"].writerow(out_row)

        if content_decision == "APPROVE" and out_row["final_keep"] == "1":
            output_writers["approved"].writerow(out_row)
            n_approve += 1
        elif content_decision == "REJECT":
            output_writers["rejected"].writerow(out_row)
            n_reject += 1
        else:
            output_writers["review"].writerow(out_row)
            n_review += 1

    flush_files(output_files)

    return n_approve, n_review, n_reject


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", default="data/debug/gpt_filter/content_decisions_api_common_v1")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sleep", type=float, default=0.3)
    parser.add_argument("--limit", type=int, default=0, help="Giới hạn số dòng mới xử lý trong run hiện tại.")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--excerpt-chars", type=int, default=1200)
    parser.add_argument("--headings-chars", type=int, default=700)
    parser.add_argument("--min-approve-score", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw_batches"
    clean_dir = output_dir / "clean"

    all_csv = clean_dir / "articles_with_content_decisions.csv"
    approved_csv = clean_dir / "articles_approved.csv"
    review_csv = clean_dir / "articles_review.csv"
    rejected_csv = clean_dir / "articles_rejected.csv"

    row_iter, original_fieldnames = input_rows_stream(input_csv)

    output_fieldnames = list(original_fieldnames)

    for extra in [
        "content_decision",
        "quality_score",
        "topic_group",
        "duplicate_group",
        "content_reason",
        "final_keep",
    ]:
        if extra not in output_fieldnames:
            output_fieldnames.append(extra)

    processed_ids = read_processed_ids(all_csv)

    print(f"Input CSV: {input_csv}")
    print(f"Output dir: {output_dir}")
    print(f"Project root: {ROOT_DIR}")
    print(f"Already processed ids: {len(processed_ids)}")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Excerpt chars: {args.excerpt_chars}")
    print(f"Headings chars: {args.headings_chars}")
    print(f"Min approve score: {args.min_approve_score}")

    if args.dry_run:
        sample = []

        for row in row_iter:
            if len(sample) >= 5:
                break
            sample.append(compact_row(row, args.excerpt_chars, args.headings_chars))

        print(json.dumps(sample, ensure_ascii=False, indent=2))
        return

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Missing OPENAI_API_KEY. "
            f"Script đã thử đọc .env tại: {ROOT_DIR / '.env'}"
        )

    client = OpenAI()

    all_f, all_writer = open_append_writer(all_csv, output_fieldnames)
    approved_f, approved_writer = open_append_writer(approved_csv, output_fieldnames)
    review_f, review_writer = open_append_writer(review_csv, output_fieldnames)
    rejected_f, rejected_writer = open_append_writer(rejected_csv, output_fieldnames)

    output_files = {
        "all": all_f,
        "approved": approved_f,
        "review": review_f,
        "rejected": rejected_f,
    }

    output_writers = {
        "all": all_writer,
        "approved": approved_writer,
        "review": review_writer,
        "rejected": rejected_writer,
    }

    total_seen = 0
    total_selected_this_run = 0
    total_skipped = 0
    total_submitted = 0
    total_approve = 0
    total_review = 0
    total_reject = 0

    batch_original: list[dict[str, Any]] = []
    batch_compact: list[dict[str, Any]] = []
    batch_no = next_raw_batch_no(raw_dir)

    try:
        for row in row_iter:
            total_seen += 1

            rid = str(row.get("id", "")).strip()

            if rid in processed_ids:
                total_skipped += 1
                continue

            if args.limit > 0 and total_selected_this_run >= args.limit:
                break

            total_selected_this_run += 1

            c = compact_row(row, args.excerpt_chars, args.headings_chars)

            # Nếu preview gần như rỗng thì không tốn GPT, đưa vào REVIEW.
            if not c["title"] and not c["excerpt"] and not c["headings"]:
                decision = write_direct_row(
                    row=row,
                    decision="REVIEW",
                    quality_score=2,
                    topic_group="other_skin_related",
                    reason="Preview thiếu title/headings/excerpt, cần kiểm tra tay.",
                    output_writers=output_writers,
                    output_files=output_files,
                    min_approve_score=args.min_approve_score,
                )

                if decision == "REVIEW":
                    total_review += 1

                processed_ids.add(rid)
                continue

            batch_original.append(row)
            batch_compact.append(c)

            if len(batch_original) >= args.batch_size:
                n_approve, n_review, n_reject = process_batch(
                    client=client,
                    model=args.model,
                    batch_no=batch_no,
                    original_rows=batch_original,
                    compact_rows=batch_compact,
                    output_writers=output_writers,
                    output_files=output_files,
                    raw_dir=raw_dir,
                    temperature=args.temperature,
                    max_retries=args.max_retries,
                    min_approve_score=args.min_approve_score,
                )

                batch_no += 1
                total_submitted += len(batch_original)
                total_approve += n_approve
                total_review += n_review
                total_reject += n_reject

                for r in batch_original:
                    processed_ids.add(str(r.get("id", "")).strip())

                batch_original = []
                batch_compact = []

                time.sleep(args.sleep)

        if batch_original:
            n_approve, n_review, n_reject = process_batch(
                client=client,
                model=args.model,
                batch_no=batch_no,
                original_rows=batch_original,
                compact_rows=batch_compact,
                output_writers=output_writers,
                output_files=output_files,
                raw_dir=raw_dir,
                temperature=args.temperature,
                max_retries=args.max_retries,
                min_approve_score=args.min_approve_score,
            )

            total_submitted += len(batch_original)
            total_approve += n_approve
            total_review += n_review
            total_reject += n_reject

    finally:
        for f in output_files.values():
            f.close()

    print("\nDONE")
    print(f"seen rows: {total_seen}")
    print(f"selected this run: {total_selected_this_run}")
    print(f"skipped existing: {total_skipped}")
    print(f"submitted this run: {total_submitted}")
    print(f"APPROVE this run: {total_approve}")
    print(f"REVIEW this run: {total_review}")
    print(f"REJECT this run: {total_reject}")
    print(f"all csv: {all_csv}")
    print(f"approved csv: {approved_csv}")
    print(f"review csv: {review_csv}")
    print(f"rejected csv: {rejected_csv}")
    print(f"raw batches: {raw_dir}")


if __name__ == "__main__":
    main()