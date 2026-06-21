
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
    Load OPENAI_API_KEY từ file .env ở root project.

    Ưu tiên python-dotenv nếu có.
    Nếu chưa cài python-dotenv thì fallback đọc .env thủ công.
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
Bạn là trợ lý lọc link-level cho corpus Vietnamese Medical Hybrid RAG.

Phạm vi corpus:
- Da liễu mở rộng: bệnh da, tóc, móng, da đầu, dị ứng da, bệnh có ban/phát ban/tổn thương ngoài da.

Bạn chỉ được dùng metadata:
id, source, url, title, anchor/candidate_anchors, matched_keywords, topic_hint.
Không được bịa nội dung ngoài metadata.

Nhiệm vụ:
Phân loại từng link thành LINK_PASS / LINK_REJECT / NEED_CONTENT.

LINK_PASS nếu title/url/metadata rõ ràng liên quan đến:
- Viêm da/chàm: viêm da cơ địa, chàm, chàm sữa, viêm da tiếp xúc, viêm da tiết bã, viêm da dị ứng, hăm tã, rôm sảy, khô da.
- Mụn/nang lông: mụn trứng cá, mụn đầu đen, mụn bọc, mụn nhọt, viêm nang lông, mụn cóc.
- Nấm da: nấm da, hắc lào, lang ben, nấm móng, nấm da đầu.
- Dị ứng/miễn dịch biểu hiện da: mày đay, mề đay, dị ứng da, dị ứng thời tiết, dị ứng thuốc gây phát ban, phù mạch.
- Triệu chứng ngoài da: ngứa da, nổi mẩn đỏ, phát ban, ban đỏ, bong tróc da, da khô, mụn nước, mụn mủ, sưng đỏ da, loét da.
- Vảy/sắc tố: vảy nến, á sừng, dày sừng nang lông, bạch biến, nám da, tăng sắc tố da.
- Bệnh có ban/tổn thương da: zona thần kinh, thủy đậu, tay chân miệng, sởi, rubella, herpes, u mềm lây.
- Tóc/móng/da đầu: rụng tóc, gàu, nấm da đầu, nấm móng.

LINK_REJECT nếu rõ ràng ngoài scope:
- Trọng tâm là hô hấp, tim mạch, tiêu hóa, tiết niệu, sản khoa, thần kinh, cơ xương khớp, dinh dưỡng, nha khoa, mắt, tai mũi họng... và không có dấu hiệu tập trung vào biểu hiện da.
- Bài chăm sóc chung không liên quan da: thở máy, nôn trớ, mất nước, mọc răng, béo phì, dinh dưỡng, tiêm chủng chung.
- Bài dịch vụ/bệnh viện: bảng giá, đặt lịch, giới thiệu bác sĩ/khoa/phòng khám, tin tức bệnh viện, ưu đãi, tuyển dụng.
- Category/tag/search/author/video/landing page.
- Làm đẹp thuần túy không có bệnh/tình trạng y khoa.

NEED_CONTENT nếu:
- Title mơ hồ, có thể liên quan da nhưng chưa chắc.
- Bệnh toàn thân nhưng có khả năng biểu hiện da quan trọng.
- Bài vaccine/dị ứng/virus/trẻ em có thể liên quan da nhưng chưa rõ trọng tâm.
- candidate_anchors liên quan nhưng title không đủ chắc.
- Không chắc thì chọn NEED_CONTENT, không REJECT quá tay.

Output:
- Trả về JSON hợp lệ, không markdown.
- Mỗi input id phải có đúng một item output.
- Không bỏ sót id.
- Schema:
{
  "items": [
    {
      "id": "A00001",
      "decision": "LINK_PASS|LINK_REJECT|NEED_CONTENT",
      "confidence": 0.0,
      "reason": "lý do ngắn"
    }
  ]
}
"""


def first_existing(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_id(row_number: int) -> str:
    return f"A{row_number:05d}"


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row.get("id", "")).strip(),
        "source": first_existing(row, ["source", "domain"]),
        "url": first_existing(row, ["url", "article_url", "link", "href"]),
        "title": first_existing(row, ["title", "article_title", "name"]),
        "anchor": first_existing(row, ["anchor_entity", "anchor", "query", "keyword", "topic"]),
        "candidate_anchors": first_existing(row, ["candidate_anchors", "anchors"]),
        "matched_keywords": first_existing(row, ["matched_keywords", "keywords"]),
        "topic_hint": first_existing(row, ["topic_hint", "cluster", "topic_group"]),
    }


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

    if value in {"LINK_PASS", "LINK_REJECT", "NEED_CONTENT"}:
        return value

    return "NEED_CONTENT"


def normalize_confidence(value: Any) -> float:
    try:
        v = float(value)
    except Exception:
        return 0.5

    return max(0.0, min(1.0, v))


def call_llm(
    client: OpenAI,
    model: str,
    batch_rows: list[dict[str, Any]],
    temperature: float,
    max_retries: int,
) -> list[dict[str, Any]]:
    wanted_ids = [str(r["id"]) for r in batch_rows]

    payload = {
        "task": "Classify link-level relevance for dermatology corpus. Return JSON only.",
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

                seen.add(rid)
                out.append({
                    "id": rid,
                    "llm_decision": normalize_decision(item.get("decision", "")),
                    "llm_confidence": normalize_confidence(item.get("confidence", 0.5)),
                    "llm_reason": str(item.get("reason", "") or "").strip(),
                })

            missing = [rid for rid in wanted_ids if rid not in seen]

            for rid in missing:
                out.append({
                    "id": rid,
                    "llm_decision": "NEED_CONTENT",
                    "llm_confidence": 0.5,
                    "llm_reason": "Model output thiếu ID này, giữ lại để tránh loại nhầm.",
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
    """
    Tránh overwrite raw batch khi resume.
    Nếu đã có link_filter_batch_0001.json ... thì lần sau bắt đầu từ max + 1.
    """
    if not raw_dir.exists():
        return 1

    max_no = 0

    for path in raw_dir.glob("link_filter_batch_*.json"):
        match = re.search(r"link_filter_batch_(\d+)\.json$", path.name)
        if match:
            max_no = max(max_no, int(match.group(1)))

    return max_no + 1


def write_raw_batch(raw_dir: Path, batch_no: int, obj: dict[str, Any]) -> None:
    ensure_dir(raw_dir)

    tmp = raw_dir / f"link_filter_batch_{batch_no:04d}.json.tmp"
    final = raw_dir / f"link_filter_batch_{batch_no:04d}.json"

    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(final)


def write_no_model_row(
    row: dict[str, Any],
    decision: str,
    confidence: float,
    reason: str,
    output_writers: dict[str, csv.DictWriter],
    output_files: dict[str, Any],
) -> str:
    crawl_keep = "1" if decision in {"LINK_PASS", "NEED_CONTENT"} else "0"

    out_row = dict(row)
    out_row["llm_decision"] = decision
    out_row["llm_confidence"] = confidence
    out_row["llm_reason"] = reason
    out_row["crawl_keep"] = crawl_keep

    output_writers["all"].writerow(out_row)

    if crawl_keep == "1":
        output_writers["kept"].writerow(out_row)
    else:
        output_writers["rejected"].writerow(out_row)

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

    # Ghi raw batch trước để audit/debug.
    write_raw_batch(raw_dir, batch_no, {
        "batch_no": batch_no,
        "model": model,
        "ids": [r["id"] for r in compact_rows],
        "inputs": compact_rows,
        "decisions": decisions,
    })

    n_pass = 0
    n_need = 0
    n_reject = 0

    for row in original_rows:
        rid = str(row.get("id", "")).strip()

        decision = decision_by_id.get(rid) or {
            "llm_decision": "NEED_CONTENT",
            "llm_confidence": 0.5,
            "llm_reason": "Không có decision, giữ lại để tránh loại nhầm.",
        }

        llm_decision = decision["llm_decision"]
        crawl_keep = "1" if llm_decision in {"LINK_PASS", "NEED_CONTENT"} else "0"

        out_row = dict(row)
        out_row["llm_decision"] = llm_decision
        out_row["llm_confidence"] = decision["llm_confidence"]
        out_row["llm_reason"] = decision["llm_reason"]
        out_row["crawl_keep"] = crawl_keep

        output_writers["all"].writerow(out_row)

        if crawl_keep == "1":
            output_writers["kept"].writerow(out_row)
        else:
            output_writers["rejected"].writerow(out_row)

        if llm_decision == "LINK_PASS":
            n_pass += 1
        elif llm_decision == "NEED_CONTENT":
            n_need += 1
        else:
            n_reject += 1

    # Flush sau mỗi batch để crash cũng không mất.
    flush_files(output_files)

    return n_pass, n_need, n_reject


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", default="data/debug/gpt_filter/link_decisions_api_common_v1")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sleep", type=float, default=0.3)
    parser.add_argument("--limit", type=int, default=0, help="Giới hạn số dòng mới cần xử lý trong run hiện tại.")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw_batches"
    clean_dir = output_dir / "clean"

    all_csv = clean_dir / "candidate_links_with_llm_decisions.csv"
    kept_csv = clean_dir / "candidate_links_kept_for_crawl.csv"
    rejected_csv = clean_dir / "candidate_links_rejected.csv"

    row_iter, original_fieldnames = input_rows_stream(input_csv)

    output_fieldnames = list(original_fieldnames)

    for extra in ["llm_decision", "llm_confidence", "llm_reason", "crawl_keep"]:
        if extra not in output_fieldnames:
            output_fieldnames.append(extra)

    processed_ids = read_processed_ids(all_csv)

    print(f"Input CSV: {input_csv}")
    print(f"Output dir: {output_dir}")
    print(f"Project root: {ROOT_DIR}")
    print(f"Already processed ids: {len(processed_ids)}")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")

    if args.dry_run:
        sample = []

        for row in row_iter:
            if len(sample) >= 5:
                break
            sample.append(compact_row(row))

        print(json.dumps(sample, ensure_ascii=False, indent=2))
        return

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Missing OPENAI_API_KEY. "
            f"Script đã thử đọc .env tại: {ROOT_DIR / '.env'}"
        )

    client = OpenAI()

    all_f, all_writer = open_append_writer(all_csv, output_fieldnames)
    kept_f, kept_writer = open_append_writer(kept_csv, output_fieldnames)
    rejected_f, rejected_writer = open_append_writer(rejected_csv, output_fieldnames)

    output_files = {
        "all": all_f,
        "kept": kept_f,
        "rejected": rejected_f,
    }

    output_writers = {
        "all": all_writer,
        "kept": kept_writer,
        "rejected": rejected_writer,
    }

    total_seen = 0
    total_selected_this_run = 0
    total_submitted = 0
    total_skipped = 0
    total_pass = 0
    total_need = 0
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

            c = compact_row(row)

            # Nếu không có cả URL lẫn title thì không gửi GPT, giữ lại để tránh loại nhầm.
            if not c["url"] and not c["title"]:
                decision = write_no_model_row(
                    row=row,
                    decision="NEED_CONTENT",
                    confidence=0.5,
                    reason="Thiếu title/url, giữ lại để xem nội dung.",
                    output_writers=output_writers,
                    output_files=output_files,
                )

                if decision == "NEED_CONTENT":
                    total_need += 1

                processed_ids.add(rid)
                continue

            batch_original.append(row)
            batch_compact.append(c)

            if len(batch_original) >= args.batch_size:
                n_pass, n_need, n_reject = process_batch(
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
                )

                batch_no += 1
                total_submitted += len(batch_original)
                total_pass += n_pass
                total_need += n_need
                total_reject += n_reject

                for r in batch_original:
                    processed_ids.add(str(r.get("id", "")).strip())

                batch_original = []
                batch_compact = []

                time.sleep(args.sleep)

        if batch_original:
            n_pass, n_need, n_reject = process_batch(
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
            )

            total_submitted += len(batch_original)
            total_pass += n_pass
            total_need += n_need
            total_reject += n_reject

    finally:
        for f in output_files.values():
            f.close()

    print("\nDONE")
    print(f"seen rows: {total_seen}")
    print(f"selected this run: {total_selected_this_run}")
    print(f"skipped existing: {total_skipped}")
    print(f"submitted this run: {total_submitted}")
    print(f"LINK_PASS this run: {total_pass}")
    print(f"NEED_CONTENT this run: {total_need}")
    print(f"LINK_REJECT this run: {total_reject}")
    print(f"all csv: {all_csv}")
    print(f"kept csv: {kept_csv}")
    print(f"rejected csv: {rejected_csv}")
    print(f"raw batches: {raw_dir}")


if __name__ == "__main__":
    main()
