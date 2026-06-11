import hashlib
import re
import unicodedata


def normalize_text(text: str) -> str:
    """
    Chuẩn hóa text để dùng cho entity matching.

    Ví dụ:
    "Viêm da cơ địa" -> "viem da co dia"
    "Atopic Dermatitis" -> "atopic dermatitis"
    """
    text = text.strip().lower()

    # Bỏ Markdown emphasis nếu LLM lỡ giữ trong tên.
    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.replace("*", "")
    text = text.replace("_", "")

    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")

    text = re.sub(r"[“”\"'`]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" -–—:;,.()[]{}")

    return text.strip()


def clean_surface_text(text: str) -> str:
    """
    Clean nhẹ để lưu lại aliases/surface_forms.

    Khác normalize_text:
    - Không bỏ dấu tiếng Việt.
    - Vì đây là text để hiển thị/lưu lại.
    """
    text = text.strip()
    text = text.replace("**", "")
    text = text.replace("__", "")
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" -–—:;,.()[]{}")
    return text.strip()


def make_stable_id(prefix: str, raw: str) -> str:
    """
    Tạo ID ổn định từ nội dung raw.
    Cùng raw thì luôn ra cùng ID.
    """
    h = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{h}"


def make_entity_id(name: str, entity_type: str) -> str:
    """
    Tạo entity_id theo canonical name + entity_type.

    Đây là nền của hard merge:
    cùng một canonical entity thì luôn cùng entity_id.
    """
    normalized = normalize_text(name)
    return make_stable_id("entity", f"{entity_type}:{normalized}")


def make_relation_id(
    subject_entity_id: str,
    relation_type: str,
    object_entity_id: str,
) -> str:
    """
    Tạo relation_id ổn định theo:
    subject canonical ID + relation_type + object canonical ID.

    Nếu cùng một fact xuất hiện ở nhiều chunk, nó sẽ merge vào cùng edge.
    """
    raw = f"{subject_entity_id}:{relation_type}:{object_entity_id}"
    return make_stable_id("relation", raw)