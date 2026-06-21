import re
import unicodedata
from enum import Enum
from typing import Any


class EntityType(str, Enum):
    BENH_LY = "BENH_LY"
    BIEU_HIEN_LAM_SANG = "BIEU_HIEN_LAM_SANG"
    YEU_TO_BENH_SINH = "YEU_TO_BENH_SINH"
    YEU_TO_NGOAI_SINH = "YEU_TO_NGOAI_SINH"
    PHUONG_PHAP_CHAN_DOAN = "PHUONG_PHAP_CHAN_DOAN"
    CAN_THIEP_Y_TE = "CAN_THIEP_Y_TE"
    THUOC_VA_HOAT_CHAT = "THUOC_VA_HOAT_CHAT"
    VI_TRI_GIAI_PHAU = "VI_TRI_GIAI_PHAU"
    DOI_TUONG = "DOI_TUONG"


class RelationType(str, Enum):
    LA_DANG_CUA = "LA_DANG_CUA"
    CHAN_DOAN_PHAN_BIET_VOI = "CHAN_DOAN_PHAN_BIET_VOI"
    CO_BIEU_HIEN = "CO_BIEU_HIEN"
    LIEN_QUAN_YEU_TO = "LIEN_QUAN_YEU_TO"
    GAY_BIEN_CHUNG = "GAY_BIEN_CHUNG"
    ANH_HUONG_DEN = "ANH_HUONG_DEN"
    CHAN_DOAN_BANG = "CHAN_DOAN_BANG"
    DIEU_TRI_BANG = "DIEU_TRI_BANG"
    PHONG_NGUA_BANG = "PHONG_NGUA_BANG"
    CAN_TRANH = "CAN_TRANH"
    CHONG_CHI_DINH = "CHONG_CHI_DINH"
    TUONG_TAC_VOI = "TUONG_TAC_VOI"
    DONG_NGHIA_VOI = "DONG_NGHIA_VOI"
    LIEN_QUAN_TRUC_TIEP_DEN = "LIEN_QUAN_TRUC_TIEP_DEN"


ENTITY_TYPES = {item.value for item in EntityType}
RELATION_TYPES = {item.value for item in RelationType}
NON_DISEASE_ENTITY_TYPES = ENTITY_TYPES - {EntityType.BENH_LY.value}


ENTITY_TYPE_DEFINITIONS: dict[str, str] = {
    EntityType.BENH_LY.value: "bệnh, hội chứng, tình trạng y khoa, biến chứng. Ví dụ: nhiễm trùng da, hen suyễn",
    EntityType.BIEU_HIEN_LAM_SANG.value: "triệu chứng hoặc dấu hiệu. Ví dụ: ngứa, da khô, mụn nước, đau, sốt",
    EntityType.YEU_TO_BENH_SINH.value: "nguyên nhân, cơ chế bệnh sinh, yếu tố nguy cơ nội tại, tác nhân gây bệnh. Ví dụ: di truyền, vi khuẩn, béo phì",
    EntityType.YEU_TO_NGOAI_SINH.value: "yếu tố bên ngoài tác động đến bệnh. Ví dụ: bụi, rượu bia, thời tiết, thực phẩm bẩn",
    EntityType.PHUONG_PHAP_CHAN_DOAN.value: "cách phát hiện/chẩn đoán bệnh. Ví dụ: khám lâm sàng, xét nghiệm máu",
    EntityType.CAN_THIEP_Y_TE.value: "biện pháp điều trị/chăm sóc không phải thuốc. Ví dụ: bôi kem dưỡng ẩm, chườm lạnh, ăn nhiều rau",
    EntityType.THUOC_VA_HOAT_CHAT.value: "thuốc, hoạt chất, nhóm thuốc. Ví dụ: corticosteroid, kháng histamine",
    EntityType.VI_TRI_GIAI_PHAU.value: "cơ quan/vị trí trên cơ thể. Ví dụ: da, phổi, khuỷu tay, hai bên má",
    EntityType.DOI_TUONG.value: "nhóm người/bệnh nhân. Ví dụ: trẻ em, phụ nữ mang thai",
}


# Single source of truth for relation schema, prompt text, keyword intent,
# direction normalization and strict validation flags.
#
# Adding a relation type should normally only require editing this dictionary
# plus the RelationType enum above. Prompt/validator/extraction models read from it.
RELATION_SPECS: dict[str, dict[str, Any]] = {
    RelationType.LA_DANG_CUA.value: {
        "description": "Quan hệ phân loại giữa hai bệnh/tình trạng. Dùng khi evidence nói A là dạng/thể/loại/phân loại của B.",
        "subject": {EntityType.BENH_LY.value},
        "object": {EntityType.BENH_LY.value},
        "keyword_intent": "phân loại",
        "normalize_direction": False,
        "prompt_rules": [
            "Dùng khi evidence nói rõ A là dạng/thể/loại/phân loại của B.",
            "Không dùng LA_DANG_CUA cho bệnh gắn vị trí nếu text chỉ nói bệnh xuất hiện tại vị trí đó.",
        ],
        "examples": [
            '"Vảy nến thể giọt là một thể của bệnh vảy nến" => vảy nến thể giọt LA_DANG_CUA bệnh vảy nến',
            '"hắc lào ở mông" không tự động là LA_DANG_CUA của "hắc lào" nếu text chỉ nói vị trí tổn thương.',
        ],
    },
    RelationType.CHAN_DOAN_PHAN_BIET_VOI.value: {
        "description": "Hai bệnh/tình trạng cần phân biệt khi chẩn đoán do có đặc điểm giống nhau. Chỉ dùng khi evidence nói kiểu 'dễ nhầm với', 'cần phân biệt với'.",
        "subject": {EntityType.BENH_LY.value},
        "object": {EntityType.BENH_LY.value},
        "keyword_intent": "chẩn đoán phân biệt",
        "normalize_direction": False,
    },
    RelationType.CO_BIEU_HIEN.value: {
        "description": "Bệnh có triệu chứng/dấu hiệu.",
        "subject": {EntityType.BENH_LY.value},
        "object": {EntityType.BIEU_HIEN_LAM_SANG.value},
        "keyword_intent": "triệu chứng",
        "normalize_direction": True,
    },
    RelationType.LIEN_QUAN_YEU_TO.value: {
        "description": "Bệnh hoặc biểu hiện liên quan nguyên nhân/yếu tố nguy cơ/cơ chế/tác nhân.",
        "subject": {EntityType.BENH_LY.value, EntityType.BIEU_HIEN_LAM_SANG.value},
        "object": {EntityType.YEU_TO_BENH_SINH.value, EntityType.YEU_TO_NGOAI_SINH.value},
        "keyword_intent": "yếu tố liên quan",
        "normalize_direction": True,
    },
    RelationType.GAY_BIEN_CHUNG.value: {
        "description": "Bệnh gây biến chứng/tình trạng bệnh lý khác. Dùng khi evidence nói A gây ra/dẫn đến/làm tăng nguy cơ phát sinh B.",
        "subject": {EntityType.BENH_LY.value},
        "object": {EntityType.BENH_LY.value},
        "keyword_intent": "biến chứng",
        "normalize_direction": False,
    },
    RelationType.ANH_HUONG_DEN.value: {
        "description": "Bệnh/biểu hiện ảnh hưởng vị trí giải phẫu hoặc nhóm đối tượng.",
        "subject": {EntityType.BENH_LY.value, EntityType.BIEU_HIEN_LAM_SANG.value},
        "object": {EntityType.VI_TRI_GIAI_PHAU.value, EntityType.DOI_TUONG.value},
        "keyword_intent": "ảnh hưởng",
        "normalize_direction": True,
    },
    RelationType.CHAN_DOAN_BANG.value: {
        "description": "Bệnh được chẩn đoán bằng phương pháp/xét nghiệm/chẩn đoán. Ưu tiên object là phương pháp chẩn đoán như soi tươi, KOH, đèn Wood; không dùng triệu chứng làm object của relation này.",
        "subject": {EntityType.BENH_LY.value},
        "object": {EntityType.PHUONG_PHAP_CHAN_DOAN.value},
        "keyword_intent": "chẩn đoán",
        "normalize_direction": True,
    },
    RelationType.DIEU_TRI_BANG.value: {
        "description": "Bệnh/biểu hiện được điều trị, kiểm soát hoặc cải thiện bằng thuốc/can thiệp.",
        "subject": {EntityType.BENH_LY.value, EntityType.BIEU_HIEN_LAM_SANG.value},
        "object": {EntityType.CAN_THIEP_Y_TE.value, EntityType.THUOC_VA_HOAT_CHAT.value},
        "keyword_intent": "điều trị",
        "normalize_direction": True,
    },
    RelationType.PHONG_NGUA_BANG.value: {
        "description": "Bệnh/biểu hiện được phòng ngừa hoặc giảm nguy cơ bằng hành động/yếu tố/can thiệp.",
        "subject": {EntityType.BENH_LY.value, EntityType.BIEU_HIEN_LAM_SANG.value},
        "object": {EntityType.CAN_THIEP_Y_TE.value, EntityType.YEU_TO_NGOAI_SINH.value},
        "keyword_intent": "phòng ngừa",
        "normalize_direction": True,
    },
    RelationType.CAN_TRANH.value: {
        "description": "Thể hiện người bệnh/triệu chứng/đối tượng nên tránh yếu tố, thuốc, thực phẩm hoặc hành vi.",
        "subject": {EntityType.BENH_LY.value, EntityType.BIEU_HIEN_LAM_SANG.value, EntityType.DOI_TUONG.value},
        "object": {EntityType.YEU_TO_NGOAI_SINH.value, EntityType.THUOC_VA_HOAT_CHAT.value},
        "keyword_intent": "cần tránh",
        "normalize_direction": True,
    },
    RelationType.CHONG_CHI_DINH.value: {
        "description": "Thuốc/can thiệp không nên dùng cho bệnh hoặc đối tượng nhất định.",
        "subject": {EntityType.THUOC_VA_HOAT_CHAT.value, EntityType.CAN_THIEP_Y_TE.value},
        "object": {EntityType.BENH_LY.value, EntityType.DOI_TUONG.value},
        "keyword_intent": "chống chỉ định",
        "normalize_direction": True,
    },
    RelationType.TUONG_TAC_VOI.value: {
        "description": "Thuốc/hoạt chất tương tác với thuốc/hoạt chất/yếu tố khác.",
        "subject": {EntityType.THUOC_VA_HOAT_CHAT.value},
        "object": {EntityType.THUOC_VA_HOAT_CHAT.value, EntityType.YEU_TO_NGOAI_SINH.value},
        "keyword_intent": "tương tác",
        "normalize_direction": False,
    },
    RelationType.DONG_NGHIA_VOI.value: {
        "description": "Hai entity là hai tên gọi khác nhau của cùng một khái niệm, chỉ khi văn bản nói rõ bằng cue như 'còn gọi là', 'tên khác là', 'tiếng Anh là', 'viết tắt là', hoặc dạng ngoặc giải thích như 'bệnh chàm (Eczema)'.",
        "subject": set(ENTITY_TYPES),
        "object": set(ENTITY_TYPES),
        "keyword_intent": "tên gọi khác",
        "normalize_direction": False,
        "enforce_schema": True,
        "require_same_entity_type": True,
        "prompt_rules": [
            "Chỉ dùng khi chunk có tín hiệu ngôn ngữ rõ ràng như 'còn gọi là', 'tên khác là', 'tiếng Anh là', 'viết tắt là' hoặc ngoặc giải thích tên gọi.",
            "Không dùng DONG_NGHIA_VOI chỉ vì hai cụm từ gần giống nhau hoặc khác trật tự từ.",
            "Không dùng DONG_NGHIA_VOI để nối bệnh gắn vị trí với bệnh gốc nếu text không nói rõ đó là tên gọi tương đương.",
        ],
        "examples": [
            '"Tiểu đường còn gọi là đái tháo đường" => tiểu đường DONG_NGHIA_VOI đái tháo đường',
            '"da vùng mông" và "vùng da mông" không tự động là DONG_NGHIA_VOI nếu text không nói rõ.',
            '"hắc lào ở mông" và "hắc lào" không tự động là DONG_NGHIA_VOI.',
        ],
    },
    RelationType.LIEN_QUAN_TRUC_TIEP_DEN.value: {
        "description": "Quan hệ trực tiếp giữa hai thực thể y khoa không phải BENH_LY, dùng để giữ cạnh chéo khi evidence nối rõ hai yếu tố/biểu hiện/vị trí/thuốc/cơ chế/phương pháp và không có relation_type cụ thể hơn.",
        "subject": set(NON_DISEASE_ENTITY_TYPES),
        "object": set(NON_DISEASE_ENTITY_TYPES),
        "keyword_intent": "liên quan trực tiếp",
        "normalize_direction": False,
        "enforce_schema": True,
        "forbid_benh_ly_endpoint": True,
        "prompt_rules": [
            "Chỉ dùng khi cả subject và object đều không phải BENH_LY.",
            "Chỉ dùng khi evidence_text trực tiếp nối hai thực thể.",
            "Không dùng để thay thế relation cụ thể hơn như DIEU_TRI_BANG, CO_BIEU_HIEN, CHAN_DOAN_BANG, DONG_NGHIA_VOI hoặc LA_DANG_CUA.",
            "Dùng relation này để tạo cạnh chéo giữa triệu chứng, yếu tố, vị trí, thuốc, cơ chế và phương pháp, tránh ép mọi fact về bệnh trong title.",
        ],
        "examples": [
            '"vùng da mông kín và ẩm tạo điều kiện cho nấm phát triển" => vùng da mông LIEN_QUAN_TRUC_TIEP_DEN độ ẩm',
            '"độ ẩm tạo điều kiện cho nấm phát triển" => độ ẩm LIEN_QUAN_TRUC_TIEP_DEN nấm phát triển',
            '"Ketoconazole ức chế tổng hợp ergosterol ở màng tế bào nấm" => Ketoconazole LIEN_QUAN_TRUC_TIEP_DEN ergosterol',
            '"ergosterol cấu tạo màng tế bào nấm" => ergosterol LIEN_QUAN_TRUC_TIEP_DEN màng tế bào nấm',
        ],
    },
}


RELATION_RULES: dict[str, dict[str, set[str]]] = {
    name: {
        "subject": set(spec["subject"]),
        "object": set(spec["object"]),
    }
    for name, spec in RELATION_SPECS.items()
}


BLOCKED_ENTITY_HINTS = {
    "bệnh viện",
    "phòng khám",
    "hệ thống y tế",
    "vinmec",
    "medlatec",
    "tâm anh",
    "tam anh",
    "hotline",
    "tổng đài",
    "fanpage",
    "website",
    "app",
    "myvinmec",
    "medon",
    "đặt lịch",
    "gói khám",
    "dịch vụ xét nghiệm tại nhà",
    "lấy mẫu xét nghiệm tại nhà",
    "bác sĩ",
    "thạc sĩ",
    "tiến sĩ",
    "ts.bs",
    "bs.",
    "địa chỉ",
    "copyright",
    # Schema/generic labels, không phải entity y khoa cụ thể.
    "doi_tuong",
    "benh_ly",
    "bieu_hien_lam_sang",
    "dấu hiệu",
    "triệu chứng",
    "nguyên nhân",
    "biến chứng",
    "điều trị",
    "phòng ngừa",
    "chẩn đoán",
}


def _normalize_for_match(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


_NORMALIZED_BLOCKED_ENTITY_HINTS = {
    _normalize_for_match(hint)
    for hint in BLOCKED_ENTITY_HINTS
}


def _ordered_entity_types(items: set[str]) -> list[str]:
    return [item.value for item in EntityType if item.value in items]


def _ordered_relation_types(items: set[str] | None = None) -> list[str]:
    allowed = items if items is not None else RELATION_TYPES
    return [item.value for item in RelationType if item.value in allowed]


def _format_type_set(items: set[str]) -> str:
    return ", ".join(_ordered_entity_types(items))


def build_entity_type_descriptions() -> str:
    return "\n".join(
        f"{entity_type}: {ENTITY_TYPE_DEFINITIONS[entity_type]}"
        for entity_type in _ordered_entity_types(ENTITY_TYPES)
    )


def build_relation_type_descriptions() -> str:
    lines: list[str] = []
    for relation_type in _ordered_relation_types():
        spec = RELATION_SPECS[relation_type]
        lines.append(f"{relation_type}: {spec['description']}")
    return "\n".join(lines)


def build_relation_rules_text() -> str:
    """Return schema direction hints, not a hard ontology contract.

    The extraction prompt should use this as a guide for choosing the most
    natural relation direction. Hard validation is controlled separately by
    RELATION_SPECS flags such as enforce_schema/forbid_benh_ly_endpoint.
    """
    lines: list[str] = []
    for relation_type in _ordered_relation_types():
        rule = RELATION_RULES[relation_type]
        lines.append(
            f"- {relation_type}: subject gợi ý=[{_format_type_set(rule['subject'])}], "
            f"object gợi ý=[{_format_type_set(rule['object'])}]"
        )
    return "\n".join(lines)


def build_relation_prompt_guidance_text() -> str:
    """Build old-style relation instructions from RELATION_SPECS.

    Earlier prompts worked better because relation guidance was written as
    natural extraction instructions instead of compact ontology tables. This
    keeps the schema as the single source of truth while rendering it in that
    old, LLM-friendly style.
    """
    blocks: list[str] = []

    for relation_type in _ordered_relation_types():
        spec = RELATION_SPECS[relation_type]
        rules = list(spec.get("prompt_rules") or [])
        examples = list(spec.get("examples") or [])
        rule = RELATION_RULES[relation_type]

        lines = [f"{relation_type}:"]
        lines.append(f"- Ý nghĩa: {spec['description']}")
        lines.append(
            "- Hướng gợi ý: "
            f"subject thường là [{_format_type_set(rule['subject'])}], "
            f"object thường là [{_format_type_set(rule['object'])}]."
        )
        lines.append(
            "- Đây là hướng dẫn chọn relation_type/direction, không phải lý do để bỏ một fact rõ ràng. "
            "Nếu evidence trực tiếp và relation có ích cho retrieval, hãy chọn relation_type gần nhất rồi giữ evidence_text sát câu gốc."
        )

        for rule_text in rules:
            lines.append(f"- {rule_text}")

        if examples:
            lines.append("Ví dụ:")
            for example in examples:
                lines.append(f"- {example}")

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks).strip()


def get_relation_spec(relation_type: str) -> dict[str, Any]:
    return RELATION_SPECS.get(str(relation_type or "").strip(), {})


def get_relation_keyword_intent(relation_type: str, default: str | None = None) -> str:
    spec = get_relation_spec(relation_type)
    if spec.get("keyword_intent"):
        return str(spec["keyword_intent"])
    return default if default is not None else str(relation_type or "").lower().replace("_", " ").strip()


def get_all_relation_keyword_intents() -> set[str]:
    return {
        str(spec.get("keyword_intent"))
        for spec in RELATION_SPECS.values()
        if spec.get("keyword_intent")
    }


def is_valid_entity_type(entity_type: str) -> bool:
    return entity_type in ENTITY_TYPES


def is_valid_relation_type(relation_type: str) -> bool:
    return relation_type in RELATION_TYPES


def is_valid_relation_schema(
    relation_type: str,
    subject_type: str,
    object_type: str,
) -> bool:
    if relation_type not in RELATION_RULES:
        return False

    rule = RELATION_RULES[relation_type]
    return subject_type in rule["subject"] and object_type in rule["object"]


def is_valid_relation_by_schema_policy(
    relation_type: str,
    subject_type: str,
    object_type: str,
) -> bool:
    """Validate relation-type-specific invariant flags from RELATION_SPECS.

    Existing project philosophy keeps recall high and avoids broad semantic
    filtering. Therefore most legacy relations do not enforce domain/range here.
    New or risky relation types can opt into strict checks with `enforce_schema`.
    """
    if not is_valid_relation_type(relation_type):
        return False

    spec = get_relation_spec(relation_type)

    if spec.get("require_same_entity_type") and subject_type != object_type:
        return False

    if spec.get("forbid_benh_ly_endpoint"):
        if EntityType.BENH_LY.value in {subject_type, object_type}:
            return False

    if spec.get("enforce_schema") and not is_valid_relation_schema(
        relation_type,
        subject_type,
        object_type,
    ):
        return False

    return True


def should_flip_relation_direction(
    relation_type: str,
    subject_type: str,
    object_type: str,
) -> bool:
    """Return True when a relation is clearly reversed by schema direction.

    Direction policy is schema-driven: a relation only flips if its spec opts in
    with `normalize_direction=True`, and the current subject type belongs to the
    canonical object range while the current object type belongs to the canonical
    subject range.
    """
    spec = get_relation_spec(relation_type)
    if not spec.get("normalize_direction"):
        return False

    rule = RELATION_RULES.get(relation_type)
    if not rule:
        return False

    subject_types = rule["subject"]
    object_types = rule["object"]

    if subject_type in subject_types and object_type in object_types:
        return False

    return subject_type in object_types and object_type in subject_types


def looks_like_blocked_entity(name: str) -> bool:
    lowered = name.strip().lower()
    normalized = _normalize_for_match(name)

    return any(
        hint in lowered or hint in normalized
        for hint in _NORMALIZED_BLOCKED_ENTITY_HINTS
    )
