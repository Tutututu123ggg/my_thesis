import re
import unicodedata
from enum import Enum


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


ENTITY_TYPES = {item.value for item in EntityType}
RELATION_TYPES = {item.value for item in RelationType}


RELATION_RULES: dict[str, dict[str, set[str]]] = {
    RelationType.LA_DANG_CUA.value: {
        "subject": {EntityType.BENH_LY.value},
        "object": {EntityType.BENH_LY.value},
    },
    RelationType.CHAN_DOAN_PHAN_BIET_VOI.value: {
        "subject": {EntityType.BENH_LY.value},
        "object": {EntityType.BENH_LY.value},
    },
    RelationType.CO_BIEU_HIEN.value: {
        "subject": {EntityType.BENH_LY.value},
        "object": {EntityType.BIEU_HIEN_LAM_SANG.value},
    },
    RelationType.LIEN_QUAN_YEU_TO.value: {
        "subject": {EntityType.BENH_LY.value},
        "object": {
            EntityType.YEU_TO_BENH_SINH.value,
            EntityType.YEU_TO_NGOAI_SINH.value,
        },
    },
    RelationType.GAY_BIEN_CHUNG.value: {
        "subject": {EntityType.BENH_LY.value},
        "object": {EntityType.BENH_LY.value},
    },
    RelationType.ANH_HUONG_DEN.value: {
        "subject": {
            EntityType.BENH_LY.value,
            EntityType.BIEU_HIEN_LAM_SANG.value,
        },
        "object": {
            EntityType.VI_TRI_GIAI_PHAU.value,
            EntityType.DOI_TUONG.value,
        },
    },
    RelationType.CHAN_DOAN_BANG.value: {
        "subject": {EntityType.BENH_LY.value},
        "object": {
            EntityType.PHUONG_PHAP_CHAN_DOAN.value,
            EntityType.BIEU_HIEN_LAM_SANG.value,
        },
    },
    RelationType.DIEU_TRI_BANG.value: {
        "subject": {
            EntityType.BENH_LY.value,
            EntityType.BIEU_HIEN_LAM_SANG.value,
        },
        "object": {
            EntityType.CAN_THIEP_Y_TE.value,
            EntityType.THUOC_VA_HOAT_CHAT.value,
        },
    },
    RelationType.PHONG_NGUA_BANG.value: {
        "subject": {EntityType.BENH_LY.value},
        "object": {
            EntityType.CAN_THIEP_Y_TE.value,
            EntityType.YEU_TO_NGOAI_SINH.value,
        },
    },
    RelationType.CAN_TRANH.value: {
        "subject": {
            EntityType.BENH_LY.value,
            EntityType.DOI_TUONG.value,
        },
        "object": {
            EntityType.YEU_TO_NGOAI_SINH.value,
            EntityType.THUOC_VA_HOAT_CHAT.value,
        },
    },
    RelationType.CHONG_CHI_DINH.value: {
        "subject": {
            EntityType.THUOC_VA_HOAT_CHAT.value,
            EntityType.CAN_THIEP_Y_TE.value,
        },
        "object": {
            EntityType.BENH_LY.value,
            EntityType.DOI_TUONG.value,
        },
    },
    RelationType.TUONG_TAC_VOI.value: {
        "subject": {EntityType.THUOC_VA_HOAT_CHAT.value},
        "object": {
            EntityType.THUOC_VA_HOAT_CHAT.value,
            EntityType.YEU_TO_NGOAI_SINH.value,
        },
    },
    # Validator sẽ enforce thêm: subject_type == object_type.
    RelationType.DONG_NGHIA_VOI.value: {
        "subject": set(ENTITY_TYPES),
        "object": set(ENTITY_TYPES),
    },
}


BLOCKED_ENTITY_HINTS = {
    "bệnh viện",
    "phòng khám",
    "hotline",
    "fanpage",
    "website",
    "đặt lịch",
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


def looks_like_blocked_entity(name: str) -> bool:
    lowered = name.strip().lower()
    normalized = _normalize_for_match(name)

    return any(
        hint in lowered or hint in normalized
        for hint in _NORMALIZED_BLOCKED_ENTITY_HINTS
    )
