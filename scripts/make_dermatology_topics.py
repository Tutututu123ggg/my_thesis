from __future__ import annotations

import json
from pathlib import Path

OUT = Path("data/crawl/topic_seed_dermatology_common_v1.json")

CLUSTERS = {
    "viem_da_cham": [
        ("viêm da cơ địa", ["bệnh chàm", "chàm thể tạng", "eczema", "atopic dermatitis"]),
        ("chàm sữa", ["lác sữa", "eczema ở trẻ sơ sinh"]),
        ("viêm da tiếp xúc", ["dị ứng tiếp xúc", "contact dermatitis"]),
        ("viêm da tiết bã", ["viêm da dầu", "seborrheic dermatitis"]),
        ("viêm da dị ứng", ["dị ứng da", "viêm da do dị ứng"]),
        ("khô da", ["da khô", "nứt nẻ da"]),
        ("hăm tã", ["viêm da tã lót"]),
        ("rôm sảy", ["phát ban nhiệt"]),
    ],
    "mun_nang_long": [
        ("mụn trứng cá", ["trứng cá", "acne", "acne vulgaris"]),
        ("mụn đầu đen", ["blackheads"]),
        ("mụn bọc", ["mụn viêm", "mụn mủ"]),
        ("mụn nhọt", ["nhọt da", "áp xe da"]),
        ("viêm nang lông", ["folliculitis"]),
        ("mụn cóc", ["mụn cơm", "wart"]),
    ],
    "nam_da": [
        ("nấm da", ["nhiễm nấm da", "nấm ngoài da"]),
        ("hắc lào", ["lác đồng tiền", "nấm da thân", "ringworm", "tinea corporis"]),
        ("lang ben", ["nấm lang ben", "tinea versicolor", "pityriasis versicolor"]),
        ("nấm móng", ["onychomycosis"]),
        ("nấm da đầu", ["tinea capitis"]),
    ],
    "di_ung_may_day": [
        ("mày đay", ["mề đay", "nổi mề đay", "urticaria", "hives"]),
        ("dị ứng da", ["phát ban dị ứng", "nổi mẩn dị ứng"]),
        ("dị ứng thời tiết", ["nổi mẩn do thời tiết"]),
        ("dị ứng thuốc", ["phát ban do thuốc", "dị ứng thuốc biểu hiện da"]),
        ("phù mạch", ["angioedema"]),
        ("ngứa da", ["ngứa ngoài da", "ngứa toàn thân"]),
        ("nổi mẩn đỏ", ["nổi mẩn", "ban đỏ"]),
        ("phát ban", ["ban da", "rash"]),
    ],
    "vay_sung_sac_to": [
        ("vảy nến", ["psoriasis", "bệnh vảy nến"]),
        ("á sừng", ["viêm da cơ địa á sừng"]),
        ("dày sừng nang lông", ["keratosis pilaris"]),
        ("bạch biến", ["vitiligo"]),
        ("nám da", ["melasma"]),
        ("tăng sắc tố da", ["thâm da", "thâm mụn"]),
    ],
    "virus_ban_da": [
        ("zona thần kinh", ["giời leo", "herpes zoster"]),
        ("thủy đậu", ["varicella", "chickenpox"]),
        ("tay chân miệng", ["bệnh tay chân miệng"]),
        ("sởi", ["bệnh sởi", "measles"]),
        ("herpes", ["mụn rộp", "herpes môi", "mụn rộp sinh dục"]),
        ("u mềm lây", ["molluscum contagiosum"]),
    ],
    "toc_mong_da_dau": [
        ("rụng tóc", ["alopecia", "rụng tóc nhiều"]),
        ("gàu", ["dandruff"]),
        ("nấm da đầu", ["tinea capitis"]),
        ("nấm móng", ["onychomycosis"]),
    ],
}


def main() -> None:
    data = {"clusters": []}

    for cluster_name, items in CLUSTERS.items():
        data["clusters"].append({
            "cluster": cluster_name,
            "organ_system": "da_lieu_mo_rong",
            "anchors": [
                {
                    "anchor_entity": name,
                    "aliases": aliases,
                }
                for name, aliases in items
            ],
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    total = sum(len(c["anchors"]) for c in data["clusters"])
    print(f"Wrote {OUT}")
    print(f"clusters={len(data['clusters'])}, anchors={total}")


if __name__ == "__main__":
    main()