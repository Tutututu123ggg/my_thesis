from __future__ import annotations

import json
from typing import Any

from app.domain.medical_schema import (
    RELATION_TYPES,
    build_relation_prompt_guidance_text,
    build_relation_rules_text,
    build_relation_type_descriptions,
)


RELATION_RECALL_GUIDE = """
Mục tiêu relation:
- Tạo các quan hệ y khoa/sức khỏe có evidence rõ ràng, hữu ích cho truy hồi.
- Dùng input entity list làm endpoint.
- Dùng title/heading của các đề mục cha để khôi phục subject ẩn của claim trong list/bullet.
- Nếu fact rõ nhưng relation_type chưa hoàn hảo, chọn relation_type gần nhất trong schema và giữ edge.
- Khi nhiều endpoint có cùng subject, cùng relation_type và cùng evidence/list/heading, ưu tiên relation_groups dạng 1-n.
- Khi nhiều subject cùng liên quan tới một object theo cùng relation_type, ưu tiên relation_groups dạng n-1.
""".strip()


COVERAGE_CHECKLIST = """
Coverage checklist bắt buộc sau khi draft relation để đảm bảo đã lấy toàn bộ các quan hệ phù hợp với schema giữa các entity đã được xác định trong đoạn văn bản:

1. Với mỗi entity thuộc THUOC_VA_HOAT_CHAT, CAN_THIEP_Y_TE, PHUONG_PHAP_CHAN_DOAN, YEU_TO_NGOAI_SINH:
   nếu text/heading nói rõ nó dùng để điều trị, giảm triệu chứng, phòng ngừa, tránh nguy cơ hoặc chẩn đoán,
   phải tạo ít nhất một relation nối nó với bệnh/triệu chứng tương ứng.

2. Với mỗi bullet/câu có tri thức y khoa thuộc schema và có evidence:
cố tạo ít nhất một relation nếu có thể xác định hai endpoint trong entity list hoặc trong title/heading.

3. Nếu biện pháp/thuốc cải thiện triệu chứng cụ thể:
   tạo relation "triệu chứng --DIEU_TRI_BANG--> biện pháp/thuốc".
   Nếu heading/title cũng nói đang điều trị bệnh chính, tạo thêm "bệnh --DIEU_TRI_BANG--> biện pháp/thuốc".

4. Nếu có quan hệ trực tiếp giữa hai entity không phải BENH_LY và không có relation_type cụ thể hơn:
   dùng LIEN_QUAN_TRUC_TIEP_DEN để giữ cạnh chéo.

5. evidence chỉ nói một thuốc/biện pháp được dùng sai, dùng nhầm, không hiệu quả hoặc làm nặng hơn, không tạo DIEU_TRI_BANG,
   dùng CAN_TRANH nếu text có ý khuyến cáo không dùng, không làm, hạn chế hoặc tránh.

6. Không bỏ relation chỉ vì subject không lặp lại trong câu.
   Được dùng bệnh/tình trạng trong title/heading làm subject nếu bullet/list đang nằm dưới heading điều trị/phòng ngừa/xử lý/cần tránh của bệnh đó.

7. Sau khi sinh relation, kiểm tra lại các entity dạng thuốc/can thiệp/chẩn đoán/chăm sóc/phòng ngừa.
   Nếu còn entity quan trọng không xuất hiện ở bất kỳ endpoint nào mà text nêu vai trò rõ, hãy thêm relation phù hợp.

8. Với mọi cấp của list/bullet:
   - Dùng title/heading/bullet cha để xác định subject và intent nếu subject bị ẩn.
   - Nếu bullet cha nêu category và bullet/câu có các item cụ thể, hãy tạo relation cho cả category nếu hữu ích và relation riêng cho từng item cụ thể quan trọng.
   - Không dừng lại sau item đầu hoặc sau category-level relation.
""".strip()


RELATION_TYPE_HINTS = """
Gợi ý chọn relation_type gần nhất:
- điều trị, giảm triệu chứng, phục hồi, chăm sóc -> DIEU_TRI_BANG
- phòng ngừa, giảm nguy cơ, tránh tái phát/lây lan -> PHONG_NGUA_BANG
- nguyên nhân, tác nhân, kích hoạt, độc tố, dị nguyên, kích ứng, làm nặng -> LIEN_QUAN_YEU_TO
- triệu chứng/dấu hiệu -> CO_BIEU_HIEN
- xét nghiệm/thăm khám/chẩn đoán -> CHAN_DOAN_BANG
- cảnh báo tránh rõ ràng -> CAN_TRANH
- quan hệ có evidence nhưng không khớp loại nào hơn -> LIEN_QUAN_TRUC_TIEP_DEN
""".strip()


RELATION_GROUP_GUIDE = """
Cách dùng relation_groups:
- Dùng relation_groups cho quan hệ 1-n hoặc n-1 khi các cạnh có cùng relation_type và cùng evidence/list/heading.
- 1-n: một subject nối tới nhiều object. Ví dụ một bệnh có nhiều biện pháp điều trị/phòng ngừa trong cùng bullet/list.
- n-1: nhiều subject nối tới một object. Ví dụ nhiều triệu chứng cùng được cải thiện bởi một thuốc/biện pháp.
- Với mỗi group, một phía có thể là danh sách nhiều endpoint; phía còn lại nên giữ một endpoint để tránh many-vs-many mơ hồ.
- evidence_text của group có thể là câu/list/heading ngắn hỗ trợ toàn bộ group.
- Dùng relations[] cho fact 1-1 riêng lẻ; dùng relation_groups[] cho list rõ ràng để không bỏ sót item cuối và giảm output token.
- Với nested list, relation_groups nên mở rộng các item cụ thể ở mọi cấp list khi chúng đã có trong entity list và cùng chia sẻ subject hoặc relation_type hoặc evidence hoặc heading.
""".strip()


RELATION_FEW_SHOTS = """
Ví dụ 1 — chăm sóc tại nhà, 1-n
Đề mục: "Cách trị viêm da dị ứng tại nhà"
Text: "Chườm lạnh hoặc tắm nước mát giúp làm dịu cơn ngứa, giảm tình trạng mề đay và nổi đỏ trên da."
Entity list:
- viêm da dị ứng [BENH_LY]
- chườm lạnh [CAN_THIEP_Y_TE]
- tắm nước mát [CAN_THIEP_Y_TE]
- ngứa [BIEU_HIEN_LAM_SANG]
- mề đay [BIEU_HIEN_LAM_SANG]
- nổi đỏ trên da [BIEU_HIEN_LAM_SANG]
Relation group nên có:
- subjects: [viêm da dị ứng [BENH_LY]]
  relation_type: DIEU_TRI_BANG
  objects: [chườm lạnh [CAN_THIEP_Y_TE], tắm nước mát [CAN_THIEP_Y_TE]]
  evidence_text: "Chườm lạnh hoặc tắm nước mát giúp làm dịu cơn ngứa, giảm tình trạng mề đay và nổi đỏ trên da."
Relation 1-1 có thể thêm nếu evidence rõ:
- ngứa [BIEU_HIEN_LAM_SANG] --DIEU_TRI_BANG--> chườm lạnh [CAN_THIEP_Y_TE]

Ví dụ 2 — nguyên liệu/chăm sóc da, 1-n theo intent khác nhau
Đề mục: "Cách trị viêm da dị ứng tại nhà"
Text: "Nha đam có hiệu quả dưỡng ẩm, ức chế vi khuẩn, sát trùng da và giúp ngăn ngừa bội nhiễm khi bị dị ứng."
Entity list:
- viêm da dị ứng [BENH_LY]
- nha đam [YEU_TO_NGOAI_SINH]
- bội nhiễm [BENH_LY]
Relation nên có:
- viêm da dị ứng [BENH_LY] --DIEU_TRI_BANG--> nha đam [YEU_TO_NGOAI_SINH]
- viêm da dị ứng [BENH_LY] --PHONG_NGUA_BANG--> nha đam [YEU_TO_NGOAI_SINH]

Ví dụ 3 — phòng ngừa theo list, 1-n
Đề mục: "Gợi ý các biện pháp tránh viêm da tiếp xúc do kiến ba khoang"
Text: "Dùng lưới chống côn trùng, ngủ trong mùng và kiểm tra chăn gối, quần áo trước khi sử dụng."
Entity list:
- viêm da tiếp xúc do kiến ba khoang [BENH_LY]
- lưới chống côn trùng [CAN_THIEP_Y_TE]
- ngủ trong mùng [CAN_THIEP_Y_TE]
- kiểm tra chăn gối, quần áo [CAN_THIEP_Y_TE]
Relation group nên có:
- subjects: [viêm da tiếp xúc do kiến ba khoang [BENH_LY]]
  relation_type: PHONG_NGUA_BANG
  objects: [lưới chống côn trùng [CAN_THIEP_Y_TE], ngủ trong mùng [CAN_THIEP_Y_TE], kiểm tra chăn gối, quần áo [CAN_THIEP_Y_TE]]
  evidence_text: "Dùng lưới chống côn trùng, ngủ trong mùng và kiểm tra chăn gối, quần áo trước khi sử dụng."


Ví dụ 4 — nhiều triệu chứng cùng được làm dịu, n-1
Text: "Chườm lạnh giúp giảm ngứa, mề đay và nổi đỏ trên da."
Entity list:
- ngứa [BIEU_HIEN_LAM_SANG]
- mề đay [BIEU_HIEN_LAM_SANG]
- nổi đỏ trên da [BIEU_HIEN_LAM_SANG]
- chườm lạnh [CAN_THIEP_Y_TE]
Relation group nên có:
- subjects: [ngứa [BIEU_HIEN_LAM_SANG], mề đay [BIEU_HIEN_LAM_SANG], nổi đỏ trên da [BIEU_HIEN_LAM_SANG]]
  relation_type: DIEU_TRI_BANG
  objects: [chườm lạnh [CAN_THIEP_Y_TE]]
  evidence_text: "Chườm lạnh giúp giảm ngứa, mề đay và nổi đỏ trên da."
""".strip()


def build_relation_system_prompt(mode: str = "strict_text") -> str:
    if mode != "strict_text":
        raise ValueError("Only strict_text mode is supported.")

    return f"""
Bạn là bộ trích xuất relation y khoa tiếng Việt cho Medical GraphRAG.

Nhiệm vụ:
- Đọc title/heading/chunk cùng input entity list.
- Tạo relation y khoa hoặc sức khỏe có evidence rõ ràng.
- Ưu tiên endpoint đúng và evidence đúng; relation_type chọn gần nhất theo schema.
- Mỗi relation cần có evidence_text ngắn (có thể lấy ở nhiều câu) để kiểm chứng.

RELATION TYPES được phép:
{", ".join(sorted(RELATION_TYPES))}

Định nghĩa Relation Types:
{build_relation_type_descriptions()}

Gợi ý subject/object theo schema:
{build_relation_rules_text()}

Hướng dẫn chi tiết theo schema:
{build_relation_prompt_guidance_text()}

{RELATION_RECALL_GUIDE}

{COVERAGE_CHECKLIST}

{RELATION_TYPE_HINTS}

{RELATION_GROUP_GUIDE}

Few-shot:
{RELATION_FEW_SHOTS}

Output theo schema LLMRelationExtractionResult, gồm relations[] cho quan hệ 1-1 và relation_groups[] cho quan hệ 1-n/n-1.
""".strip()


def _compact_json(data: Any) -> str:
    return json.dumps(data or [], ensure_ascii=False, indent=2)


def build_relation_user_prompt(
    markdown_text: str,
    entities: list[dict[str, Any]] | None = None,
    explicit_synonym_pairs: list[dict[str, Any]] | None = None,
) -> str:
    return f"""
Nội dung markdown cần tạo relation:
{markdown_text}

Input entity list:
```json
{_compact_json(entities)}
```

Explicit synonym pairs nếu có:
```json
{_compact_json(explicit_synonym_pairs)}
```
""".strip()


# Backward-compatible aliases for older imports.
def build_system_prompt(mode: str = "strict_text") -> str:
    return build_relation_system_prompt(mode)


def build_user_prompt(
    markdown_text: str,
    entities: list[dict[str, Any]] | None = None,
    explicit_synonym_pairs: list[dict[str, Any]] | None = None,
) -> str:
    return build_relation_user_prompt(markdown_text, entities, explicit_synonym_pairs)


SYSTEM_PROMPT = build_relation_system_prompt("strict_text")
