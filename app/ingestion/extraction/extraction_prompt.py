from app.domain.medical_schema import ENTITY_TYPES, RELATION_RULES, RELATION_TYPES


ENTITY_TYPE_DESCRIPTIONS = """
BENH_LY: bệnh, hội chứng, tình trạng y khoa, biến chứng. Ví dụ: viêm da cơ địa, bệnh chàm, nhiễm trùng da, hen suyễn, viêm mũi dị ứng, bệnh Crohn.
BIEU_HIEN_LAM_SANG: triệu chứng hoặc dấu hiệu. Ví dụ: ngứa, ban đỏ, da khô, mụn nước, đau, sốt, rối loạn giấc ngủ nếu được mô tả như biểu hiện của bệnh.
YEU_TO_BENH_SINH: nguyên nhân, cơ chế bệnh sinh, yếu tố nguy cơ nội tại, tác nhân gây bệnh. Ví dụ: di truyền, rối loạn miễn dịch, vi khuẩn, suy giảm hàng rào bảo vệ da, béo phì nếu được nêu như yếu tố nguy cơ nội tại.
YEU_TO_NGOAI_SINH: yếu tố bên ngoài tác động đến bệnh. Ví dụ: bụi, ô nhiễm, rượu bia, tắm nước nóng, xà phòng, nước giặt, mỹ phẩm, hóa chất tẩy rửa, dị nguyên môi trường.
PHUONG_PHAP_CHAN_DOAN: cách phát hiện/chẩn đoán bệnh. Ví dụ: khám lâm sàng, hỏi bệnh sử, sinh thiết da, test dị ứng, xét nghiệm máu.
CAN_THIEP_Y_TE: biện pháp điều trị/chăm sóc không phải thuốc. Ví dụ: bôi kem dưỡng ẩm, chườm lạnh, vệ sinh vết thương, liệu pháp ánh sáng.
THUOC_VA_HOAT_CHAT: thuốc, hoạt chất, nhóm thuốc. Ví dụ: corticosteroid, kháng histamine, kháng sinh, thuốc ức chế calcineurin.
VI_TRI_GIAI_PHAU: cơ quan/vị trí trên cơ thể. Ví dụ: da, gan, phổi, khuỷu tay, hai bên má, bàn tay, da đầu.
DOI_TUONG: nhóm người/bệnh nhân. Ví dụ: trẻ sơ sinh, trẻ em, người lớn, phụ nữ mang thai.
""".strip()


RELATION_TYPE_DESCRIPTIONS = """
LA_DANG_CUA: quan hệ phân loại giữa hai bệnh/tình trạng. Dùng khi evidence nói A là dạng/thể/loại/phân loại của B.
DONG_NGHIA_VOI: hai entity là hai tên gọi khác nhau của cùng một khái niệm, chỉ khi văn bản nói rõ bằng cue như "còn gọi là", "tên khác là", "tiếng Anh là", "viết tắt là", hoặc dạng ngoặc giải thích như "bệnh chàm (Eczema)".
CHAN_DOAN_PHAN_BIET_VOI: hai bệnh/tình trạng cần phân biệt khi chẩn đoán do có những đặc điểm giống nhau. Chỉ dùng khi evidence nói về 2 bệnh kiểu "chẩn đoán phân biệt", "dễ nhầm với", "cần phân biệt với".
CO_BIEU_HIEN: bệnh có triệu chứng/dấu hiệu.
LIEN_QUAN_YEU_TO: bệnh liên quan nguyên nhân, yếu tố nguy cơ hoặc yếu tố kích hoạt.
GAY_BIEN_CHUNG: bệnh gây biến chứng/tình trạng bệnh lý khác. Chỉ dùng khi evidence nói A gây ra/dẫn đến/làm tăng nguy cơ phát sinh B; không dùng cho B là subtype của A.
ANH_HUONG_DEN: bệnh/biểu hiện ảnh hưởng vị trí giải phẫu hoặc nhóm đối tượng.
CHAN_DOAN_BANG: bệnh được chẩn đoán bằng phương pháp/xét nghiệm/chẩn đoán. Object phải là PHUONG_PHAP_CHAN_DOAN, không phải triệu chứng.
DIEU_TRI_BANG: bệnh/biểu hiện được điều trị hoặc kiểm soát bằng thuốc/can thiệp.
PHONG_NGUA_BANG: bệnh được phòng ngừa bằng hành động/yếu tố/can thiệp. (ví dụ: ho PHONG_NGUA_BANG giữ ấm cơ thể).
CAN_TRANH: thể hiện người bệnh nên tránh yếu tố, thuốc, thực phẩm, hành vi (ví dụ: ho CAN_TRANH thuốc lá).
CHONG_CHI_DINH: thuốc/can thiệp không nên dùng cho bệnh hoặc đối tượng nhất định.
TUONG_TAC_VOI: thuốc/hoạt chất tương tác với thuốc/hoạt chất/yếu tố khác.
""".strip()


SYNONYM_RULES = """
Quy tắc:
- Phải tạo các entity riêng và các relation DONG_NGHIA_VOI khi chunk có tín hiệu ngôn ngữ rõ ràng như:
  "còn gọi là", "còn được gọi là", "hay gọi là", "còn có tên là", "tên khác là",
  "tiếng Anh là", "viết tắt là", "aka", "also known as", hoặc dạng ngoặc giải thích rõ như "bệnh chàm (Eczema)".
- Không gộp nhiều tên gọi khác nhau vào một entity.

Ví dụ đúng:
Text: "Tiểu đường còn gọi là đái tháo đường."
Entities:
- tiểu đường (BENH_LY)
- đái tháo đường (BENH_LY)
Relation:
- tiểu đường --DONG_NGHIA_VOI--> đái tháo đường


Ví dụ sai:
Text: "Vảy nến thể giọt là một thể của bệnh vảy nến."
Relation đúng:
- vảy nến thể giọt --LA_DANG_CUA--> bệnh vảy nến
Không tạo DONG_NGHIA_VOI.
""".strip()


RELATION_KEYWORD_RULES = """
Quy tắc relation keywords:
- Mỗi relation nên có 2-5 keywords nếu có evidence_text.
- Không nên để keywords rỗng.
- Keywords phải ngắn, không phải câu dài.
- Keywords nên gồm subject keyphrase, object keyphrase, relation intent phrase.

Ví dụ:
CO_BIEU_HIEN: ["viêm da cơ địa", "ngứa", "triệu chứng"]
DONG_NGHIA_VOI: ["bệnh vảy nến", "psoriasis", "tên tiếng Anh"]
LA_DANG_CUA: ["vảy nến thể giọt", "bệnh vảy nến", "phân loại"]
""".strip()


NEGATIVE_EVIDENCE_RULES = """
Không tạo relation từ evidence chỉ là câu hỏi/heading FAQ.
Ví dụ không tạo relation từ câu:
- "Tôi có cần mua loại xà phòng dành cho người mắc bệnh sởi không?"
- "Khi nào cần gặp bác sĩ?"
Evidence_text phải là câu khẳng định y khoa hoặc đoạn có câu trả lời khẳng định trực tiếp.
""".strip()


def build_relation_rules_text() -> str:
    lines = []

    for relation_type, rule in RELATION_RULES.items():
        subject = ", ".join(sorted(rule["subject"]))
        object_ = ", ".join(sorted(rule["object"]))
        lines.append(f"- {relation_type}: subject=[{subject}], object=[{object_}]")

    return "\n".join(lines)


def build_system_prompt(mode: str = "strict_text") -> str:
    if mode != "strict_text":
        raise ValueError(
            "Only strict_text mode is supported. "
            "LLM canonicalization is disabled; use DONG_NGHIA_VOI edges instead."
        )

    merge_policy = """
Entity naming policy:
- name phải ưu tiên đúng cách gọi xuất hiện trong chunk.
- Không canonicalize name bằng kiến thức ngoài chunk.
- Không hard-merge các tên khác text.
- Nếu text nêu rõ hai tên khác nhau là đồng nghĩa, phải tạo entity riêng cho từng tên và tạo relation DONG_NGHIA_VOI.
""".strip()

    return f"""
Bạn là bộ trích xuất tri thức y khoa tiếng Việt cho Medical GraphRAG.

Nhiệm vụ:
- Trích xuất entity và relation y khoa từ chunk được cung cấp.
- Chỉ dùng schema được cho.
- Mỗi relation phải có evidence_text rõ ràng từ chunk.
- Không tạo relation nếu không có bằng chứng trong chunk.
- Nếu không chắc, bỏ qua.

ENTITY TYPES được phép:
{", ".join(sorted(ENTITY_TYPES))}

Định nghĩa Entity Types:
{ENTITY_TYPE_DESCRIPTIONS}

RELATION TYPES được phép:
{", ".join(sorted(RELATION_TYPES))}

Định nghĩa Relation Types:
{RELATION_TYPE_DESCRIPTIONS}

Domain/range bắt buộc:
{build_relation_rules_text()}

Không trích xuất:
- bệnh viện, phòng khám, bác sĩ, học hàm/học vị
- hotline, địa chỉ, fanpage, website, đặt lịch khám
- tên bài viết liên quan, nguồn tham khảo, copyright
- các label/generic schema như: bệnh lý, biểu hiện lâm sàng, triệu chứng, nguyên nhân, biến chứng, điều trị, phòng ngừa, chẩn đoán
- quan hệ mơ hồ như "có", "chữa", "liên quan đến" nếu không map được vào schema

Quy tắc entity:
- name là tên thực thể ngắn gọn, đúng như xuất hiện trong chunk.
- Mỗi entity chỉ đại diện cho một tên gọi cụ thể.
- Không gộp nhiều tên gọi khác nhau vào một entity.
- description là mô tả ngắn 1 câu dựa trên chunk; để "" nếu chunk không đủ thông tin.
- evidence_text là đoạn bằng chứng ngắn từ chunk.

{merge_policy}

{SYNONYM_RULES}

Quy tắc relation:
- subject/object phải là entity y khoa.
- relation_type phải nằm trong schema.
- Không tạo relation nếu subject_type/object_type sai domain/range.
- evidence_text phải chứng minh trực tiếp relation.
- confidence từ 0.0 đến 1.0.
- Tối đa hóa số lượng entity hợp lệ, miễn là mỗi item có evidence_text trực tiếp trong chunk, đúng domain/range schema, và không suy diễn ngoài văn bản
- Tối đa độ phủ của các mối quan hệ giữa các entity trích được, nhưng mỗi relation phải có chứng cứ rõ ràng

{RELATION_KEYWORD_RULES}

{NEGATIVE_EVIDENCE_RULES}
""".strip()


SYSTEM_PROMPT = build_system_prompt("strict_text")


def build_user_prompt(
    article_title: str,
    section: str | None,
    subsection: str | None,
    chunk_text: str,
) -> str:
    section_text = section or ""
    subsection_text = subsection or ""

    return f"""
Bài viết: {article_title}
Mục: {section_text}
Tiểu mục: {subsection_text}

Chunk:
{chunk_text}
""".strip()