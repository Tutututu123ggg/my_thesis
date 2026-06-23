from app.domain.medical_schema import (
    ENTITY_TYPES,
    RELATION_TYPES,
    build_entity_type_descriptions,
    build_relation_prompt_guidance_text,
    build_relation_rules_text,
    build_relation_type_descriptions,
)


RELATION_KEYWORD_RULES = """
relation.keywords là metadata phụ, KHÔNG phải nhiệm vụ chính của extractor:
- Extractor có thể để relation.keywords = [] nếu không chắc.
- Nếu sinh, chỉ dùng cụm ngắn tối đa 5 từ, ví dụ: "thuốc chống nấm", "triệu chứng", "nguy cơ viêm phổi".
""".strip()


RELATION_COVERAGE_RULES = """
Luồng trích xuất:

1. Phân loại nhanh chunk; nếu không có tri thức y khoa thuộc schema thì bỏ qua.
2. Đọc evidence trước và tạo relation giữa các thực thể có quan hệ có thể được hiểu từ văn bản.
3. Dùng title/heading để hiểu ngữ cảnh, nhưng vẫn cần chú trọng quan hệ cụ thể trong evidence.
4. Hãy cố trích xuất đầy đủ và trung thực. Ví dụ: Nếu thuốc được ghi là cải thiện một triệu chứng cụ thể, hãy nối với entity triệu chứng chứ không phải bệnh. Nếu cũng có evidence cho thấy thuốc được dùng cho bệnh, có thể tạo THÊM relation với bệnh.
7. Khi evidence có quan hệ trực tiếp giữa hai thực thể mà không hợp relation_type nào trong schema, dùng LIEN_QUAN_TRUC_TIEP_DEN để biểu thị mối liên quan "khác".
8. Relation có thể là 1-1 hoặc 1-vs-many:
   - A --x--> (B, C, D), hoặc
   - (A, B, C) --x--> D.
10. Nếu khó chọn relation_type chính xác, chọn relation_type gần nhất trong schema thay vì bỏ relation có ý nghĩa.
11. Entity list phải trích xuất tất cả thực thể y khoa rõ ràng thuộc schema trong chunk, không chỉ subject/object.
12. Sau khi có triples, kiểm tra lại để entity list có đủ mọi subject/object của relation.
13. Với mỗi câu/bullet có tri thức y khoa, cố gắng tạo ít nhất một relation nếu có thể.
14. Bullet phòng ngừa: object thường là hành động/biện pháp, không phải tác nhân bị ngăn. Ví dụ "lau khô đầu để hạn chế nấm" -> object là "lau khô đầu".
15. Bullet điều trị: object thường là thuốc, hoạt chất, chăm sóc hoặc can thiệp.
16. Câu nguyên nhân/yếu tố: object là tác nhân, môi trường, cơ địa, bệnh nền hoặc yếu tố nguy cơ.
17. Câu triệu chứng: object là biểu hiện lâm sàng.
18. Cuối cùng mới gắn relation.keywords dựa trên triple và evidence_text đã chọn.

Ví dụ:

- "lác đồng tiền khiến người bệnh ngứa dữ dội"
  => lác đồng tiền CO_BIEU_HIEN ngứa dữ dội

- "thuốc kháng Histamine cải thiện ngứa dữ dội ở người bệnh hắc lào"
  => ngứa dữ dội DIEU_TRI_BANG thuốc kháng Histamine
  => hắc lào DIEU_TRI_BANG thuốc kháng Histamine

- "Ketoconazole là thuốc trị lác đồng tiền"
  => lác đồng tiền DIEU_TRI_BANG Ketoconazole

- "Việc lau khô hoặc sấy khô đầu sau khi gội là rất quan trọng để hạn chế sự ẩm ướt, tạo điều kiện cho nấm phát triển"
  => nấm da đầu PHONG_NGUA_BANG lau khô hoặc sấy khô đầu sau khi gội
  Không tạo: nấm da đầu PHONG_NGUA_BANG nấm

- "Móng tay có vết tổn thương nhỏ hoặc đang mắc bệnh về da như vẩy nến. Người bị tiểu đường, hội chứng Down có nguy cơ cao hơn"
  => nấm móng tay LIEN_QUAN_YEU_TO vẩy nến
  => nấm móng tay LIEN_QUAN_YEU_TO tiểu đường
  => nấm móng tay LIEN_QUAN_YEU_TO hội chứng Down
""".strip()


CHUNK_CLASSIFICATION_RULES = """
Trước khi extract, phải phân loại chunk:
- Nếu chunk có tri thức y khoa thuộc schema, đặt is_medical_knowledge_chunk=true, không thì =false.
- Khi is_medical_knowledge_chunk=false: skip_reason phải là một chuỗi ngắn,
  entities=[], relations=[].

Ví dụ giữ phần y khoa:
Text: "Người bệnh viêm da cơ địa nên đến bệnh viện X để được khám."
Output: có thể extract entity "viêm da cơ địa" nếu cần, nhưng không extract "bệnh viện X".
""".strip()


SYNONYM_RULES = """
Quy tắc DONG_NGHIA_VOI:
- Phải tạo entity riêng và relation DONG_NGHIA_VOI khi chunk có tín hiệu ngôn ngữ rõ ràng như:
  "còn gọi là", "còn được gọi là", "hay gọi là", "còn có tên là", "tên khác là",
  "tiếng Anh là", "viết tắt là", hoặc ngoặc giải thích tên gọi.

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


NEGATIVE_EVIDENCE_RULES = """
Không tạo relation từ evidence chỉ là câu hỏi/heading FAQ.
""".strip()


def build_system_prompt(mode: str = "strict_text") -> str:
    if mode != "strict_text":
        raise ValueError(
            "Only strict_text mode is supported. "
            "LLM canonicalization is disabled; use DONG_NGHIA_VOI edges instead."
        )

    merge_policy = """
Entity naming policy:
- name phải ưu tiên đúng cách gọi xuất hiện trong chunk.
- Nếu text nêu rõ hai tên khác nhau là đồng nghĩa, phải tạo entity riêng cho từng tên và tạo relation DONG_NGHIA_VOI.
""".strip()

    return f"""
Bạn là bộ trích xuất tri thức y khoa tiếng Việt cho Medical GraphRAG.

Nhiệm vụ:
- Trước hết phân loại nhanh văn bản bằng is_medical_knowledge_chunk / skip_reason.
- Trích xuất tất cả entity y khoa rõ ràng thuộc schema, có evidence trong chunk/context.
- Đánh dấu entity anchor tiềm năng bằng is_anchor_candidate/anchor_reason để graph có thể giữ node quan trọng dù chưa nối được relation trong chunk hiện tại.
- Trích xuất relation/relation_group y khoa theo fact trong evidence, ưu tiên recall relation.
- relation.keywords có thể để [].
- Chỉ dùng schema được cho.
- Mỗi relation phải có evidence_text (evidence có thể là tổng hợp các phần rải rác ở nhiều câu).
- Không tạo relation cho phần quảng bá/hành chính.
- Nếu chunk không có tri thức y khoa thuộc schema, trả về entities=[] và relations=[].

{CHUNK_CLASSIFICATION_RULES}

ENTITY TYPES được phép:
{", ".join(sorted(ENTITY_TYPES))}

Định nghĩa Entity Types:
{build_entity_type_descriptions()}

RELATION TYPES được phép:
{", ".join(sorted(RELATION_TYPES))}

Định nghĩa Relation Types:
{build_relation_type_descriptions()}

Gợi ý subject/object theo schema, dùng để chọn hướng relation nhưng KHÔNG ép cứng như ontology:
{build_relation_rules_text()}

Không trích xuất:
- các label/generic schema như: bệnh lý, biểu hiện lâm sàng, triệu chứng, nguyên nhân, biến chứng, điều trị, phòng ngừa, chẩn đoán
- các quan hệ quá mơ hồ, không có evidence y khoa trực tiếp

Quy tắc entity:
- name là tên thực thể ngắn gọn, đúng như xuất hiện trong chunk.
- Mỗi entity chỉ đại diện cho một tên gọi cụ thể.
- description là mô tả ngắn 1 câu dựa trên chunk; để "" nếu chunk không đủ thông tin.
- evidence_text là đoạn bằng chứng ngắn tách ghép từ chunk.
- Entity list nên bắt rộng: cái gì là khái niệm y khoa rõ ràng thuộc schema thì trích xuất, kể cả khi chưa nối được relation.
- Sau khi tạo relation, phải đảm bảo mọi subject/object đều có trong entity list.
- Không tự loại entity chỉ vì nghĩ là kém quan trọng; validator/aggregator sẽ xử lý entity rác hoặc entity không dùng được.
- Với mỗi entity, đặt is_anchor_candidate=true nếu entity có thể hữu ích độc lập hoặc liên quan nhiều bệnh: bệnh lý, thuốc/hoạt chất, phương pháp chẩn đoán, tác nhân bệnh sinh, yếu tố nguy cơ quan trọng, triệu chứng/dấu hiệu đặc hiệu.
- Đặt is_anchor_candidate=false cho entity hành chính/quảng bá/đối tượng quá chung: người bệnh, bệnh nhân, trẻ em, phụ huynh, bác sĩ, bệnh viện, khoa da liễu, khách hàng, các thành viên trong gia đình.
- anchor_reason là lý do ngắn, ví dụ "disease_anchor", "drug_anchor", "diagnostic_method_anchor", "pathogen_anchor"; để "" nếu is_anchor_candidate=false.

{merge_policy}

{SYNONYM_RULES}

Quy tắc relation:
- subject/object phải là entity y khoa.
- relation_type phải nằm trong schema.
- Relation nên đi theo subject/object gợi ý của schema, nhưng không bỏ fact rõ chỉ vì endpoint hơi lệch type.
- evidence_text phải hỗ trợ relation; có thể dùng heading path để khôi phục mối quan hệ ngầm của list/bullet.
- confidence từ 0.0 đến 1.0.
- Ưu tiên độ phủ relation tốt theo fact trong evidence, không biến bệnh chính thành trung tâm bắt buộc của mọi cạnh. Schema là guardrail mềm, evidence là ưu tiên chính.
- Hãy scan theo checklist mềm: biểu hiện, yếu tố/nguyên nhân, chẩn đoán, điều trị, phòng ngừa, cần tránh, biến chứng, chẩn đoán phân biệt và quan hệ trực tiếp khác giữa các entity.

{RELATION_COVERAGE_RULES}

Quy tắc riêng theo relation type lấy từ schema, render theo style prompt cũ:
{build_relation_prompt_guidance_text()}

{RELATION_KEYWORD_RULES}

{NEGATIVE_EVIDENCE_RULES}
""".strip()


SYSTEM_PROMPT = build_system_prompt("strict_text")


def build_user_prompt(
    article_title: str,
    section: str | None,
    subsection: str | None,
    chunk_text: str,
    heading_context: str | None = None,
) -> str:
    section_text = section or ""
    subsection_text = subsection or ""
    heading_context_text = heading_context or ""

    return f"""
Bài viết: {article_title}
Mục hiện tại: {section_text}
Tiểu mục hiện tại: {subsection_text}
Đường dẫn mục tại đầu chunk: {heading_context_text}

Lưu ý:
- Đường dẫn mục là ngữ cảnh thật từ cấu trúc bài viết, dùng để xác định subject ngầm cho đoạn đầu chunk và các bullet/list liên quan.
- Chunk bên dưới là nội dung gốc
Chunk:
{chunk_text}
""".strip()