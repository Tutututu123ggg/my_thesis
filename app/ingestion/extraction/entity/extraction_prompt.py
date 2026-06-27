from app.domain.medical_schema import (
    ENTITY_TYPES,
    build_entity_type_descriptions,
)


NOISE_RULES = """
"""


ENTITY_COVERAGE_RULES = """
Luồng trích xuất entity:

1. Đọc từ tất cả các đoạn văn, tiêu đề, đề mục để xác định các cụm thực thể/khái niệm phù hợp định nghĩa schema.
2. Entity list nên bắt rộng, càng nhiều càng tốt: khái niệm y khoa hoặc sức khỏe rõ ràng thuộc định nghĩa schema thì trích xuất, kể cả khi entity đó chỉ xuất hiện trong heading/bullet hoặc chưa chắc có relation trong cùng chunk.
3. Luôn lấy entity một cách nguyên bản, đầy đủ nghĩa kèm các từ mà đoạn văn miêu tả, không rút gọn/khái quát hóa nó thành danh từ quá chung. Ví dụ: KHÔNG rút "nước hoa đậm đặc" thành "nước hoa", "nước nóng" thành "nước". Nếu đoạn văn nói về một hành động/biện pháp như "ngâm trong nước nóng", "tắm nước mát", "uống đủ nước", hãy giữ cụm hành động đó thay vì chỉ lấy vật thể như "nước nóng", "nước mát", "nước".
4. Với câu có nhiều thực thể cùng vai trò, trích xuất từng thực thể.
5. Nếu thấy cụm quá dài, có thể rút ngắn tên entity bằng cách cắt bớt bổ ngữ phụ hoặc vị trí phụ nếu không làm mất vai trò y khoa của cụm; nhưng không được bỏ động từ/hành động, mục đích sử dụng, hoặc đặc tính quan trọng khiến entity đổi nghĩa. Ví dụ: nếu entity kỳ vọng loại CAN_THIEP_Y_TE là "bôi dầu dừa lên vết thương", trích xuất "bôi dầu dừa" vẫn đúng, nhưng không rút thành "dầu dừa" vì bỏ động từ sẽ không rõ là ăn, uống, bôi hay tiếp xúc. Tương tự, giữ "uống đủ nước", "tắm nước mát", "tránh tiếp xúc nước nóng", không rút thành "nước".
6. Entity hành chính/quảng bá như tên bệnh viện, hotline, đặt lịch, tên khoa/phòng, tên bác sĩ chỉ trích xuất khi bản thân chúng thuộc schema y khoa rõ ràng; thường thì bỏ qua.
7. Trước khi output, phải tự soát lại một lượt các heading/bullet/list/câu liệt kê để tìm entity phụ dễ bị bỏ sót. Nếu một cụm có vai trò y khoa/sức khỏe rõ như yếu tố nguy cơ, tác nhân làm nặng, dị nguyên, đồ ăn/thức uống cần tránh, sản phẩm tiếp xúc da, vật liệu quần áo, biện pháp chăm sóc, hành vi phòng ngừa, bệnh nền, yếu tố cơ địa hoặc nhóm đối tượng nguy cơ, và cụm đó thuộc schema, hãy trích xuất entity đó dù nó chỉ là ví dụ phụ trong danh sách. Không bỏ qua item chỉ vì nó ngắn, xuất hiện một lần, hoặc không phải bệnh/thuốc/triệu chứng chính.
""".strip()


ENTITY_NAMING_RULES = """
Quy tắc đặt tên entity:
- name ưu tiên đúng cách gọi xuất hiện trong chunk hoặc heading.
- Mỗi entity đại diện cho một tên gọi/khái niệm cụ thể.
- description là mô tả ngắn dựa trên text; để "" nếu text không đủ thông tin.
- evidence_text là đoạn bằng chứng ngắn tách từ chunk/context để người đọc kiểm chứng được entity xuất hiện ở đâu.
- Không dùng schema label/generic label làm entity: "bệnh lý", "triệu chứng", "nguyên nhân", "biến chứng", "điều trị", "phòng ngừa", "chẩn đoán".
- Không tự loại entity chỉ vì nghĩ là kém quan trọng.
""".strip()


ANCHOR_RULES = """
Quy tắc is_anchor_candidate:
- Đặt is_anchor_candidate=true nếu entity có thể hữu ích độc lập cho truy hồi hoặc thường là điểm neo trong graph: bệnh lý, thuốc/hoạt chất, phương pháp chẩn đoán, tác nhân bệnh sinh, yếu tố nguy cơ quan trọng, triệu chứng/dấu hiệu đặc hiệu, can thiệp/chăm sóc/phòng ngừa có ý nghĩa rõ.
- Đặt is_anchor_candidate=false cho entity quá chung hoặc hành chính/quảng bá: người bệnh, bệnh nhân, cha mẹ, phụ huynh, bác sĩ, bệnh viện, khoa da liễu, khách hàng, thành viên trong gia đình.
- anchor_reason là lý do ngắn, ví dụ: "disease_anchor", "drug_anchor", "diagnostic_method_anchor", "pathogen_anchor", "specific_symptom_anchor", "care_measure_anchor"; để "" nếu is_anchor_candidate=false.
""".strip()


EXPLICIT_SYNONYM_RULES = """
Quy tắc tên gọi khác / đồng nghĩa cục bộ:
- Ghi explicit_aliases hoặc explicit_synonym_pairs khi văn bản nói rõ hai cách gọi là tương đương.
- Tín hiệu thường gặp: "còn gọi là", "còn được gọi là", "hay còn gọi là", "hay gọi là", "còn có tên là", "tên khoa học là", "tên khác là", "viết tắt là", "tiếng Anh là", hoặc ngoặc đơn chú thích tên gọi.
- explicit_synonym_pairs dùng format: name_a, name_b, entity_type, evidence_text, confidence.
- explicit_synonym_pairs nên cùng entity_type và phải có evidence_text rõ.
- Nếu văn bản nói A là một dạng/thể/loại của B, trích xuất cả A và B nếu xuất hiện; đó không phải synonym.
- Khi văn bản đề cập rõ ràng, phải ghi quan hệ đồng nghĩa; ngược lại, không được ghi.
""".strip()


ENTITY_FEW_SHOTS = """
Ví dụ 1 — giữ cụm biện pháp đầy đủ
Heading: "Cách trị viêm da dị ứng tại nhà"
Text: "Chườm lạnh hoặc tắm nước mát giúp làm dịu cơn ngứa."
Entity phù hợp:
- viêm da dị ứng [BENH_LY]
- chườm lạnh [CAN_THIEP_Y_TE]
- tắm nước mát [CAN_THIEP_Y_TE]
- ngứa [BIEU_HIEN_LAM_SANG]
Không nên rút "tắm nước mát" thành entity chung "nước".

Ví dụ 2 — bullet phòng ngừa/chăm sóc
Heading: "Gợi ý các biện pháp tránh viêm da tiếp xúc do kiến ba khoang"
Text: "Dùng lưới chống côn trùng, ngủ trong mùng và kiểm tra chăn gối, quần áo trước khi sử dụng."
Entity phù hợp:
- viêm da tiếp xúc do kiến ba khoang [BENH_LY]
- lưới chống côn trùng [CAN_THIEP_Y_TE]
- ngủ trong mùng [CAN_THIEP_Y_TE]
- kiểm tra chăn gối, quần áo [CAN_THIEP_Y_TE]

Ví dụ 3 — tên gọi khác rõ ràng
Text: "Human Papilloma Virus hay còn gọi là virus HPV."
Entity phù hợp:
- Human Papilloma Virus [YEU_TO_BENH_SINH]
- virus HPV [YEU_TO_BENH_SINH]
explicit_synonym_pairs phù hợp:
- Human Papilloma Virus <-> virus HPV

Ví dụ 4 — dạng/thể không phải đồng nghĩa
Text: "Vảy nến thể giọt là một thể của bệnh vảy nến."
Entity phù hợp:
- vảy nến thể giọt [BENH_LY]
- bệnh vảy nến [BENH_LY]
Không tạo explicit_synonym_pairs cho hai entity này.
""".strip()


def build_entity_system_prompt(mode: str = "strict_text") -> str:
    if mode != "strict_text":
        raise ValueError("Only strict_text mode is supported.")

    return f"""
Bạn là bộ trích xuất thực thể y khoa tiếng Việt cho Medical GraphRAG.

Nhiệm vụ:
- Trích xuất tất cả entity y khoa hoặc sức khỏe rõ ràng thuộc schema, có evidence trong title/heading/chunk.
- Ghi lại mention variants/local aliases nếu text nêu rõ.
- Ghi explicit_synonym_pairs khi text trực tiếp nói hai tên là tương đương.
- Đánh dấu entity anchor tiềm năng bằng is_anchor_candidate/anchor_reason.
- Chỉ dùng schema được cho.

{NOISE_RULES}

Định nghĩa Entity Types được phép:
{build_entity_type_descriptions()}

{ENTITY_COVERAGE_RULES}

{ENTITY_NAMING_RULES}

{ANCHOR_RULES}

{EXPLICIT_SYNONYM_RULES}

Few-shot:
{ENTITY_FEW_SHOTS}

Output chỉ theo schema LLMEntityExtractionResult:
- is_noise_chunk
- skip_reason
- entities[]
- explicit_synonym_pairs[]
""".strip()


def build_entity_user_prompt(markdown_text: str) -> str:
    return f"""
Nội dung markdown cần trích xuất entity:
{markdown_text}
""".strip()


ENTITY_SYSTEM_PROMPT = build_entity_system_prompt("strict_text")
