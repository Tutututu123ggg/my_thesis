from __future__ import annotations


def build_medical_answer_prompt(question: str, context: str) -> str:
    return f"""
Câu hỏi của người dùng:
{question}

Context truy xuất:
{context}

Yêu cầu trả lời:
- Trả lời bằng tiếng Việt, ngắn gọn nhưng đủ ý.
- Chỉ dùng thông tin có trong context; không bịa thêm bệnh, thuốc hoặc lời khuyên ngoài context.
- Nếu người dùng mô tả triệu chứng, chỉ nêu các khả năng liên quan, không chẩn đoán chắc chắn.
- Nêu lý do dựa trên context truy xuất.
- Nhắc người dùng đi khám nếu triệu chứng kéo dài, nặng lên, khó thở, sốt, đau nhiều, chảy dịch/mủ, sưng nóng hoặc ảnh hưởng sinh hoạt.
""".strip()
