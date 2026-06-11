from __future__ import annotations

from app.evaluation.models import EvaluationQuestion


SMALL_MEDICAL_EVAL_SET: list[EvaluationQuestion] = [
    EvaluationQuestion(
        difficulty="easy",
        question="viêm da cơ địa có triệu chứng gì",
        reference=(
            "Viêm da cơ địa thường có ngứa, da khô, ban đỏ, bong/tróc vảy, nứt nẻ; "
            "có thể có mụn nước, rỉ dịch, dày sừng, bội nhiễm hoặc mất ngủ do ngứa."
        ),
        expected_terms=["viêm da cơ địa", "ngứa", "da khô", "ban đỏ"],
    ),
    EvaluationQuestion(
        difficulty="easy",
        question="vảy nến có những dạng nào",
        reference=(
            "Các dạng/phân loại vảy nến thường gặp gồm vảy nến thể mảng, thể giọt, "
            "thể mủ, thể đảo ngược, đỏ da toàn thân và các dạng khác nếu tài liệu có nêu."
        ),
        expected_terms=["vảy nến", "thể mảng", "thể giọt"],
    ),
    EvaluationQuestion(
        difficulty="graph",
        question="Chàm thể tạng còn được gọi là gì và có liên quan gì với viêm da cơ địa?",
        reference=(
            "Chàm thể tạng là tên gọi khác/đồng nghĩa được nêu cho viêm da cơ địa hoặc viêm da dị ứng. "
            "Hệ thống nên truy xuất quan hệ DONG_NGHIA_VOI và bằng chứng từ nguồn."
        ),
        expected_terms=["chàm thể tạng", "viêm da cơ địa", "đồng nghĩa"],
    ),
    EvaluationQuestion(
        difficulty="graph",
        question="Những bệnh da nào có biểu hiện ngứa nhưng cần phân biệt với vảy nến?",
        reference=(
            "Các bệnh da có thể gây ngứa/bong vảy và cần phân biệt với vảy nến có thể gồm viêm da tiết bã, "
            "chàm/viêm da cơ địa, nấm da hoặc các viêm da khác tùy bằng chứng truy xuất."
        ),
        expected_terms=["vảy nến", "viêm da tiết bã", "ngứa"],
    ),
    EvaluationQuestion(
        difficulty="graph",
        question="Các bệnh nào có thể nặng hơn do bụi hoặc phấn hoa?",
        reference=(
            "Bụi hoặc phấn hoa là dị nguyên/yếu tố môi trường có thể làm nặng các bệnh dị ứng như viêm mũi dị ứng, "
            "hen suyễn dị ứng/hen phế quản dị ứng, viêm da cơ địa hoặc viêm da dị ứng nếu nguồn có bằng chứng."
        ),
        expected_terms=["bụi", "phấn hoa", "viêm mũi dị ứng", "hen"],
    ),
    EvaluationQuestion(
        difficulty="symptom",
        question="Da tôi bị khô, ngứa nhiều về đêm, gãi xong đỏ và bong vảy, có thể là vấn đề gì?",
        reference=(
            "Các khả năng cần tra cứu gồm viêm da cơ địa/viêm da dị ứng/chàm và một số bệnh da gây ngứa bong vảy. "
            "Nếu có rỉ dịch, đau, sưng nóng, sốt hoặc mất ngủ nhiều thì nên đi khám."
        ),
        expected_terms=["ngứa", "viêm da cơ địa", "chàm"],
    ),
    EvaluationQuestion(
        difficulty="symptom",
        question="Con tôi vài tháng tuổi bị nổi mẩn đỏ ở hai bên má, ngứa, có mụn nước nhỏ và hay quấy khóc, nên nghĩ tới bệnh gì?",
        reference=(
            "Ở trẻ nhỏ, mẩn đỏ hai má, ngứa, mụn nước nhỏ, quấy khóc có thể gợi ý viêm da cơ địa/chàm sữa/chàm thể tạng; "
            "cần khám nhi hoặc da liễu để xác nhận và xử trí an toàn."
        ),
        expected_terms=["trẻ", "hai bên má", "viêm da cơ địa", "chàm"],
    ),
    EvaluationQuestion(
        difficulty="symptom",
        question="Tôi thường ho khan, khò khè, khó thở về đêm hoặc khi gặp bụi, lông thú, phấn hoa, có thể là bệnh gì?",
        reference=(
            "Ho khan, khò khè, khó thở về đêm hoặc khi gặp dị nguyên như bụi/lông thú/phấn hoa có thể gợi ý hen suyễn, "
            "đặc biệt hen suyễn dị ứng; cần đi khám nếu khó thở, tức ngực hoặc triệu chứng tái diễn."
        ),
        expected_terms=["khò khè", "khó thở", "hen", "dị ứng"],
    ),
    EvaluationQuestion(
        difficulty="symptom",
        question="Tôi có mấy nốt nước nhỏ ở tay, rất ngứa, khỏi rồi lại bị lại, nhất là khi tay ra mồ hôi.",
        reference=(
            "Mụn/nốt nước nhỏ ở tay, rất ngứa và tái phát, nhất là khi ra mồ hôi, có thể gợi ý tổ đỉa hoặc nhóm chàm/viêm da; "
            "không nên tự quy thành mụn trứng cá nếu context không hỗ trợ."
        ),
        expected_terms=["nốt nước", "ngứa", "tổ đỉa", "chàm"],
    ),
    EvaluationQuestion(
        difficulty="symptom",
        question="Tôi vừa có da hay nổi chàm ngứa, vừa hay hắt hơi sổ mũi và thỉnh thoảng khò khè, các vấn đề này có liên quan không?",
        reference=(
            "Các biểu hiện chàm/ngứa da, viêm mũi dị ứng và khò khè/hen có thể cùng thuộc cơ địa dị ứng/atopy; "
            "hệ thống nên truy xuất viêm da cơ địa/viêm da dị ứng, viêm mũi dị ứng và hen suyễn như các vấn đề liên quan."
        ),
        expected_terms=["chàm", "viêm mũi dị ứng", "hen", "cơ địa dị ứng"],
    ),
]
