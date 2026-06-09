
### 1. Bảng Danh mục Thực thể (Entity Types - 9 Loại)

| Entity Type | Định nghĩa | Ví dụ tiêu biểu |
| --- | --- | --- |
| **`BENH_LY`** | Bệnh, hội chứng, tình trạng y khoa, biến chứng. | Viêm da cơ địa, nhiễm trùng da, suy thận. |
| **`BIEU_HIEN_LAM_SANG`** | Triệu chứng (chủ quan) hoặc dấu hiệu (khách quan). | Ngứa, ban đỏ, mụn nước, ho khan, nứt nẻ. |
| **`YEU_TO_BENH_SINH`** | Nguyên nhân, cơ chế, yếu tố nội tại, tác nhân gây bệnh. | Di truyền, rối loạn miễn dịch, vi khuẩn tụ cầu. |
| **`YEU_TO_NGOAI_SINH`** | Yếu tố bên ngoài môi trường, thói quen sinh hoạt. | Bụi, rượu bia, tắm nước nóng, phấn hoa. |
| **`PHUONG_PHAP_CHAN_DOAN`** | Cách thức/kỹ thuật phát hiện bệnh. | Khám lâm sàng, hỏi bệnh sử, test dị ứng, MRI. |
| **`CAN_THIEP_Y_TE`** | Biện pháp điều trị, chăm sóc y khoa **không phải thuốc**. | Chườm lạnh, vệ sinh vết thương, phẫu thuật. |
| **`THUOC_VA_HOAT_CHAT`** | Thuốc, hoạt chất cụ thể hoặc nhóm thuốc. | Corticosteroid, kháng sinh, Paracetamol. |
| **`VI_TRI_GIAI_PHAU`** | Cơ quan, hệ cơ quan hoặc vị trí trên cơ thể. | Da, gan, phổi, khuỷu tay, hai bên má. |
| **`DOI_TUONG`** | Nhóm người hoặc nhóm bệnh nhân. | Trẻ sơ sinh, phụ nữ mang thai, người lớn tuổi. |

---

### 2. Bảng Danh mục Quan hệ & Ranh giới (Relation Types & Rules - 12 Loại)

*Bảng này đóng vai trò là "Còng tay Validator" để chặn mọi thông tin nhiễu từ LLM.*

| Relation Type | Ý nghĩa / Ngữ cảnh | Chủ thể (Subject / Domain) | Đối tượng (Object / Range) |
| --- | --- | --- | --- |
| **`LA_DANG_CUA`** | Phân loại bệnh học (Taxonomy). | `BENH_LY` | `BENH_LY` |
| **`CHAN_DOAN_PHAN_BIET_VOI`** | Bệnh cần phân biệt để tránh chẩn đoán nhầm. | `BENH_LY` | `BENH_LY` |
| **`CO_BIEU_HIEN`** | Bệnh gây ra các triệu chứng, dấu hiệu nào. | `BENH_LY` | `BIEU_HIEN_LAM_SANG` |
| **`LIEN_QUAN_YEU_TO`** | Bệnh có nguyên nhân/nguy cơ/cơ chế/trigger từ đâu. | `BENH_LY` | `YEU_TO_BENH_SINH`, `YEU_TO_NGOAI_SINH` |
| **`GAY_BIEN_CHUNG`** | Bệnh này dẫn đến tình trạng xấu nào tiếp theo. | `BENH_LY` | `BENH_LY` |
| **`ANH_HUONG_DEN`** | Bệnh/Triệu chứng tác động lên vị trí cơ thể hoặc ai. | `BENH_LY`, `BIEU_HIEN_LAM_SANG` | `VI_TRI_GIAI_PHAU`, `DOI_TUONG` |
| **`CHAN_DOAN_BANG`** | Dùng cách gì hoặc dựa vào biểu hiện gì để phát hiện. | `BENH_LY` | `PHUONG_PHAP_CHAN_DOAN`, `BIEU_HIEN_LAM_SANG` |
| **`DIEU_TRI_BANG`** | Bệnh/Triệu chứng được kiểm soát/chữa bằng cách nào. | `BENH_LY`, `BIEU_HIEN_LAM_SANG` | `CAN_THIEP_Y_TE`, `THUOC_VA_HOAT_CHAT` |
| **`PHONG_NGUA_BANG`** | Làm gì để bệnh không xảy ra/tái phát. | `BENH_LY` | `CAN_THIEP_Y_TE`, `YEU_TO_NGOAI_SINH` |
| **`CAN_TRANH`** | Bệnh nhân nên kiêng/hạn chế cái gì để không nặng thêm. | `BENH_LY`, `DOI_TUONG` | `YEU_TO_NGOAI_SINH`, `THUOC_VA_HOAT_CHAT` |
| **`CHONG_CHI_DINH`** | Thuốc/Thủ thuật tuyệt đối không dùng cho đối tượng nào. | `THUOC_VA_HOAT_CHAT`, `CAN_THIEP_Y_TE` | `BENH_LY`, `DOI_TUONG` |
| **`TUONG_TAC_VOI`** | Thuốc xung đột với thuốc khác hoặc yếu tố bên ngoài. | `THUOC_VA_HOAT_CHAT` | `THUOC_VA_HOAT_CHAT`, `YEU_TO_NGOAI_SINH` |

---

### 3. Bảng Thuộc tính Cốt lõi (Base Properties)

Để tối giản hóa và giữ Graph sạch nhất có thể, chỉ lưu các thuộc tính sau trên Đồ thị. Mọi logic phức tạp sẽ được LLM Generation tự suy luận dựa vào `evidence_text`.

* **Trên Entity (Node):**
* `name` *(String)*: Tên thực thể trích xuất.
* `entity_type` *(Enum)*: Một trong 9 loại ở Bảng 1.


* **Trên Relation (Edge):**
* `subject` *(String)*: Tên Node gốc.
* `relation_type` *(Enum)*: Một trong 12 loại ở Bảng 2.
* `object` *(String)*: Tên Node đích.
* `evidence_text` *(String)*: Câu văn trích xuất nguyên bản từ Text Chunk để chứng minh quan hệ này tồn tại (Dùng để chống ảo giác).
* `confidence` *(Float)*: Điểm tự tin do LLM tự đánh giá (từ 0.0 đến 1.0).