# item-analysis

Hệ thống phân tích và chuẩn hóa câu trắc nghiệm (CTT, IRT, BKT) cho giảng viên.
Nhóm 2 người, kế hoạch 8 tuần, Python 3.12, Windows.

## Kiến trúc
- core/: thuần Python (numpy/pandas/scipy), có unit test. Mọi công thức nằm ở đây.
- api/ (FastAPI) và app/ (Streamlit gọi API): làm sau. storage/ (SQLite): làm sau.
- InfoSecCAT và ASSISTments chỉ là dữ liệu mẫu; hệ thống phải xử lý dataset khác của người dùng.

## Quy tắc bắt buộc
- Type hint và docstring tiếng Việt cho mọi hàm; mỗi hàm có unit test.
- Ngưỡng đọc từ core/config.py (CTTThresholds), không ghi cứng. Không ghi cứng đường dẫn.
- Không tự cài lại thứ thư viện đã có. Không bịa tên cột hay định dạng file; thiếu thông tin thì hỏi.
- Không commit dữ liệu thô: thư mục data/ nằm trong .gitignore.
- Mỗi task làm trên một nhánh Git riêng, không commit thẳng vào main.

## Quy ước dữ liệu
- Ma trận lựa chọn: index = student, cột = question_id (chuỗi), ô trống = "" (tính sai).
- Ma trận chấm điểm: 0/1 (int8). Key: index question_id, cột correct_answer, num_choices.
- Thí sinh bỏ trống toàn bộ bị loại mặc định (core.scoring.filter_blank_students).

## Chạy test
.\.venv\Scripts\python.exe -m unittest discover -s tests -t . -v
(Trên máy này `python` chưa nằm trên PATH; luôn dùng python trong .venv.)
Test dữ liệu thật tự bỏ qua nếu thiếu data/raw/infoseccat/ (hoặc đặt INFOSECCAT_DIR).

## Tiến độ
Xong: tuần 1 (io, scoring, config), T2.0 (lọc thí sinh trống), T2.1 (p và phân loại độ khó),
T2.2 (split_groups, discrimination_index; nhóm 27% sắp xếp ổn định, hòa điểm theo thứ tự dòng).
Còn lại tuần 2: T2.3 r_pbis, T2.4 KR-20, T2.5 bảng một dòng mỗi câu + cờ, T2.6 scripts/run_ctt.py.
