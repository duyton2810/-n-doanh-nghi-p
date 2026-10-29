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
- Commit ngay khi xong task, trước khi tạo nhánh khác; git status phải sạch trước khi đổi nhánh.

## Quy trình dữ liệu
nạp -> làm sạch -> chấm điểm -> phân tích; mọi phân tích dùng core.cleaning.prepare_dataset
(không tự gọi score_matrix trên dữ liệu thô).

## Quy ước dữ liệu
- Ma trận lựa chọn: index = student, cột = question_id (chuỗi), ô trống = "" (tính sai).
- Ma trận chấm điểm: 0/1 (int8). Key: index question_id, cột correct_answer, num_choices.
- Thí sinh bỏ trống toàn bộ bị loại mặc định (core.cleaning.clean_responses; exclude_blank_students=false / --keep-blank để giữ).

## Chạy test
.\.venv\Scripts\python.exe -m unittest discover -s tests -t . -v
(Trên máy này `python` chưa nằm trên PATH; luôn dùng python trong .venv.)
Test dữ liệu thật tự bỏ qua nếu thiếu data/raw/infoseccat/ (hoặc đặt INFOSECCAT_DIR).

## Tiến độ
Xong: tuần 1 (io, scoring, config); T0: core/cleaning.py (làm sạch, prepare_dataset) đứng trước mọi phân tích;
T2.0-T2.6: core/ctt.py (p, D, r_pbis, KR-20, ctt_report);
scripts/run_ctt.py (mặc định LOẠI thí sinh bỏ trống toàn bộ, N = 273; có --keep-blank, --exclude-blank và --compare).
Việc kế tiếp: tuần 3 (distractor, simulate, mapping, thử pyBKT, điều khoản ASSISTments).
