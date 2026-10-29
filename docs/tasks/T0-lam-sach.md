# T0: bước làm sạch dữ liệu, đứng TRƯỚC mọi phân tích (đọc CLAUDE.md trước)

Sửa/tạo: core/cleaning.py (mới), core/config.py, config.toml, core/ctt.py, scripts/run_ctt.py,
tests/ (thêm test mới; cập nhật test dựa vào mặc định cũ), CLAUDE.md.
Python trong .venv. Dừng nếu git status không sạch. Commit ngay khi xong.

## Quyết định của người dùng
Làm sạch dữ liệu là bước đầu tiên của quy trình: nạp -> LÀM SẠCH -> chấm điểm -> phân tích.
Thí sinh bỏ trống toàn bộ bị LOẠI (coi như không dự thi), không tính là "làm sai hết".
Kết quả chính của hệ thống là sau làm sạch: InfoSecCAT cho N = 273, không phải 276.
Các ô bỏ trống của thí sinh còn lại vẫn tính là sai (còn 7 ô).

## core/cleaning.py
- CleaningReport (frozen dataclass): n_before, n_after, removed (DataFrame: student, reason, n_answered),
  warnings (list[str]).
- clean_responses(responses, thresholds) -> tuple[pd.DataFrame, CleaningReport]:
  - loại thí sinh bỏ trống toàn bộ khi thresholds.exclude_blank_students (dùng blank_students có sẵn),
    lý do "bỏ trống toàn bộ", n_answered = 0;
  - cảnh báo (KHÔNG loại) thí sinh có tỉ lệ bỏ trống >= thresholds.warn_omit_rate (mặc định 0.5);
  - không sửa dữ liệu đầu vào; ValueError nếu loại hết thí sinh.
- prepare_dataset(responses, key, thresholds) -> PreparedDataset (frozen dataclass: responses đã sạch,
  scored 0/1, report). Thứ tự bắt buộc: validate_responses -> clean_responses -> score_matrix.
  Đây là LỐI VÀO DUY NHẤT: mọi phân tích (hiện tại ctt_report; sau này distractor, irt, bkt) chỉ nhận
  dữ liệu từ prepare_dataset, không tự gọi score_matrix trên dữ liệu thô.
- Thêm warn_omit_rate: float = 0.5 vào CTTThresholds và config.toml (khoảng (0, 1]).

## Đổi mặc định và tích hợp
- exclude_blank_students mặc định True ở CTTThresholds và config.toml.
- Giữ tùy chọn exclude_blank_students=false / --keep-blank cho dataset khác; giữ --exclude-blank và --compare.
- ctt_report gọi prepare_dataset (thay cho gọi filter_blank_students + score_matrix trực tiếp).
  summary thêm cleaning_removed (danh sách mã thí sinh) và cleaning_warnings. Công thức CTT không đổi.
- scripts/run_ctt.py: ghi thêm cleaning_removed.csv (student, reason, n_answered) vào out-dir;
  dòng tóm tắt đầu tiên in số thí sinh đã loại.
- CLAUDE.md: thêm mục "Quy trình dữ liệu: nạp -> làm sạch -> chấm điểm -> phân tích; mọi phân tích dùng
  core.cleaning.prepare_dataset"; sửa "mặc định GIỮ, N = 276" thành "mặc định LOẠI, N = 273";
  ghi T0 vào mục Tiến độ, đứng trước T2.0.

## Test
- Được phép sửa các test dựa vào mặc định cũ (test_config, test_ctt phần ctt_report, test_run_ctt):
  chỉ đổi kỳ vọng mặc định; KHÔNG xóa test, KHÔNG sửa test của chế độ giữ (exclude_blank_students=false
  vẫn phải cho n_students = 276, kr20 ≈ 0.858).
- Test mới (bộ nhỏ): loại đúng người, lý do đúng, cảnh báo ngưỡng 0.5 (đúng bằng ngưỡng thì cảnh báo),
  input không bị sửa, loại hết thì ValueError, thí sinh sai hết nhưng có trả lời KHÔNG bị loại;
  prepare_dataset: đúng thứ tự (dữ liệu sai phương án vẫn báo ValueError trước khi loại), scored chỉ gồm
  thí sinh còn lại, index/cột đúng thứ tự key.
- Dữ liệu thật InfoSecCAT (skip nếu thiếu), mặc định phải cho đúng:
  n_students = 273, n_excluded = 3 (S050, S179, S225), số ô bỏ trống còn lại = 7,
  kr20 ≈ 0.7339 (sai số 1e-3), nhóm 27% = 74 người, 23 câu review_flag,
  độ khó 1 too_hard / 9 suitable / 40 too_easy (suitable: 7, 18, 22, 25, 31, 34, 36, 44, 50).
- Với exclude_blank_students=false: n_students = 276, kr20 ≈ 0.8578, 6 câu review_flag.
- Chạy cả bộ test; báo kết quả + giả định, dưới 10 dòng. Không dùng --force; làm trên nhánh task-t0-lam-sach (tạo từ nhánh hiện tại), không chuyển sang nhánh nào khác.