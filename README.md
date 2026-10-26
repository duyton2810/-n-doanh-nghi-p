# item-analysis

Phân tích & chuẩn hóa câu trắc nghiệm (CTT, IRT, BKT). Python 3.12.

## Chạy
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    # đặt dữ liệu vào data/raw/infoseccat/ (responses.csv, key.csv, LICENSE, CITATION.cff)
    python -m unittest discover -s tests -t . -v
    # hoặc: pip install pytest && pytest

Dữ liệu: Open InfoSecCAT (MIT) - xem CITATION.cff đi kèm bộ dữ liệu để trích dẫn.
