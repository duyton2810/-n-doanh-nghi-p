"""Tải bộ Open InfoSecCAT về thư mục dữ liệu (không hardcode URL/đường dẫn).

Ví dụ:
    python -m scripts.fetch_dataset --base-url https://<nơi-lưu>/ --dest data/raw/infoseccat
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from core.io import download_file, sha256_file

DEFAULT_FILES = ("responses.csv", "key.csv", "LICENSE", "CITATION.cff", "README.md")


def main(argv: Sequence[str] | None = None) -> int:
    """Điểm vào dòng lệnh; trả về mã thoát (0 = thành công)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="URL gốc chứa các file")
    parser.add_argument("--dest", type=Path, required=True, help="Thư mục lưu")
    parser.add_argument("--files", nargs="+", default=list(DEFAULT_FILES))
    args = parser.parse_args(argv)
    for name in args.files:
        url = args.base_url.rstrip("/") + "/" + name
        path = download_file(url, args.dest / name)
        print(f"{name}  sha256={sha256_file(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
