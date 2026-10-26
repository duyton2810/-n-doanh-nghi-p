"""Đọc dữ liệu thô (responses/key), băm dataset và tải file về."""
from __future__ import annotations

import hashlib
import shutil
import string
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Sequence

import pandas as pd

STUDENT_COLUMN = "student"
KEY_COLUMNS = ("question_id", "correct_answer", "num_choices")
ALLOWED_URL_SCHEMES = ("https", "file")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Trả về SHA-256 (hex) của một file, đọc theo từng khối."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_hash(paths: Sequence[Path]) -> str:
    """Băm một dataset gồm nhiều file (nhạy với thứ tự truyền vào).

    Dùng để lưu vào bảng ``runs`` ở giai đoạn lưu lịch sử phân tích.
    """
    joined = "".join(sha256_file(p) for p in paths)
    return hashlib.sha256(joined.encode("ascii")).hexdigest()


def load_key(path: Path) -> pd.DataFrame:
    """Đọc file đáp án (cột ``question_id, correct_answer, num_choices``).

    Returns:
        DataFrame index là ``question_id`` (chuỗi, giữ thứ tự file), cột
        ``correct_answer`` (chữ thường) và ``num_choices`` (int).

    Raises:
        ValueError: thiếu cột, mã câu rỗng/trùng, số phương án không hợp lệ,
            hoặc đáp án nằm ngoài các phương án của câu.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in KEY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"key thiếu cột: {missing}")
    df = df.loc[:, list(KEY_COLUMNS)].apply(lambda s: s.str.strip())
    if (df["question_id"] == "").any():
        raise ValueError("key có question_id rỗng")
    if df["question_id"].duplicated().any():
        raise ValueError("key có question_id trùng")
    try:
        df["num_choices"] = df["num_choices"].astype(int)
    except ValueError as exc:
        raise ValueError("num_choices phải là số nguyên") from exc
    if not df["num_choices"].between(2, len(string.ascii_lowercase)).all():
        raise ValueError("num_choices phải nằm trong [2, 26]")
    df["correct_answer"] = df["correct_answer"].str.lower()
    for qid, ans, n in zip(df["question_id"], df["correct_answer"], df["num_choices"]):
        if ans not in string.ascii_lowercase[:n]:
            raise ValueError(f"Câu {qid}: đáp án '{ans}' ngoài {n} phương án")
    return df.set_index("question_id")


def load_responses(path: Path) -> pd.DataFrame:
    """Đọc ma trận lựa chọn (cột ``student`` + một cột cho mỗi câu).

    Giá trị được chuẩn hóa về chữ thường, bỏ khoảng trắng; ô trống là ``""``
    (bỏ trống). Không kiểm tra phương án hợp lệ ở đây (xem
    :func:`core.scoring.validate_responses`).

    Returns:
        DataFrame index ``student``, cột là mã câu (chuỗi).

    Raises:
        ValueError: thiếu cột ``student``, mã học sinh rỗng/trùng, hoặc không có cột câu.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    if STUDENT_COLUMN not in df.columns:
        raise ValueError(f"responses thiếu cột '{STUDENT_COLUMN}'")
    df = df.apply(lambda s: s.str.strip())
    student = df[STUDENT_COLUMN]
    if (student == "").any():
        raise ValueError("responses có mã student rỗng")
    if student.duplicated().any():
        raise ValueError("responses có mã student trùng")
    items = df.drop(columns=STUDENT_COLUMN)
    if items.shape[1] == 0:
        raise ValueError("responses không có cột câu hỏi")
    items = items.apply(lambda s: s.str.lower())
    items.index = pd.Index(student.to_numpy(dtype=object), name=STUDENT_COLUMN)
    items.columns.name = "question_id"
    return items


def download_file(
    url: str,
    dest: Path,
    expected_sha256: str | None = None,
    timeout: float = 60.0,
) -> Path:
    """Tải một file về ``dest`` (ghi tạm rồi đổi tên), tùy chọn kiểm SHA-256.

    Chỉ chấp nhận ``https`` và ``file``.

    Raises:
        ValueError: scheme không được phép hoặc SHA-256 không khớp
            (file tạm bị xóa, ``dest`` không bị ghi).
    """
    scheme = urllib.parse.urlparse(url).scheme.lower()
    if scheme not in ALLOWED_URL_SCHEMES:
        raise ValueError(f"Scheme '{scheme}' không được phép; chỉ {ALLOWED_URL_SCHEMES}")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out)
    if expected_sha256 is not None and sha256_file(tmp) != expected_sha256.lower():
        tmp.unlink()
        raise ValueError(f"SHA-256 không khớp cho {url}")
    tmp.replace(dest)
    return dest
