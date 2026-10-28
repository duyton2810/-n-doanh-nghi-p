"""Chấm điểm: ma trận lựa chọn -> ma trận 0/1 (ô trống tính là sai)."""
from __future__ import annotations

import string

import numpy as np
import pandas as pd


def validate_responses(responses: pd.DataFrame, key: pd.DataFrame) -> None:
    """Kiểm tra responses khớp với key.

    Raises:
        ValueError: tập mã câu hai bên khác nhau, hoặc có lựa chọn ngoài
            ``""`` và các phương án hợp lệ của câu đó.
    """
    only_resp = set(responses.columns) - set(key.index)
    only_key = set(key.index) - set(responses.columns)
    if only_resp or only_key:
        raise ValueError(
            f"Mã câu không khớp: chỉ có trong responses={sorted(only_resp)}, "
            f"chỉ có trong key={sorted(only_key)}"
        )
    bad: list[tuple[str, str]] = []
    for qid, n in key["num_choices"].items():
        allowed = ["", *string.ascii_lowercase[:n]]
        invalid = responses[qid][~responses[qid].isin(allowed)]
        bad.extend((qid, str(v)) for v in invalid.unique())
    if bad:
        raise ValueError(f"Lựa chọn không hợp lệ (câu, giá trị): {bad[:10]}")


def score_matrix(responses: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    """Chấm điểm: 1 nếu bằng đáp án đúng, 0 nếu sai hoặc bỏ trống.

    Cột kết quả được sắp theo thứ tự của ``key``.

    Returns:
        DataFrame int8, cùng index ``student``, cột theo ``key.index``.
    """
    validate_responses(responses, key)
    ordered = responses.loc[:, key.index]
    correct = ordered.to_numpy(dtype=object) == key["correct_answer"].to_numpy(dtype=object)[None, :]
    return pd.DataFrame(
        correct.astype(np.int8), index=ordered.index, columns=ordered.columns
    )


def total_scores(scored: pd.DataFrame) -> pd.Series:
    """Tổng số câu đúng của mỗi thí sinh."""
    return scored.sum(axis=1).astype(int).rename("total")


def omit_counts(responses: pd.DataFrame) -> pd.Series:
    """Số thí sinh bỏ trống ở mỗi câu."""
    return (responses == "").sum(axis=0).rename("omitted")


def omit_rate(responses: pd.DataFrame) -> pd.Series:
    """Tỉ lệ thí sinh bỏ trống ở mỗi câu (0..1)."""
    return (omit_counts(responses) / len(responses)).rename("omit_rate")


def blank_students(responses: pd.DataFrame) -> pd.Index:
    """Mã các thí sinh bỏ trống toàn bộ câu hỏi (mọi ô đều là ``""``)."""
    return responses.index[(responses == "").all(axis=1)]


def filter_blank_students(
    responses: pd.DataFrame, exclude: bool = True
) -> tuple[pd.DataFrame, pd.Index]:
    """Loại (hoặc giữ) các thí sinh bỏ trống toàn bộ.

    Args:
        responses: ma trận lựa chọn, ô trống là ``""``.
        exclude: True thì loại các thí sinh bỏ trống toàn bộ; False thì giữ nguyên.

    Returns:
        (responses sau lọc, mã các thí sinh đã bị loại). Khi ``exclude=False``
        phần thứ hai là Index rỗng. Dữ liệu đầu vào không bị sửa.

    Raises:
        ValueError: nếu sau khi loại không còn thí sinh nào.
    """
    removed = blank_students(responses) if exclude else responses.index[:0]
    kept = responses.drop(index=removed)
    if kept.empty:
        raise ValueError("Không còn thí sinh nào sau khi loại thí sinh bỏ trống toàn bộ")
    return kept, removed