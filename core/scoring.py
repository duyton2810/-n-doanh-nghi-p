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
