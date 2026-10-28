"""CTT (lý thuyết trắc nghiệm cổ điển): độ khó p và phân loại độ khó câu."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import CTTThresholds

TOO_HARD = "too_hard"
SUITABLE = "suitable"
TOO_EASY = "too_easy"
_BOUNDARY_TOL = 1e-9


def _check_scored(scored: pd.DataFrame) -> None:
    """Kiểm tra ma trận chấm điểm: không rỗng và chỉ chứa 0/1.

    Raises:
        ValueError: nếu ma trận rỗng hoặc có giá trị khác 0/1.
    """
    if scored.empty:
        raise ValueError("Ma trận chấm điểm rỗng")
    if not scored.isin([0, 1]).to_numpy().all():
        raise ValueError("Ma trận chấm điểm chỉ được chứa 0 và 1")


def item_difficulty(scored: pd.DataFrame) -> pd.Series:
    """Độ khó p của từng câu: số thí sinh đúng / số thí sinh.

    Args:
        scored: ma trận 0/1 (hàng = thí sinh, cột = câu), ô bỏ trống đã tính là 0.

    Returns:
        Series tên ``p``, index là mã câu (giữ thứ tự cột). p càng cao thì câu càng dễ.

    Raises:
        ValueError: nếu ma trận rỗng hoặc có giá trị khác 0/1.
    """
    _check_scored(scored)
    return scored.mean(axis=0).astype(float).rename("p")


def classify_difficulty(p: pd.Series, thresholds: CTTThresholds | None = None) -> pd.Series:
    """Phân loại độ khó theo ngưỡng ``p_min``, ``p_max``.

    - ``too_hard``: p < p_min
    - ``suitable``: p_min <= p <= p_max (so sánh có dung sai 1e-9 ở hai biên)
    - ``too_easy``: p > p_max

    Args:
        p: độ khó từng câu (kết quả của :func:`item_difficulty`).
        thresholds: ngưỡng; mặc định lấy ``CTTThresholds()`` (0,3 và 0,7).

    Returns:
        Series tên ``difficulty_class`` cùng index với ``p``.
    """
    t = thresholds or CTTThresholds()
    values = p.to_numpy(dtype=float)
    labels = np.where(
        values < t.p_min - _BOUNDARY_TOL,
        TOO_HARD,
        np.where(values > t.p_max + _BOUNDARY_TOL, TOO_EASY, SUITABLE),
    )
    return pd.Series(labels, index=p.index, name="difficulty_class")