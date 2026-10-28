"""CTT (lý thuyết trắc nghiệm cổ điển): độ khó p, phân loại độ khó, độ phân biệt D, r_pbis, KR-20."""
from __future__ import annotations

import math

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


def split_groups(total: pd.Series, fraction: float) -> tuple[pd.Index, pd.Index]:
    """Chia nhóm cao và nhóm thấp theo tổng điểm (phương pháp nhóm 27%).

    Mỗi nhóm có ``n = floor(fraction * N + 0.5)`` thí sinh (làm tròn nửa lên, không
    dùng ``round()`` kiểu làm tròn về số chẵn của Python), cộng dung sai 1e-9 để
    tránh sai số dấu phẩy động khi ``fraction * N`` đúng bằng x,5.

    Xử lý hòa điểm ở ranh giới: thí sinh được xếp hạng giảm dần theo tổng điểm bằng
    một lần sắp xếp ổn định (stable sort); những người bằng điểm giữ thứ tự xuất hiện
    trong dữ liệu (người xuất hiện trước xếp hạng cao hơn). Nhóm cao là ``n`` người đầu
    bảng xếp hạng, nhóm thấp là ``n`` người cuối bảng. Vì vậy kết quả lặp lại được
    với cùng dữ liệu, nhưng phụ thuộc thứ tự hàng khi có hòa điểm ở ranh giới.

    Args:
        total: tổng điểm từng thí sinh (index = student, không trùng, không thiếu giá trị).
        fraction: tỉ lệ mỗi nhóm, ví dụ 0,27.

    Returns:
        ``(high, low)``: index thí sinh nhóm cao (xếp theo hạng từ cao xuống) và
        nhóm thấp (xếp theo hạng, người thấp nhất ở cuối).

    Raises:
        ValueError: nếu ``total`` có giá trị thiếu hoặc index trùng, nếu ``n < 1``,
            hoặc nếu hai nhóm trùng nhau (``2n > N``).
    """
    if total.isna().any():
        raise ValueError("Tổng điểm có giá trị thiếu")
    if not total.index.is_unique:
        raise ValueError("Index thí sinh bị trùng")
    size = len(total)
    n = math.floor(fraction * size + 0.5 + _BOUNDARY_TOL)
    if n < 1:
        raise ValueError(f"Nhóm có {n} thí sinh (N={size}, fraction={fraction}); cần ít nhất 1")
    if 2 * n > size:
        raise ValueError(f"Hai nhóm {n} thí sinh trùng nhau (N={size})")
    order = np.argsort(-total.to_numpy(dtype=float), kind="stable")
    ranked = total.index[order]
    return ranked[:n], ranked[size - n:]


def discrimination_index(scored: pd.DataFrame, thresholds: CTTThresholds | None = None) -> pd.Series:
    """Độ phân biệt D: tỉ lệ đúng nhóm cao trừ tỉ lệ đúng nhóm thấp, cho từng câu.

    Tổng điểm là ``scored.sum(axis=1)`` (gồm cả câu đang xét); nhóm chia bằng
    :func:`split_groups` với ``thresholds.group_fraction``. D nằm trong [-1, 1].

    Args:
        scored: ma trận 0/1 (hàng = thí sinh, cột = câu).
        thresholds: ngưỡng; mặc định lấy ``CTTThresholds()`` (nhóm 27%).

    Returns:
        Series tên ``D``, index là mã câu (giữ thứ tự cột).

    Raises:
        ValueError: nếu ma trận không hợp lệ hoặc quá ít thí sinh để chia nhóm.
    """
    _check_scored(scored)
    t = thresholds or CTTThresholds()
    high, low = split_groups(scored.sum(axis=1), t.group_fraction)
    d = scored.loc[high].mean(axis=0) - scored.loc[low].mean(axis=0)
    return d.astype(float).rename("D")


def kr20(scored: pd.DataFrame, ddof: int = 0) -> float:
    """Hệ số tin cậy KR-20 của cả đề.

    ``KR-20 = L / (L - 1) * (1 - sum(p_j * q_j) / SD^2)`` với ``q_j = 1 - p_j``,
    ``L`` là số câu, ``SD^2`` là phương sai tổng điểm ``scored.sum(axis=1)``.
    ``p_j * q_j`` luôn tính theo tỉ lệ (không phụ thuộc ``ddof``); chỉ ``SD^2`` dùng ``ddof``.

    Args:
        scored: ma trận 0/1 (hàng = thí sinh, cột = câu).
        ddof: 0 (phương sai tổng thể, chia N) hoặc 1 (phương sai mẫu, chia N - 1).

    Returns:
        KR-20 dạng ``float``; tối đa 1, có thể âm khi các câu tương quan âm.

    Raises:
        ValueError: nếu ``ddof`` khác 0/1, ma trận không hợp lệ, ``L < 2``, hoặc
            ``SD^2`` bằng 0 / không xác định (ví dụ N = 1 với ``ddof = 1``).
    """
    if ddof not in (0, 1):
        raise ValueError(f"ddof phải là 0 hoặc 1, nhận {ddof}")
    _check_scored(scored)
    n_items = scored.shape[1]
    if n_items < 2:
        raise ValueError(f"KR-20 cần ít nhất 2 câu, nhận {n_items}")
    total_var = float(scored.sum(axis=1).var(ddof=ddof))
    if not math.isfinite(total_var) or total_var < _BOUNDARY_TOL:
        raise ValueError(f"Phương sai tổng điểm bằng 0 hoặc không xác định ({total_var})")
    p = item_difficulty(scored)
    sum_pq = float((p * (1 - p)).sum())
    return n_items / (n_items - 1) * (1 - sum_pq / total_var)


def point_biserial(
    scored: pd.DataFrame,
    thresholds: CTTThresholds | None = None,
    exclude_item: bool | None = None,
) -> pd.Series:
    """Hệ số tương quan điểm-nhị phân r_pbis của từng câu với tổng điểm.

    r_pbis là hệ số Pearson giữa cột câu (0/1) và tổng điểm thí sinh. Khi
    ``exclude_item`` là True, tổng điểm của mỗi câu là tổng trừ điểm chính câu đó
    (tổng hiệu chỉnh, tính riêng cho từng câu); khi False dùng tổng đầy đủ.

    Không tính được (câu mọi người cùng đúng hoặc cùng sai, hoặc tổng điểm dùng để
    tương quan là hằng số) thì trả NaN, không đặt 0.

    Args:
        scored: ma trận 0/1 (hàng = thí sinh, cột = câu).
        thresholds: ngưỡng; mặc định lấy ``CTTThresholds()``.
        exclude_item: None thì lấy ``thresholds.exclude_item_from_total`` (mặc định True).

    Returns:
        Series tên ``r_pbis``, index là mã câu (giữ thứ tự cột), giá trị trong
        [-1, 1] hoặc NaN.

    Raises:
        ValueError: nếu ma trận rỗng hoặc có giá trị khác 0/1.
    """
    _check_scored(scored)
    t = thresholds or CTTThresholds()
    if exclude_item is None:
        exclude_item = t.exclude_item_from_total
    items = scored.astype(float)
    total = items.sum(axis=1)
    # mỗi cột của ``other`` là tổng điểm dùng để tương quan với câu cùng tên
    if exclude_item:
        other = items.rsub(total, axis=0)
    else:
        full = np.broadcast_to(total.to_numpy()[:, None], items.shape)
        other = pd.DataFrame(full, index=items.index, columns=items.columns)
    # corrwith trả NaN (không cảnh báo) khi một trong hai cột có phương sai 0;
    # cột hằng số gồm số nguyên nên độ lệch khỏi trung bình đúng bằng 0
    r = items.corrwith(other, axis=0).reindex(scored.columns)
    return pd.Series(np.clip(r.to_numpy(dtype=float), -1.0, 1.0), index=scored.columns, name="r_pbis")
