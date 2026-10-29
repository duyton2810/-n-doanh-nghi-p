"""Làm sạch dữ liệu: bước đứng trước mọi phân tích.

Quy trình: nạp -> làm sạch -> chấm điểm -> phân tích. Mọi phân tích (CTT, sau này
distractor, IRT, BKT) chỉ nhận dữ liệu từ :func:`prepare_dataset`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from core.config import CTTThresholds
from core.scoring import blank_students, score_matrix, validate_responses

REASON_ALL_BLANK = "bỏ trống toàn bộ"
REMOVED_COLUMNS = ("student", "reason", "n_answered")


@dataclass(frozen=True)
class CleaningReport:
    """Kết quả làm sạch.

    Attributes:
        n_before: số thí sinh trước khi làm sạch.
        n_after: số thí sinh còn lại.
        removed: bảng thí sinh bị loại, cột ``student, reason, n_answered``.
        warnings: cảnh báo (không loại), ví dụ thí sinh bỏ trống nhiều câu.
    """

    n_before: int
    n_after: int
    removed: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PreparedDataset:
    """Dữ liệu sẵn sàng cho phân tích.

    Attributes:
        responses: ma trận lựa chọn đã làm sạch (cột theo thứ tự key).
        scored: ma trận 0/1 (int8) của các thí sinh còn lại, cột theo thứ tự key.
        report: :class:`CleaningReport`.
    """

    responses: pd.DataFrame
    scored: pd.DataFrame
    report: CleaningReport


def clean_responses(
    responses: pd.DataFrame, thresholds: CTTThresholds | None = None
) -> tuple[pd.DataFrame, CleaningReport]:
    """Làm sạch ma trận lựa chọn.

    - Loại thí sinh bỏ trống toàn bộ khi ``thresholds.exclude_blank_students``
      (lý do ``"bỏ trống toàn bộ"``, ``n_answered = 0``).
    - Cảnh báo (không loại) thí sinh còn lại có tỉ lệ bỏ trống ``>= thresholds.warn_omit_rate``.

    Args:
        responses: ma trận lựa chọn (index = student, cột = question_id, ô trống ``""``).
            Không bị sửa.
        thresholds: ngưỡng; mặc định ``CTTThresholds()``.

    Returns:
        (responses đã làm sạch, :class:`CleaningReport`).

    Raises:
        ValueError: nếu loại hết thí sinh.
    """
    t = thresholds or CTTThresholds()
    removed_ids = blank_students(responses) if t.exclude_blank_students else responses.index[:0]
    kept = responses.drop(index=removed_ids)
    if kept.empty:
        raise ValueError("Không còn thí sinh nào sau khi làm sạch (tất cả bỏ trống toàn bộ)")
    removed = pd.DataFrame(
        {
            "student": removed_ids.to_numpy(dtype=object),
            "reason": [REASON_ALL_BLANK] * len(removed_ids),
            "n_answered": [0] * len(removed_ids),
        },
        columns=list(REMOVED_COLUMNS),
    )
    n_items = kept.shape[1]
    omitted = (kept == "").sum(axis=1)
    warnings = [
        f"{student}: bỏ trống {int(n)}/{n_items} câu ({n / n_items:.0%})"
        for student, n in omitted.items()
        if n_items and n / n_items >= t.warn_omit_rate
    ]
    report = CleaningReport(
        n_before=len(responses), n_after=len(kept), removed=removed, warnings=warnings
    )
    return kept, report


def prepare_dataset(
    responses: pd.DataFrame, key: pd.DataFrame, thresholds: CTTThresholds | None = None
) -> PreparedDataset:
    """Lối vào duy nhất của mọi phân tích: kiểm tra -> làm sạch -> chấm điểm.

    Args:
        responses: ma trận lựa chọn thô. Không bị sửa.
        key: đáp án (index question_id, cột ``correct_answer``, ``num_choices``).
        thresholds: ngưỡng; mặc định ``CTTThresholds()``.

    Returns:
        :class:`PreparedDataset`; cột của ``responses`` và ``scored`` theo thứ tự key.

    Raises:
        ValueError: responses không khớp key (kiểm tra trước khi làm sạch) hoặc loại hết thí sinh.
    """
    validate_responses(responses, key)
    cleaned, report = clean_responses(responses, thresholds)
    cleaned = cleaned.loc[:, key.index]
    return PreparedDataset(responses=cleaned, scored=score_matrix(cleaned, key), report=report)
