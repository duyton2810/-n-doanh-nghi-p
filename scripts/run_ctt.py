"""Chạy phân tích CTT từ dòng lệnh và xuất kết quả ra thư mục.

Ví dụ:
    python -m scripts.run_ctt --data-dir data/raw/infoseccat --out-dir out/ctt --compare
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

from core.config import CTTThresholds, load_config
from core.ctt import CTTReport, ctt_report
from core.io import dataset_hash, load_key, load_responses

RESPONSES_FILE = "responses.csv"
KEY_FILE = "key.csv"
ITEMS_FILE = "ctt_items.csv"
SUMMARY_FILE = "ctt_summary.json"
COMPARE_FILE = "ctt_compare.csv"
CLEANING_FILE = "cleaning_removed.csv"
COMPARE_COLUMNS = ("question_id", "p_keep", "p_exclude", "r_pbis_keep", "r_pbis_exclude",
                   "flag_keep", "flag_exclude")


def build_parser() -> argparse.ArgumentParser:
    """Tạo bộ phân tích tham số dòng lệnh (``--exclude-blank`` và ``--keep-blank`` loại trừ nhau)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help=f"Thư mục chứa {RESPONSES_FILE}, {KEY_FILE}")
    parser.add_argument("--out-dir", type=Path, required=True, help="Thư mục xuất kết quả (tự tạo nếu chưa có)")
    parser.add_argument("--config", type=Path, default=None, help="File cấu hình TOML")
    blank = parser.add_mutually_exclusive_group()
    blank.add_argument("--exclude-blank", dest="exclude_blank", action="store_const", const=True,
                       help="Loại thí sinh bỏ trống toàn bộ (ghi đè cấu hình)")
    blank.add_argument("--keep-blank", dest="exclude_blank", action="store_const", const=False,
                       help="Giữ thí sinh bỏ trống toàn bộ (ghi đè cấu hình)")
    parser.add_argument("--compare", action="store_true",
                        help=f"Chạy cả hai chế độ và ghi {COMPARE_FILE} các câu khác cờ")
    return parser


def compare_flags(keep: CTTReport, exclude: CTTReport) -> pd.DataFrame:
    """So sánh hai báo cáo (giữ / loại thí sinh bỏ trống), chỉ giữ các câu có ``review_flag`` khác nhau.

    Args:
        keep: báo cáo khi giữ thí sinh bỏ trống toàn bộ.
        exclude: báo cáo khi loại thí sinh bỏ trống toàn bộ.

    Returns:
        DataFrame với các cột ``COMPARE_COLUMNS`` (``question_id`` là cột, không phải index),
        theo thứ tự câu của ``keep``.
    """
    a, b = keep.items, exclude.items.reindex(keep.items.index)
    table = pd.DataFrame({
        "question_id": a.index.to_numpy(dtype=object),
        "p_keep": a["p"].to_numpy(),
        "p_exclude": b["p"].to_numpy(),
        "r_pbis_keep": a["r_pbis"].to_numpy(),
        "r_pbis_exclude": b["r_pbis"].to_numpy(),
        "flag_keep": a["review_flag"].to_numpy(dtype=bool),
        "flag_exclude": b["review_flag"].to_numpy(dtype=bool),
    })
    return table.loc[table["flag_keep"] != table["flag_exclude"]].reset_index(drop=True)


def summary_lines(summary: dict[str, object]) -> list[str]:
    """Sáu dòng tóm tắt: số thí sinh đã loại khi làm sạch, N, số thí sinh bỏ trống, KR-20,
    số câu bị cờ, số câu theo độ khó."""
    counts = summary["difficulty_counts"]
    removed = summary["cleaning_removed"]
    return [
        f"Đã loại khi làm sạch: {len(removed)} thí sinh" + (f" ({', '.join(removed)})" if removed else ""),
        f"N: {summary['n_students']}",
        f"Thí sinh bỏ trống toàn bộ: {summary['n_blank_students']} "
        f"({'đã loại' if summary['blank_students_excluded'] else 'giữ lại'})",
        f"KR-20: {summary['kr20']:.4f}",
        f"Số câu bị cờ: {summary['n_flagged']}/{summary['n_items']}",
        "Số câu theo độ khó: " + ", ".join(f"{k}={v}" for k, v in counts.items()),
    ]


def run(args: argparse.Namespace) -> int:
    """Thực hiện phân tích theo tham số đã phân tích; ném lỗi để :func:`main` xử lý.

    Raises:
        FileNotFoundError: thiếu file dữ liệu hoặc cấu hình.
        ValueError: dữ liệu/cấu hình không hợp lệ.
    """
    responses_path = args.data_dir / RESPONSES_FILE
    key_path = args.data_dir / KEY_FILE
    for path in (responses_path, key_path):
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file {path}")
    if args.config is not None and not args.config.is_file():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình {args.config}")

    thresholds: CTTThresholds = load_config(args.config).ctt
    if args.exclude_blank is not None:
        thresholds = dataclasses.replace(thresholds, exclude_blank_students=args.exclude_blank)

    responses = load_responses(responses_path)
    key = load_key(key_path)
    report = ctt_report(responses, key, thresholds)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report.items.to_csv(args.out_dir / ITEMS_FILE, index_label="question_id", encoding="utf-8")
    report.cleaning.removed.to_csv(args.out_dir / CLEANING_FILE, index=False, encoding="utf-8")
    summary = {
        **report.summary,
        "dataset_hash": dataset_hash([responses_path, key_path]),
        "thresholds": dataclasses.asdict(thresholds),
    }
    with (args.out_dir / SUMMARY_FILE).open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    for line in summary_lines(report.summary):
        print(line)

    if args.compare:
        other = ctt_report(
            responses, key, dataclasses.replace(thresholds, exclude_blank_students=not thresholds.exclude_blank_students)
        )
        keep, exclude = (other, report) if thresholds.exclude_blank_students else (report, other)
        diff = compare_flags(keep, exclude)
        diff.to_csv(args.out_dir / COMPARE_FILE, index=False, encoding="utf-8")
        print(f"Số câu khác cờ giữa giữ/loại thí sinh bỏ trống: {len(diff)}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Điểm vào dòng lệnh; trả về mã thoát (0 = thành công, 1 = lỗi dữ liệu/file).

    Lỗi tham số do argparse xử lý (thoát mã 2 qua ``SystemExit``).
    """
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except FileNotFoundError as exc:
        message = f"không tìm thấy file {exc.filename}" if exc.filename else str(exc)
        print(f"Lỗi: {message}", file=sys.stderr)
    except (ValueError, OSError) as exc:
        print(f"Lỗi: dữ liệu hoặc cấu hình không hợp lệ: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    # console/redirect trên Windows có thể không phải UTF-8; tránh lỗi khi in tiếng Việt
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    raise SystemExit(main())
