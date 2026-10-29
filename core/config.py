"""Cấu hình ứng dụng: đường dẫn dữ liệu và ngưỡng CTT (không hardcode)."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ENV_DATA_DIR = "ITEM_ANALYSIS_DATA_DIR"


@dataclass(frozen=True)
class CTTThresholds:
    """Ngưỡng phân tích CTT theo bài báo; mọi giá trị đều cấu hình được."""

    p_min: float = 0.3
    p_max: float = 0.7
    group_fraction: float = 0.27
    r_pbis_acceptable: float = 0.2
    r_pbis_good: float = 0.4
    exclude_item_from_total: bool = True
    exclude_blank_students: bool = True
    warn_omit_rate: float = 0.5

    def __post_init__(self) -> None:
        """Kiểm tra tính hợp lệ của các ngưỡng."""
        if not 0.0 <= self.p_min < self.p_max <= 1.0:
            raise ValueError("Cần 0 <= p_min < p_max <= 1")
        if not 0.0 < self.group_fraction < 0.5:
            raise ValueError("group_fraction phải nằm trong (0, 0.5)")
        if not self.r_pbis_acceptable < self.r_pbis_good:
            raise ValueError("Cần r_pbis_acceptable < r_pbis_good")
        if not 0.0 < self.warn_omit_rate <= 1.0:
            raise ValueError("warn_omit_rate phải nằm trong (0, 1]")


@dataclass(frozen=True)
class AppConfig:
    """Cấu hình chung của ứng dụng."""

    data_dir: Path = Path("data")
    ctt: CTTThresholds = field(default_factory=CTTThresholds)


def load_config(path: Path | None = None) -> AppConfig:
    """Đọc cấu hình từ file TOML (nếu có) và biến môi trường.

    Đường dẫn tương đối được tính từ thư mục chứa file cấu hình
    (hoặc thư mục làm việc hiện tại nếu không truyền ``path``).
    Biến môi trường ``ITEM_ANALYSIS_DATA_DIR`` ghi đè ``paths.data_dir``.

    Raises:
        ValueError: nếu có khóa ``[ctt]`` lạ hoặc ngưỡng không hợp lệ.
    """
    raw: dict = {}
    base = Path.cwd()
    if path is not None:
        path = Path(path)
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
        base = path.resolve().parent
    data_dir = Path(os.environ.get(ENV_DATA_DIR) or raw.get("paths", {}).get("data_dir", "data"))
    if not data_dir.is_absolute():
        data_dir = base / data_dir
    try:
        ctt = CTTThresholds(**raw.get("ctt", {}))
    except TypeError as exc:
        raise ValueError(f"Khóa không hợp lệ trong [ctt]: {exc}") from exc
    return AppConfig(data_dir=data_dir, ctt=ctt)
