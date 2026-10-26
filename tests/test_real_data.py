"""Kiểm thử tích hợp trên Open InfoSecCAT (tự bỏ qua nếu chưa có dữ liệu).

Đặt biến môi trường INFOSECCAT_DIR trỏ tới thư mục chứa responses.csv và key.csv,
hoặc đặt dữ liệu tại <data_dir>/raw/infoseccat.
"""
import os
import unittest
from pathlib import Path

from core.config import load_config
from core.io import load_key, load_responses
from core.scoring import omit_counts, score_matrix, total_scores


def _data_dir() -> Path:
    env = os.environ.get("INFOSECCAT_DIR")
    return Path(env) if env else load_config().data_dir / "raw" / "infoseccat"


@unittest.skipUnless((_data_dir() / "responses.csv").exists(), "chưa có dữ liệu InfoSecCAT")
class TestInfoSecCAT(unittest.TestCase):
    """Kiểm tra các con số mô tả trong README của bộ dữ liệu."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.key = load_key(_data_dir() / "key.csv")
        cls.responses = load_responses(_data_dir() / "responses.csv")
        cls.scored = score_matrix(cls.responses, cls.key)

    def test_shapes(self) -> None:
        self.assertEqual(self.responses.shape, (276, 50))
        self.assertEqual(len(self.key), 50)
        self.assertEqual(set(self.key["num_choices"]), {4})

    def test_blank_cells_count(self) -> None:
        self.assertEqual(int(omit_counts(self.responses).sum()), 157)

    def test_scores_in_range(self) -> None:
        totals = total_scores(self.scored)
        self.assertTrue(totals.between(0, 50).all())


if __name__ == "__main__":
    unittest.main()
