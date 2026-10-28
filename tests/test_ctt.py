"""Test cho core.ctt (T2.1: độ khó p và phân loại)."""
import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import CTTThresholds, load_config
from core.ctt import (SUITABLE, TOO_EASY, TOO_HARD, classify_difficulty,
                      item_difficulty)
from core.io import load_key, load_responses
from core.scoring import filter_blank_students, score_matrix


def _scored() -> pd.DataFrame:
    """10 thí sinh x 4 câu với p = 0,1 / 0,3 / 0,7 / 0,9 (đếm đúng 1, 3, 7, 9)."""
    counts = [1, 3, 7, 9]
    data = np.zeros((10, 4), dtype=np.int8)
    for j, c in enumerate(counts):
        data[:c, j] = 1
    return pd.DataFrame(
        data,
        index=pd.Index([f"S{i}" for i in range(10)], name="student"),
        columns=pd.Index(["q1", "q2", "q3", "q4"], name="question_id"),
    )


class TestItemDifficulty(unittest.TestCase):
    """item_difficulty."""

    def test_values_name_and_order(self) -> None:
        p = item_difficulty(_scored())
        self.assertEqual(p.name, "p")
        self.assertEqual(list(p.index), ["q1", "q2", "q3", "q4"])
        np.testing.assert_allclose(p.to_numpy(), [0.1, 0.3, 0.7, 0.9])

    def test_all_correct_and_all_wrong(self) -> None:
        df = pd.DataFrame({"a": [1, 1, 1], "b": [0, 0, 0]})
        np.testing.assert_allclose(item_difficulty(df).to_numpy(), [1.0, 0.0])

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            item_difficulty(pd.DataFrame())

    def test_non_binary_raises(self) -> None:
        with self.assertRaises(ValueError):
            item_difficulty(pd.DataFrame({"a": [0, 1, 2]}))


class TestClassifyDifficulty(unittest.TestCase):
    """classify_difficulty."""

    def test_default_thresholds_including_boundaries(self) -> None:
        cls = classify_difficulty(item_difficulty(_scored()))
        self.assertEqual(cls.name, "difficulty_class")
        self.assertEqual(list(cls), [TOO_HARD, SUITABLE, SUITABLE, TOO_EASY])

    def test_custom_thresholds(self) -> None:
        t = CTTThresholds(p_min=0.2, p_max=0.8)
        cls = classify_difficulty(item_difficulty(_scored()), t)
        self.assertEqual(list(cls), [TOO_HARD, SUITABLE, SUITABLE, TOO_EASY])
        t2 = CTTThresholds(p_min=0.05, p_max=0.95)
        cls2 = classify_difficulty(item_difficulty(_scored()), t2)
        self.assertEqual(set(cls2), {SUITABLE})

    def test_float_noise_at_boundary_is_suitable(self) -> None:
        p = pd.Series([0.3 - 1e-12, 0.7 + 1e-12], index=["a", "b"])
        self.assertEqual(list(classify_difficulty(p)), [SUITABLE, SUITABLE])

    def test_keeps_index(self) -> None:
        p = pd.Series([0.5], index=["x"])
        self.assertEqual(list(classify_difficulty(p).index), ["x"])


def _data_dir() -> Path:
    env = os.environ.get("INFOSECCAT_DIR")
    return Path(env) if env else load_config().data_dir / "raw" / "infoseccat"


@unittest.skipUnless((_data_dir() / "responses.csv").exists(), "chưa có dữ liệu InfoSecCAT")
class TestRealData(unittest.TestCase):
    """Độ khó trên InfoSecCAT."""

    @classmethod
    def setUpClass(cls) -> None:
        key = load_key(_data_dir() / "key.csv")
        cls.responses = load_responses(_data_dir() / "responses.csv")
        cls.scored_all = score_matrix(cls.responses, key)
        kept, _ = filter_blank_students(cls.responses)
        cls.scored_kept = score_matrix(kept, key)

    def test_all_276_students_nine_suitable_items(self) -> None:
        cls = classify_difficulty(item_difficulty(self.scored_all))
        self.assertEqual(int((cls == SUITABLE).sum()), 9)

    def test_p_in_unit_interval_and_50_items(self) -> None:
        for scored in (self.scored_all, self.scored_kept):
            p = item_difficulty(scored)
            self.assertEqual(len(p), 50)
            self.assertTrue(p.between(0, 1).all())

    def test_removing_blank_students_never_lowers_p(self) -> None:
        # ba thí sinh bị loại có điểm 0 nên p chỉ có thể tăng
        p_all = item_difficulty(self.scored_all)
        p_kept = item_difficulty(self.scored_kept)
        self.assertTrue((p_kept >= p_all - 1e-12).all())


if __name__ == "__main__":
    unittest.main()