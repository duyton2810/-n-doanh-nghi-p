"""Test cho core.ctt (T2.1: độ khó p và phân loại; T2.2: độ phân biệt D; T2.4: KR-20)."""
import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import CTTThresholds, load_config
from core.ctt import (SUITABLE, TOO_EASY, TOO_HARD, classify_difficulty,
                      discrimination_index, item_difficulty, kr20, split_groups)
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


def _hand_scored() -> pd.DataFrame:
    """10 thí sinh x 3 câu, có hòa điểm ở cả hai ranh giới nhóm."""
    rows = [
        [1, 1, 1],  # S0 tổng 3
        [1, 1, 0],  # S1 tổng 2
        [1, 0, 1],  # S2 tổng 2
        [0, 1, 1],  # S3 tổng 2
        [1, 0, 0],  # S4 tổng 1
        [0, 1, 0],  # S5 tổng 1
        [0, 0, 1],  # S6 tổng 1
        [1, 0, 0],  # S7 tổng 1
        [0, 0, 0],  # S8 tổng 0
        [0, 1, 0],  # S9 tổng 1
    ]
    return pd.DataFrame(
        np.array(rows, dtype=np.int8),
        index=pd.Index([f"S{i}" for i in range(10)], name="student"),
        columns=pd.Index(["q1", "q2", "q3"], name="question_id"),
    )


class TestSplitGroups(unittest.TestCase):
    """split_groups."""

    def _n(self, size: int, fraction: float = 0.27) -> int:
        high, low = split_groups(pd.Series(np.arange(size, dtype=float)), fraction)
        self.assertEqual(len(high), len(low))
        return len(high)

    def test_group_size_half_up(self) -> None:
        self.assertEqual(self._n(276), 75)  # 74,52 + 0,5 -> 75
        self.assertEqual(self._n(273), 74)  # 73,71 + 0,5 -> 74
        self.assertEqual(self._n(10), 3)  # 2,7 + 0,5 -> 3
        # round() của Python cho round(2.5) = 2; ở đây 0,25 * 10 = 2,5 phải lên 3
        self.assertEqual(self._n(10, 0.25), 3)

    def test_small_n_raises(self) -> None:
        with self.assertRaises(ValueError):
            split_groups(pd.Series([1.0]), 0.27)  # 0,27 + 0,5 -> n = 0
        with self.assertRaises(ValueError):
            split_groups(pd.Series([], dtype=float), 0.27)

    def test_overlapping_groups_raise(self) -> None:
        with self.assertRaises(ValueError):
            split_groups(pd.Series(np.arange(10, dtype=float)), 0.6)  # n = 6, 2n > 10

    def test_nan_and_duplicate_index_raise(self) -> None:
        with self.assertRaises(ValueError):
            split_groups(pd.Series([1.0, np.nan, 2.0, 3.0]), 0.27)
        with self.assertRaises(ValueError):
            split_groups(pd.Series([1.0, 2.0, 3.0, 4.0], index=["a", "a", "b", "c"]), 0.27)

    def test_ties_follow_data_order(self) -> None:
        total = _hand_scored().sum(axis=1)
        high, low = split_groups(total, 0.27)
        # xếp hạng ổn định: S0 | S1 S2 S3 | S4 S5 S6 S7 S9 | S8
        self.assertEqual(list(high), ["S0", "S1", "S2"])
        self.assertEqual(list(low), ["S7", "S9", "S8"])

    def test_ties_reproducible_and_input_unchanged(self) -> None:
        total = pd.Series([5, 3, 3, 3, 3, 1, 1, 1, 1, 0], index=list("abcdefghij"))
        before = total.copy()
        first = split_groups(total, 0.27)
        second = split_groups(total, 0.27)
        self.assertEqual([list(g) for g in first], [list(g) for g in second])
        pd.testing.assert_series_equal(total, before)
        self.assertEqual(set(first[0]) & set(first[1]), set())

    def test_all_equal_totals_still_disjoint(self) -> None:
        high, low = split_groups(pd.Series([2.0] * 10), 0.27)
        self.assertEqual(list(high), [0, 1, 2])
        self.assertEqual(list(low), [7, 8, 9])


class TestDiscriminationIndex(unittest.TestCase):
    """discrimination_index."""

    def test_hand_computed(self) -> None:
        # N = 10, n = floor(0,27 * 10 + 0,5) = 3.
        # Nhóm cao S0, S1, S2 (S3 cũng 2 điểm nhưng xuất hiện sau):
        #   q1 = 3/3, q2 = 2/3, q3 = 2/3
        # Nhóm thấp S7, S9, S8 (S4..S6 cũng 1 điểm nhưng xuất hiện trước nên xếp cao hơn):
        #   q1 = 1/3 (S7), q2 = 1/3 (S9), q3 = 0/3
        # D = q1: 1 - 1/3 = 2/3, q2: 2/3 - 1/3 = 1/3, q3: 2/3 - 0 = 2/3
        d = discrimination_index(_hand_scored())
        self.assertEqual(d.name, "D")
        self.assertEqual(list(d.index), ["q1", "q2", "q3"])
        np.testing.assert_allclose(d.to_numpy(), [2 / 3, 1 / 3, 2 / 3])

    def test_extremes_plus_and_minus_one(self) -> None:
        # N = 2, fraction 0,25: n = floor(0,5 + 0,5) = 1. Cao = A (tổng 3), thấp = B (tổng 1).
        df = pd.DataFrame({"x": [1, 0], "y": [1, 0], "z": [1, 0], "rev": [0, 1]},
                          index=["A", "B"])
        d = discrimination_index(df, CTTThresholds(group_fraction=0.25))
        np.testing.assert_allclose(d.to_numpy(), [1.0, 1.0, 1.0, -1.0])

    def test_all_correct_item_is_zero_and_bounds(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(20):
            data = rng.integers(0, 2, size=(40, 6)).astype(np.int8)
            data[:, 0] = 1
            df = pd.DataFrame(data, columns=[f"q{i}" for i in range(6)])
            d = discrimination_index(df)
            self.assertEqual(d["q0"], 0.0)
            self.assertTrue(d.between(-1, 1).all())

    def test_uses_threshold_fraction(self) -> None:
        df = _hand_scored()
        # fraction 0,1: n = floor(1 + 0,5) = 1, cao = S0, thấp = S8 -> D = 1 cho mọi câu
        d = discrimination_index(df, CTTThresholds(group_fraction=0.1))
        np.testing.assert_allclose(d.to_numpy(), [1.0, 1.0, 1.0])

    def test_input_unchanged(self) -> None:
        df = _hand_scored()
        before = df.copy()
        discrimination_index(df)
        pd.testing.assert_frame_equal(df, before)

    def test_invalid_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            discrimination_index(pd.DataFrame())
        with self.assertRaises(ValueError):
            discrimination_index(pd.DataFrame({"a": [0, 2]}))
        with self.assertRaises(ValueError):
            discrimination_index(pd.DataFrame({"a": [1]}))  # n = 0


def _kr20_reference(scored: pd.DataFrame, ddof: int) -> float:
    """KR-20 viết lại độc lập bằng numpy: sum(p*q) = tổng phương sai từng câu (ddof=0)."""
    x = scored.to_numpy(dtype=float)
    k = x.shape[1]
    return k / (k - 1) * (1 - np.var(x, axis=0).sum() / np.var(x.sum(axis=1), ddof=ddof))


def _staircase() -> pd.DataFrame:
    """4 thí sinh x 3 câu dạng bậc thang (tổng 3, 2, 1, 0)."""
    return pd.DataFrame([[1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 0]],
                        columns=["q1", "q2", "q3"], dtype=np.int8)


class TestKR20(unittest.TestCase):
    """kr20."""

    def test_hand_computed_staircase(self) -> None:
        # L = 3; p = 3/4, 2/4, 1/4 -> pq = 0,1875 + 0,25 + 0,1875 = 0,625
        # Tổng 3, 2, 1, 0: trung bình 1,5; độ lệch bình phương 2,25 + 0,25 + 0,25 + 2,25 = 5
        # ddof=0: SD^2 = 5/4 = 1,25 -> KR-20 = 3/2 * (1 - 0,625/1,25) = 0,75
        # ddof=1: SD^2 = 5/3        -> KR-20 = 3/2 * (1 - 0,375)      = 0,9375
        self.assertAlmostEqual(kr20(_staircase()), 0.75)
        self.assertAlmostEqual(kr20(_staircase(), ddof=1), 0.9375)

    def test_hand_computed_negative(self) -> None:
        # _hand_scored: L = 3; p = 0,5 / 0,5 / 0,4 -> pq = 0,25 + 0,25 + 0,24 = 0,74
        # Tổng 3,2,2,2,1,1,1,1,0,1: trung bình 1,4; trung bình bình phương 26/10 = 2,6
        # SD^2 (ddof=0) = 2,6 - 1,96 = 0,64 -> KR-20 = 3/2 * (1 - 0,74/0,64) = -0,234375
        self.assertAlmostEqual(kr20(_hand_scored()), -0.234375)

    def test_matches_numpy_reference(self) -> None:
        rng = np.random.default_rng(1)
        for _ in range(20):
            ability = rng.normal(size=(60, 1))
            data = (rng.normal(size=(60, 8)) < ability).astype(np.int8)
            df = pd.DataFrame(data, columns=[f"q{i}" for i in range(8)])
            for ddof in (0, 1):
                self.assertAlmostEqual(kr20(df, ddof=ddof), _kr20_reference(df, ddof))

    def test_ddof_relationship(self) -> None:
        # SD^2(ddof=1) = SD^2(ddof=0) * N/(N-1), nên 1 - KR/(L/(L-1)) co lại theo (N-1)/N
        # và KR-20 với ddof=1 luôn lớn hơn ddof=0 (khi sum(pq) > 0).
        for df in (_staircase(), _hand_scored()):
            size, k = df.shape
            factor = k / (k - 1)
            k0, k1 = kr20(df, ddof=0), kr20(df, ddof=1)
            self.assertGreater(k1, k0)
            self.assertAlmostEqual(1 - k1 / factor, (1 - k0 / factor) * (size - 1) / size)

    def test_input_unchanged(self) -> None:
        df = _hand_scored()
        before = df.copy()
        kr20(df)
        kr20(df, ddof=1)
        pd.testing.assert_frame_equal(df, before)

    def test_invalid_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            kr20(_staircase(), ddof=2)
        with self.assertRaises(ValueError):
            kr20(_staircase(), ddof=-1)
        with self.assertRaises(ValueError):
            kr20(pd.DataFrame())
        with self.assertRaises(ValueError):
            kr20(pd.DataFrame({"a": [0, 2], "b": [1, 0]}))
        with self.assertRaises(ValueError):
            kr20(pd.DataFrame({"a": [0, 1, 1]}))  # L = 1
        with self.assertRaises(ValueError):
            kr20(pd.DataFrame({"a": [1, 0], "b": [0, 1]}))  # tổng đều bằng 1 -> SD^2 = 0
        with self.assertRaises(ValueError):
            kr20(pd.DataFrame({"a": [1], "b": [0]}), ddof=1)  # N = 1 -> SD^2 không xác định


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

    def test_group_sizes_75_and_74(self) -> None:
        fraction = CTTThresholds().group_fraction
        self.assertEqual(len(self.scored_all), 276)
        self.assertEqual(len(self.scored_kept), 273)
        for scored, n in ((self.scored_all, 75), (self.scored_kept, 74)):
            high, low = split_groups(scored.sum(axis=1), fraction)
            self.assertEqual((len(high), len(low)), (n, n))

    def test_d_in_range_for_50_items(self) -> None:
        for scored in (self.scored_all, self.scored_kept):
            d = discrimination_index(scored)
            self.assertEqual(len(d), 50)
            self.assertEqual(list(d.index), list(scored.columns))
            self.assertTrue(d.between(-1, 1).all())

    def test_kr20_finite_for_276_and_273(self) -> None:
        for scored, size in ((self.scored_all, 276), (self.scored_kept, 273)):
            self.assertEqual(len(scored), size)
            for ddof in (0, 1):
                value = kr20(scored, ddof=ddof)
                self.assertTrue(np.isfinite(value))
                self.assertGreater(value, -1)
                self.assertLessEqual(value, 1)


if __name__ == "__main__":
    unittest.main()