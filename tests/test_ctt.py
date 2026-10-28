"""Test cho core.ctt (T2.1: độ khó p và phân loại; T2.2: độ phân biệt D; T2.3: r_pbis; T2.4: KR-20;
T2.5: bảng CTT và cờ "cần xem lại")."""
import os
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr

from core.config import CTTThresholds, load_config
from core.ctt import (SUITABLE, TOO_EASY, TOO_HARD, CTTReport, classify_difficulty,
                      classify_point_biserial, ctt_report, discrimination_index,
                      item_difficulty, kr20, point_biserial, review_flags, split_groups)
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


def _pbis_scored() -> pd.DataFrame:
    """5 thí sinh x 3 câu cho r_pbis tính tay."""
    rows = [
        [1, 1, 1],  # S0 tổng 3
        [1, 1, 0],  # S1 tổng 2
        [1, 0, 0],  # S2 tổng 1
        [0, 1, 0],  # S3 tổng 1
        [0, 0, 0],  # S4 tổng 0
    ]
    return pd.DataFrame(
        np.array(rows, dtype=np.int8),
        index=pd.Index([f"S{i}" for i in range(5)], name="student"),
        columns=pd.Index(["q1", "q2", "q3"], name="question_id"),
    )


def _scipy_pbis(scored: pd.DataFrame, exclude_item: bool) -> list[float]:
    """r_pbis từng câu tính bằng scipy.stats.pointbiserialr để đối chiếu."""
    total = scored.sum(axis=1).to_numpy(dtype=float)
    out = []
    for col in scored.columns:
        x = scored[col].to_numpy(dtype=float)
        out.append(pointbiserialr(x, total - x if exclude_item else total).statistic)
    return out


class TestPointBiserial(unittest.TestCase):
    """point_biserial."""

    def test_hand_computed_full_total(self) -> None:
        # Công thức r = (M1 - M0) / s * sqrt(p * q), s là độ lệch chuẩn tổng thể của tổng.
        # Tổng T = [3, 2, 1, 1, 0]: trung bình 1,4; s^2 = (9+4+1+1+0)/5 - 1,96 = 1,04.
        # q1 = [1,1,1,0,0]: p = 0,6, M1 = (3+2+1)/3 = 2, M0 = (1+0)/2 = 0,5
        #   r = 1,5 * sqrt(0,24 / 1,04) = 1,5 * sqrt(3/13) ≈ 0,7206
        # q2 = [1,1,0,1,0]: p = 0,6, M1 = (3+2+1)/3 = 2, M0 = (1+0)/2 = 0,5 -> như q1
        # q3 = [1,0,0,0,0]: p = 0,2, M1 = 3, M0 = (2+1+1+0)/4 = 1
        #   r = 2 * sqrt(0,16 / 1,04) ≈ 0,7845
        r = point_biserial(_pbis_scored(), exclude_item=False)
        self.assertEqual(r.name, "r_pbis")
        self.assertEqual(list(r.index), ["q1", "q2", "q3"])
        np.testing.assert_allclose(
            r.to_numpy(),
            [1.5 * np.sqrt(3 / 13), 1.5 * np.sqrt(3 / 13), 2 * np.sqrt(0.16 / 1.04)],
            rtol=0, atol=1e-12,
        )

    def test_hand_computed_corrected_total(self) -> None:
        # Tổng trừ câu R = T - x, mỗi câu một R riêng; ở cả ba câu s^2(R) = 0,56.
        # q1: R = [2,1,0,1,0], trung bình 0,8, s^2 = 6/5 - 0,64 = 0,56
        #   M1 = (2+1+0)/3 = 1, M0 = (1+0)/2 = 0,5 -> r = 0,5 * sqrt(0,24/0,56) = 0,5 * sqrt(3/7)
        # q2: R = [2,1,1,0,0], trung bình 0,8, s^2 = 0,56
        #   M1 = (2+1+0)/3 = 1, M0 = (1+0)/2 = 0,5 -> như q1
        # q3: R = [2,2,1,1,0], trung bình 1,2, s^2 = 10/5 - 1,44 = 0,56
        #   M1 = 2, M0 = (2+1+1+0)/4 = 1 -> r = sqrt(0,16/0,56) = sqrt(2/7)
        expected = [0.5 * np.sqrt(3 / 7), 0.5 * np.sqrt(3 / 7), np.sqrt(2 / 7)]
        np.testing.assert_allclose(
            point_biserial(_pbis_scored(), exclude_item=True).to_numpy(), expected, rtol=0, atol=1e-12
        )
        # mặc định (None) lấy exclude_item_from_total = True từ CTTThresholds
        np.testing.assert_allclose(point_biserial(_pbis_scored()).to_numpy(), expected, rtol=0, atol=1e-12)

    def test_exclude_item_from_thresholds(self) -> None:
        df = _pbis_scored()
        t = CTTThresholds(exclude_item_from_total=False)
        pd.testing.assert_series_equal(point_biserial(df, t), point_biserial(df, exclude_item=False))
        # tham số tường minh ghi đè thresholds
        pd.testing.assert_series_equal(point_biserial(df, t, exclude_item=True), point_biserial(df))

    def test_matches_scipy_random(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(10):
            ability = rng.normal(size=(150, 1))
            difficulty = rng.normal(size=(1, 12))
            prob = 1 / (1 + np.exp(-(ability - difficulty)))
            data = (rng.random((150, 12)) < prob).astype(np.int8)
            df = pd.DataFrame(data, columns=[f"q{i}" for i in range(12)])
            for exclude in (True, False):
                r = point_biserial(df, exclude_item=exclude)
                np.testing.assert_allclose(r.to_numpy(), _scipy_pbis(df, exclude), rtol=0, atol=1e-9)

    def test_p_zero_or_one_is_nan_without_warning(self) -> None:
        rng = np.random.default_rng(1)
        data = rng.integers(0, 2, size=(30, 5)).astype(np.int8)
        data[:, 0] = 1
        data[:, 3] = 0
        df = pd.DataFrame(data, columns=[f"q{i}" for i in range(5)])
        for exclude in (True, False):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                r = point_biserial(df, exclude_item=exclude)
            self.assertTrue(np.isnan(r["q0"]) and np.isnan(r["q3"]))
            self.assertFalse(r[["q1", "q2", "q4"]].isna().any())

    def test_constant_corrected_total_is_nan(self) -> None:
        # q2 ai cũng đúng nên tổng trừ q1 = [1,1,1,1] là hằng số -> NaN, không phải 0
        df = pd.DataFrame({"q1": [1, 0, 1, 0], "q2": [1, 1, 1, 1]})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            r_ex = point_biserial(df, exclude_item=True)
            r_full = point_biserial(df, exclude_item=False)
        self.assertTrue(r_ex.isna().all())
        # tổng đầy đủ = q1 + 1 nên tương quan hoàn hảo
        self.assertAlmostEqual(r_full["q1"], 1.0, places=12)
        self.assertTrue(np.isnan(r_full["q2"]))

    def test_constant_full_total_is_nan(self) -> None:
        # mỗi thí sinh đúng đúng một câu: tổng đầy đủ là hằng số
        df = pd.DataFrame({"q1": [1, 0, 1, 0], "q2": [0, 1, 0, 1]})
        self.assertTrue(point_biserial(df, exclude_item=False).isna().all())
        np.testing.assert_allclose(point_biserial(df, exclude_item=True).to_numpy(), [-1.0, -1.0])

    def test_bounds_random(self) -> None:
        rng = np.random.default_rng(7)
        for _ in range(30):
            n_students = int(rng.integers(2, 20))
            data = rng.integers(0, 2, size=(n_students, 4)).astype(np.int8)
            df = pd.DataFrame(data, columns=list("abcd"))
            for exclude in (True, False):
                r = point_biserial(df, exclude_item=exclude).dropna()
                self.assertTrue(((r >= -1.0) & (r <= 1.0)).all())

    def test_input_unchanged(self) -> None:
        df = _pbis_scored()
        before = df.copy()
        for exclude in (True, False):
            point_biserial(df, exclude_item=exclude)
        pd.testing.assert_frame_equal(df, before)

    def test_invalid_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            point_biserial(pd.DataFrame())
        with self.assertRaises(ValueError):
            point_biserial(pd.DataFrame({"a": [0, 1, 2]}))
        with self.assertRaises(ValueError):
            point_biserial(pd.DataFrame({"a": [0.5, 1.0]}), exclude_item=False)


class TestClassifyPointBiserial(unittest.TestCase):
    """classify_point_biserial."""

    def test_default_thresholds_including_boundaries(self) -> None:
        r = pd.Series([0.5, 0.4, 0.3999, 0.2, 0.1999, -0.3, np.nan], index=list("abcdefg"))
        cls = classify_point_biserial(r)
        self.assertEqual(cls.name, "r_pbis_class")
        self.assertEqual(list(cls.index), list("abcdefg"))
        self.assertEqual(
            list(cls), ["good", "good", "acceptable", "acceptable", "poor", "poor", "undefined"]
        )

    def test_float_noise_and_custom_thresholds(self) -> None:
        r = pd.Series([0.2 - 1e-12, 0.1, 0.25])
        self.assertEqual(list(classify_point_biserial(r)), ["acceptable", "poor", "acceptable"])
        t = CTTThresholds(r_pbis_acceptable=0.1, r_pbis_good=0.25)
        self.assertEqual(list(classify_point_biserial(r, t)), ["acceptable", "acceptable", "good"])


class TestReviewFlags(unittest.TestCase):
    """review_flags."""

    def test_flag_priority_reason(self) -> None:
        # a: r tốt; b: r kém, p cao; c: r kém, p = p_min (biên, tính "high");
        # d: r = NaN, p = 1; e: r = NaN, p = 0; f: r đúng bằng ngưỡng 0,2 (không cờ)
        idx = list("abcdef")
        p = pd.Series([0.5, 0.8, 0.3, 1.0, 0.0, 0.1], index=idx)
        r = pd.Series([0.5, 0.1, -0.2, np.nan, np.nan, 0.2], index=idx)
        flags = review_flags(p, r)
        self.assertEqual(list(flags.columns), ["review_flag", "review_priority", "review_reason"])
        self.assertEqual(list(flags.index), idx)
        self.assertEqual(list(flags["review_flag"]), [False, True, True, True, True, False])
        self.assertEqual(list(flags["review_priority"]), ["", "normal", "high", "normal", "high", ""])
        self.assertEqual(
            list(flags["review_reason"]),
            ["", "r_pbis<0.2", "r_pbis<0.2 & p<=0.3", "r_pbis undefined",
             "r_pbis undefined & p<=0.3", ""],
        )

    def test_reason_uses_thresholds(self) -> None:
        t = CTTThresholds(p_min=0.25, r_pbis_acceptable=0.15, r_pbis_good=0.3)
        flags = review_flags(pd.Series([0.2, 0.5]), pd.Series([0.16, 0.1]), t)
        self.assertEqual(list(flags["review_flag"]), [False, True])
        self.assertEqual(list(flags["review_reason"]), ["", "r_pbis<0.15"])

    def test_mismatched_index_raises(self) -> None:
        with self.assertRaises(ValueError):
            review_flags(pd.Series([0.5], index=["a"]), pd.Series([0.5], index=["b"]))


# 6 thí sinh x 5 câu, đáp án đúng đều là "a"; "b" hoặc "" là sai.
# Ma trận 0/1:      q1 q2 q3 q4 q5 | tổng
#   S1              1  0  0  1  0  |  2
#   S2              1  1  0  1  1  |  4
#   S3              1  0  1  1  1  |  4
#   S4              1  1  0  0  0  |  2
#   S5              1  1  0  1  1  |  4
#   S6              1  1  0  1  0  |  3
_HAND_SCORED = [
    [1, 0, 0, 1, 0],
    [1, 1, 0, 1, 1],
    [1, 0, 1, 1, 1],
    [1, 1, 0, 0, 0],
    [1, 1, 0, 1, 1],
    [1, 1, 0, 1, 0],
]
_HAND_ITEMS = ["q1", "q2", "q3", "q4", "q5"]


def _hand_key(items: list[str] = _HAND_ITEMS) -> pd.DataFrame:
    """Đáp án: mọi câu đáp án "a", 4 phương án."""
    return pd.DataFrame(
        {"correct_answer": ["a"] * len(items), "num_choices": [4] * len(items)},
        index=pd.Index(items, name="question_id"),
    )


def _hand_responses() -> pd.DataFrame:
    """Ma trận lựa chọn ứng với ``_HAND_SCORED``; ô sai xen kẽ "b" và "" (đều tính sai)."""
    rows = []
    for i, row in enumerate(_HAND_SCORED):
        rows.append(["a" if v else ("" if (i + j) % 2 else "b") for j, v in enumerate(row)])
    return pd.DataFrame(
        rows,
        index=pd.Index([f"S{i + 1}" for i in range(6)], name="student"),
        columns=pd.Index(_HAND_ITEMS, name="question_id"),
    )


def _with_blank_students(responses: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Thêm các thí sinh bỏ trống toàn bộ (mọi ô là "")."""
    blank = pd.DataFrame("", index=pd.Index(names, name="student"), columns=responses.columns)
    return pd.concat([responses, blank])


class TestCTTReport(unittest.TestCase):
    """ctt_report trên bộ nhỏ tính tay."""

    def test_hand_computed_items(self) -> None:
        # p = số đúng / 6: q1 6/6 = 1; q2 4/6; q3 1/6; q4 5/6; q5 3/6.
        # Phân loại (0,3 / 0,7): q1 too_easy, q2 suitable, q3 too_hard, q4 too_easy, q5 suitable.
        #
        # D: N = 6, n = floor(0,27*6 + 0,5) = 2. Tổng [2,4,4,2,4,3]; xếp ổn định giảm dần:
        # S2,S3,S5,S6,S1,S4 -> nhóm cao {S2,S3}, nhóm thấp {S1,S4}.
        # q1: 1-1 = 0; q2: (1+0)/2-(0+1)/2 = 0; q3: (0+1)/2-0 = 0,5;
        # q4: (1+1)/2-(1+0)/2 = 0,5; q5: 1-0 = 1.
        #
        # r_pbis với tổng hiệu chỉnh y = tổng - câu đang xét (q1 hằng số nên không ảnh hưởng r).
        # Dùng Sxy = sum(xy) - N*mx*my, Sxx = N*p*q, Syy = sum(y^2) - N*my^2, r = Sxy/sqrt(Sxx*Syy).
        # q1: x hằng số -> NaN.
        # q2: x=[0,1,0,1,1,1], y=[2,3,4,1,3,2]; bỏ hằng số 1 của q1: y'=[1,2,3,0,2,1], my=1,5;
        #     Sxy = 5 - 6*(2/3)*1,5 = -1; Sxx = 4/3; Syy = 19 - 13,5 = 5,5;
        #     r = -1/sqrt(22/3) ≈ -0,369274.
        # q3: x=[0,0,1,0,0,0], y'=[1,3,2,1,3,2], my=2; Sxy = 2 - 6*(1/6)*2 = 0 -> r = 0.
        # q4: x=[1,1,1,0,1,1], y'=[0,2,2,1,2,1], my=4/3; Sxy = 7 - 6*(5/6)*(4/3) = 1/3;
        #     Sxx = 5/6; Syy = 14 - 32/3 = 10/3; r = (1/3)/sqrt(25/9) = 0,2 (đúng bằng ngưỡng).
        # q5: x=[0,1,1,0,1,0], y'=[1,2,2,1,2,2], my=5/3; Sxy = 6 - 5 = 1; Sxx = 1,5;
        #     Syy = 18 - 50/3 = 4/3; r = 1/sqrt(2) ≈ 0,707107.
        #
        # Phân loại r (0,2 / 0,4): q1 undefined, q2 poor, q3 poor, q4 acceptable (biên), q5 good.
        # Cờ: q1 (NaN, p = 1 > 0,3 -> normal), q2 (r<0,2, p = 2/3 -> normal),
        #     q3 (r<0,2, p = 1/6 <= 0,3 -> high); q4 đúng bằng ngưỡng nên không cờ.
        report = ctt_report(_hand_responses(), _hand_key())
        self.assertIsInstance(report, CTTReport)
        items = report.items
        self.assertEqual(
            list(items.columns),
            ["p", "difficulty_class", "D", "r_pbis", "r_pbis_class",
             "review_flag", "review_priority", "review_reason"],
        )
        self.assertEqual(list(items.index), _HAND_ITEMS)
        np.testing.assert_allclose(items["p"], [1, 4 / 6, 1 / 6, 5 / 6, 3 / 6])
        self.assertEqual(
            list(items["difficulty_class"]), [TOO_EASY, SUITABLE, TOO_HARD, TOO_EASY, SUITABLE]
        )
        np.testing.assert_allclose(items["D"], [0, 0, 0.5, 0.5, 1])
        r = items["r_pbis"].to_numpy()
        self.assertTrue(np.isnan(r[0]))
        np.testing.assert_allclose(r[1:], [-1 / np.sqrt(22 / 3), 0, 0.2, 1 / np.sqrt(2)], atol=1e-12)
        self.assertEqual(list(items["r_pbis_class"]), ["undefined", "poor", "poor", "acceptable", "good"])
        self.assertEqual(list(items["review_flag"]), [True, True, True, False, False])
        self.assertEqual(list(items["review_priority"]), ["normal", "normal", "high", "", ""])
        self.assertEqual(
            list(items["review_reason"]),
            ["r_pbis undefined", "r_pbis<0.2", "r_pbis<0.2 & p<=0.3", "", ""],
        )

    def test_hand_computed_summary(self) -> None:
        # KR-20: L = 5; tổng [2,4,4,2,4,3], trung bình 19/6, SD^2 (ddof 0) = 65/6 - (19/6)^2 = 29/36;
        # sum(pq) = 0 + 2/9 + 5/36 + 5/36 + 1/4 = 27/36; KR-20 = 5/4 * (1 - 27/29) = 5/58.
        s = ctt_report(_hand_responses(), _hand_key()).summary
        self.assertAlmostEqual(s.pop("kr20"), 5 / 58, places=12)
        self.assertEqual(
            s,
            {
                "n_students": 6, "n_blank_students": 0, "n_excluded": 0,
                "blank_students_excluded": False, "n_items": 5, "kr20_excluding_blank": None,
                "ddof": 0, "exclude_item_from_total": True, "group_fraction": 0.27, "n_flagged": 3,
                "difficulty_counts": {TOO_HARD: 1, SUITABLE: 2, TOO_EASY: 2},
            },
        )

    def test_boundary_with_custom_threshold(self) -> None:
        # q5 có r = 1/sqrt(2); đặt ngưỡng đúng bằng giá trị đó -> "acceptable", không cờ.
        t = CTTThresholds(r_pbis_acceptable=1 / np.sqrt(2), r_pbis_good=0.9)
        items = ctt_report(_hand_responses(), _hand_key(), t).items
        self.assertEqual(items.loc["q5", "r_pbis_class"], "acceptable")
        self.assertFalse(items.loc["q5", "review_flag"])
        self.assertTrue(items.loc["q4", "review_flag"])
        self.assertEqual(items.loc["q4", "review_reason"], "r_pbis<0.707107")

    def test_full_total_follows_thresholds(self) -> None:
        t = CTTThresholds(exclude_item_from_total=False)
        report = ctt_report(_hand_responses(), _hand_key(), t)
        expected = point_biserial(score_matrix(_hand_responses(), _hand_key()), exclude_item=False)
        pd.testing.assert_series_equal(report.items["r_pbis"], expected)
        self.assertFalse(report.summary["exclude_item_from_total"])

    def test_items_follow_key_order(self) -> None:
        order = ["q5", "q3", "q1", "q4", "q2"]
        items = ctt_report(_hand_responses(), _hand_key(order)).items
        self.assertEqual(list(items.index), order)
        self.assertAlmostEqual(items.loc["q3", "p"], 1 / 6)

    def test_exclude_blank_students_true_and_false(self) -> None:
        responses = _with_blank_students(_hand_responses(), ["B1", "B2"])
        before = responses.copy()
        kept = ctt_report(responses, _hand_key())
        excluded = ctt_report(responses, _hand_key(), CTTThresholds(exclude_blank_students=True))
        pd.testing.assert_frame_equal(responses, before)

        # Giữ: N = 8, p(q1) = 6/8, KR-20 khác bộ 6 người; kr20_excluding_blank = 5/58.
        self.assertEqual(kept.summary["n_students"], 8)
        self.assertEqual(kept.summary["n_blank_students"], 2)
        self.assertEqual(kept.summary["n_excluded"], 0)
        self.assertFalse(kept.summary["blank_students_excluded"])
        self.assertAlmostEqual(kept.items.loc["q1", "p"], 0.75)
        self.assertAlmostEqual(kept.summary["kr20_excluding_blank"], 5 / 58, places=12)
        self.assertNotAlmostEqual(kept.summary["kr20"], 5 / 58, places=6)

        # Loại: giống hệt kết quả trên 6 thí sinh ban đầu.
        self.assertEqual(excluded.summary["n_students"], 6)
        self.assertEqual(excluded.summary["n_blank_students"], 2)
        self.assertEqual(excluded.summary["n_excluded"], 2)
        self.assertTrue(excluded.summary["blank_students_excluded"])
        self.assertAlmostEqual(excluded.summary["kr20"], 5 / 58, places=12)
        self.assertEqual(excluded.summary["kr20_excluding_blank"], excluded.summary["kr20"])
        pd.testing.assert_frame_equal(
            excluded.items, ctt_report(_hand_responses(), _hand_key()).items
        )

    def test_all_wrong_but_answered_is_not_blank(self) -> None:
        # S7 trả lời "b" mọi câu: toàn 0 trong ma trận chấm điểm nhưng không phải bỏ trống
        responses = _hand_responses()
        responses.loc["S7"] = ["b"] * len(_HAND_ITEMS)
        t = CTTThresholds(exclude_blank_students=True)
        s = ctt_report(responses, _hand_key(), t).summary
        self.assertEqual((s["n_students"], s["n_blank_students"], s["n_excluded"]), (7, 0, 0))
        self.assertIsNone(s["kr20_excluding_blank"])

    def test_report_is_frozen(self) -> None:
        report = ctt_report(_hand_responses(), _hand_key())
        with self.assertRaises(AttributeError):
            report.items = pd.DataFrame()  # type: ignore[misc]

    def test_invalid_inputs_raise(self) -> None:
        all_blank = pd.DataFrame("", index=["S1", "S2"], columns=_HAND_ITEMS)
        with self.assertRaises(ValueError):
            ctt_report(all_blank, _hand_key(), CTTThresholds(exclude_blank_students=True))
        with self.assertRaises(ValueError):
            ctt_report(_hand_responses(), _hand_key(["q1", "q2"]))
        bad = _hand_responses()
        bad.iloc[0, 0] = "z"
        with self.assertRaises(ValueError):
            ctt_report(bad, _hand_key())


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

    def test_r_pbis_matches_scipy_for_50_items(self) -> None:
        for scored in (self.scored_all, self.scored_kept):
            p = item_difficulty(scored)
            self.assertTrue(((p >= 0.196 - 1e-3) & (p <= 0.989 + 1e-3)).all())
            for exclude in (True, False):
                r = point_biserial(scored, exclude_item=exclude)
                self.assertEqual(len(r), 50)
                self.assertEqual(list(r.index), list(scored.columns))
                self.assertFalse(r.isna().any())
                np.testing.assert_allclose(r.to_numpy(), _scipy_pbis(scored, exclude), rtol=0, atol=1e-9)

    def test_ctt_report_default_keeps_all_276(self) -> None:
        key = load_key(_data_dir() / "key.csv")
        report = ctt_report(self.responses, key)
        s = report.summary
        self.assertEqual((s["n_students"], s["n_blank_students"], s["n_excluded"]), (276, 3, 0))
        self.assertAlmostEqual(s["kr20"], 0.858, delta=1e-3)
        self.assertAlmostEqual(s["kr20_excluding_blank"], 0.734, delta=1e-3)
        self.assertEqual(len(report.items), 50)
        self.assertEqual(list(report.items.index), list(key.index))

    def test_ctt_report_excluding_blank_273(self) -> None:
        key = load_key(_data_dir() / "key.csv")
        report = ctt_report(self.responses, key, CTTThresholds(exclude_blank_students=True))
        s = report.summary
        self.assertEqual((s["n_students"], s["n_excluded"]), (273, 3))
        self.assertAlmostEqual(s["kr20"], s["kr20_excluding_blank"], places=12)
        self.assertAlmostEqual(s["kr20"], 0.734, delta=1e-3)
        self.assertEqual(len(report.items), 50)
        self.assertEqual(list(report.items.index), list(key.index))


if __name__ == "__main__":
    unittest.main()