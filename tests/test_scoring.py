"""Test cho core.scoring."""
import unittest

import numpy as np
import pandas as pd

from core.scoring import (blank_students, filter_blank_students, omit_counts,
                          omit_rate, score_matrix, total_scores,
                          validate_responses)


def _key() -> pd.DataFrame:
    return pd.DataFrame(
        {"correct_answer": ["d", "a", "c"], "num_choices": [4, 4, 3]},
        index=pd.Index(["1", "2", "3"], name="question_id"),
    )


def _responses() -> pd.DataFrame:
    df = pd.DataFrame(
        [["d", "a", "c"], ["d", "", "a"], ["", "", ""], ["a", "a", "c"]],
        index=pd.Index(["S1", "S2", "S3", "S4"], name="student"),
        columns=["1", "2", "3"],
    )
    return df


class TestValidate(unittest.TestCase):
    """validate_responses."""

    def test_ok(self) -> None:
        validate_responses(_responses(), _key())

    def test_item_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_responses(_responses().drop(columns="3"), _key())
        extra = _responses().assign(**{"4": "a"})
        with self.assertRaises(ValueError):
            validate_responses(extra, _key())

    def test_option_outside_range_raises(self) -> None:
        bad = _responses()
        bad.loc["S2", "3"] = "d"  # câu 3 chỉ có 3 phương án a-c
        with self.assertRaises(ValueError):
            validate_responses(bad, _key())


class TestScoring(unittest.TestCase):
    """score_matrix, total_scores, omit_*."""

    def test_score_matrix_values_blank_is_wrong(self) -> None:
        scored = score_matrix(_responses(), _key())
        expected = np.array([[1, 1, 1], [1, 0, 0], [0, 0, 0], [0, 1, 1]])
        np.testing.assert_array_equal(scored.to_numpy(), expected)
        self.assertEqual(scored.to_numpy().dtype, np.int8)

    def test_score_matrix_follows_key_order(self) -> None:
        shuffled = _responses()[["3", "1", "2"]]
        scored = score_matrix(shuffled, _key())
        self.assertEqual(list(scored.columns), ["1", "2", "3"])
        self.assertEqual(int(scored.loc["S1"].sum()), 3)

    def test_total_scores(self) -> None:
        totals = total_scores(score_matrix(_responses(), _key()))
        self.assertEqual(list(totals), [3, 1, 0, 2])
        self.assertEqual(list(totals.index), ["S1", "S2", "S3", "S4"])

    def test_omit_counts_and_rate(self) -> None:
        self.assertEqual(list(omit_counts(_responses())), [1, 2, 1])
        self.assertEqual(list(omit_rate(_responses())), [0.25, 0.5, 0.25])


class TestBlankStudents(unittest.TestCase):
    """blank_students và filter_blank_students (T2.0)."""

    def test_blank_students_finds_only_fully_blank_rows(self) -> None:
        # S3 bỏ trống cả 3 câu; S2 chỉ bỏ trống 1 câu
        self.assertEqual(list(blank_students(_responses())), ["S3"])

    def test_blank_students_none(self) -> None:
        self.assertEqual(len(blank_students(_responses().drop(index="S3"))), 0)

    def test_filter_excludes_by_default(self) -> None:
        original = _responses()
        kept, removed = filter_blank_students(original)
        self.assertEqual(list(kept.index), ["S1", "S2", "S4"])
        self.assertEqual(list(removed), ["S3"])
        self.assertEqual(len(original), 4)  # không sửa dữ liệu gốc

    def test_filter_keep_when_exclude_false(self) -> None:
        kept, removed = filter_blank_students(_responses(), exclude=False)
        self.assertEqual(len(kept), 4)
        self.assertEqual(len(removed), 0)

    def test_filter_preserves_index_name_and_columns(self) -> None:
        kept, _ = filter_blank_students(_responses())
        self.assertEqual(kept.index.name, "student")
        self.assertEqual(list(kept.columns), ["1", "2", "3"])

    def test_filter_all_blank_raises(self) -> None:
        all_blank = _responses().loc[["S3"]]
        with self.assertRaises(ValueError):
            filter_blank_students(all_blank)

    def test_scores_unchanged_for_remaining_students(self) -> None:
        kept, _ = filter_blank_students(_responses())
        totals = total_scores(score_matrix(kept, _key()))
        self.assertEqual(list(totals), [3, 1, 2])


if __name__ == "__main__":
    unittest.main()