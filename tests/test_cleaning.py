"""Test cho core.cleaning."""
import dataclasses
import os
import unittest
from pathlib import Path

import pandas as pd

from core.cleaning import (REASON_ALL_BLANK, CleaningReport, PreparedDataset, clean_responses,
                           prepare_dataset)
from core.config import CTTThresholds, load_config
from core.ctt import SUITABLE, ctt_report, split_groups
from core.io import load_key, load_responses

KEEP = CTTThresholds(exclude_blank_students=False)


def _key() -> pd.DataFrame:
    """Đáp án 4 câu, đáp án "a", 4 phương án."""
    return pd.DataFrame(
        {"correct_answer": ["a"] * 4, "num_choices": [4] * 4},
        index=pd.Index(["q1", "q2", "q3", "q4"], name="question_id"),
    )


def _responses() -> pd.DataFrame:
    """S1 đủ; S2 bỏ trống 2/4 (đúng ngưỡng 0,5); S3 bỏ trống 1/4; S4 sai hết nhưng có trả lời; B1, B2 bỏ trống toàn bộ."""
    return pd.DataFrame(
        [
            ["a", "a", "b", "a"],
            ["a", "", "", "a"],
            ["", "a", "a", "b"],
            ["b", "c", "d", "b"],
            ["", "", "", ""],
            ["", "", "", ""],
        ],
        index=pd.Index(["S1", "S2", "S3", "S4", "B1", "B2"], name="student"),
        columns=pd.Index(["q1", "q2", "q3", "q4"], name="question_id"),
    )


class TestCleanResponses(unittest.TestCase):
    """clean_responses."""

    def test_removes_blank_students_with_reason(self) -> None:
        cleaned, report = clean_responses(_responses())
        self.assertIsInstance(report, CleaningReport)
        self.assertEqual(list(cleaned.index), ["S1", "S2", "S3", "S4"])
        self.assertEqual((report.n_before, report.n_after), (6, 4))
        self.assertEqual(list(report.removed.columns), ["student", "reason", "n_answered"])
        self.assertEqual(list(report.removed["student"]), ["B1", "B2"])
        self.assertEqual(list(report.removed["reason"]), [REASON_ALL_BLANK] * 2)
        self.assertEqual(list(report.removed["n_answered"]), [0, 0])

    def test_all_wrong_but_answered_is_kept(self) -> None:
        cleaned, _ = clean_responses(_responses())
        self.assertIn("S4", cleaned.index)

    def test_warns_at_threshold_without_removing(self) -> None:
        cleaned, report = clean_responses(_responses())
        # S2 bỏ trống đúng 2/4 = 0,5 -> cảnh báo; S3 1/4 thì không
        self.assertEqual(len(report.warnings), 1)
        self.assertTrue(report.warnings[0].startswith("S2:"))
        self.assertIn("S2", cleaned.index)
        _, strict = clean_responses(_responses(), CTTThresholds(warn_omit_rate=0.25))
        self.assertEqual([w.split(":")[0] for w in strict.warnings], ["S2", "S3"])

    def test_keep_mode_removes_nobody(self) -> None:
        cleaned, report = clean_responses(_responses(), KEEP)
        self.assertEqual(len(cleaned), 6)
        self.assertTrue(report.removed.empty)
        self.assertEqual(list(report.removed.columns), ["student", "reason", "n_answered"])

    def test_input_unchanged(self) -> None:
        responses = _responses()
        before = responses.copy()
        clean_responses(responses)
        pd.testing.assert_frame_equal(responses, before)

    def test_all_removed_raises(self) -> None:
        with self.assertRaises(ValueError):
            clean_responses(pd.DataFrame("", index=["B1", "B2"], columns=["q1", "q2"]))


class TestPrepareDataset(unittest.TestCase):
    """prepare_dataset."""

    def test_scored_only_kept_students_in_key_order(self) -> None:
        key = _key().loc[["q3", "q1", "q4", "q2"]]
        prepared = prepare_dataset(_responses(), key)
        self.assertIsInstance(prepared, PreparedDataset)
        self.assertEqual(list(prepared.scored.index), ["S1", "S2", "S3", "S4"])
        self.assertEqual(list(prepared.scored.columns), ["q3", "q1", "q4", "q2"])
        self.assertEqual(list(prepared.responses.columns), ["q3", "q1", "q4", "q2"])
        self.assertEqual(prepared.scored.loc["S1"].tolist(), [0, 1, 1, 1])
        self.assertEqual(str(prepared.scored.dtypes.iloc[0]), "int8")
        self.assertEqual(list(prepared.report.removed["student"]), ["B1", "B2"])

    def test_validation_runs_before_cleaning(self) -> None:
        # Thí sinh bỏ trống toàn bộ đứng cạnh thí sinh có phương án sai: vẫn báo lỗi phương án
        bad = _responses()
        bad.loc["S1", "q1"] = "z"
        with self.assertRaisesRegex(ValueError, "không hợp lệ"):
            prepare_dataset(bad, _key())
        # Tất cả bỏ trống nhưng key thiếu câu: lỗi khớp key, không phải lỗi loại hết
        all_blank = pd.DataFrame("", index=["B1"], columns=["q1", "q2", "q3", "q4"])
        with self.assertRaisesRegex(ValueError, "Mã câu không khớp"):
            prepare_dataset(all_blank, _key().iloc[:2])

    def test_is_frozen(self) -> None:
        prepared = prepare_dataset(_responses(), _key())
        with self.assertRaises(dataclasses.FrozenInstanceError):
            prepared.scored = pd.DataFrame()  # type: ignore[misc]


def _data_dir() -> Path:
    env = os.environ.get("INFOSECCAT_DIR")
    return Path(env) if env else load_config().data_dir / "raw" / "infoseccat"


@unittest.skipUnless((_data_dir() / "responses.csv").exists(), "chưa có dữ liệu InfoSecCAT")
class TestCleaningInfoSecCAT(unittest.TestCase):
    """Kết quả chính trên InfoSecCAT sau làm sạch (mặc định loại thí sinh bỏ trống toàn bộ)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.responses = load_responses(_data_dir() / "responses.csv")
        cls.key = load_key(_data_dir() / "key.csv")

    def test_prepare_removes_three(self) -> None:
        prepared = prepare_dataset(self.responses, self.key)
        self.assertEqual(len(prepared.scored), 273)
        self.assertEqual(list(prepared.report.removed["student"]), ["S050", "S179", "S225"])
        self.assertEqual(int((prepared.responses == "").sum().sum()), 7)
        high, low = split_groups(prepared.scored.sum(axis=1), CTTThresholds().group_fraction)
        self.assertEqual((len(high), len(low)), (74, 74))

    def test_default_ctt_report(self) -> None:
        report = ctt_report(self.responses, self.key)
        s = report.summary
        self.assertEqual((s["n_students"], s["n_excluded"]), (273, 3))
        self.assertEqual(s["cleaning_removed"], ["S050", "S179", "S225"])
        self.assertAlmostEqual(s["kr20"], 0.7339, delta=1e-3)
        self.assertEqual(s["n_flagged"], 23)
        self.assertEqual(s["difficulty_counts"], {"too_hard": 1, "suitable": 9, "too_easy": 40})
        suitable = report.items.index[report.items["difficulty_class"] == SUITABLE]
        self.assertEqual(list(suitable), ["7", "18", "22", "25", "31", "34", "36", "44", "50"])

    def test_keep_mode_ctt_report(self) -> None:
        s = ctt_report(self.responses, self.key, KEEP).summary
        self.assertEqual(s["n_students"], 276)
        self.assertAlmostEqual(s["kr20"], 0.8578, delta=1e-3)
        self.assertEqual(s["n_flagged"], 6)


if __name__ == "__main__":
    unittest.main()
