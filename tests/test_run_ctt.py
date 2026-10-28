"""Test cho scripts.run_ctt (dữ liệu nhỏ trong thư mục tạm, không dùng mạng)."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from core.config import load_config
from core.io import dataset_hash
from scripts.run_ctt import (COMPARE_COLUMNS, COMPARE_FILE, ITEMS_FILE, SUMMARY_FILE,
                             build_parser, main, summary_lines)

# q1, q2 giữ nguyên cờ; q3, q4 bị cờ chỉ khi loại hai thí sinh bỏ trống toàn bộ (S9, S10)
RESPONSES_CSV = """student,q1,q2,q3,q4
S1,a,a,a,a
S2,a,a,a,b
S3,a,a,b,a
S4,a,b,a,b
S5,b,a,b,b
S6,a,b,b,b
S7,b,b,a,b
S8,b,b,b,a
S9,,,,
S10,,,,
"""
KEY_CSV = """question_id,correct_answer,num_choices
q1,a,2
q2,a,2
q3,a,2
q4,a,2
"""
ITEM_COLUMNS = ["question_id", "p", "difficulty_class", "D", "r_pbis", "r_pbis_class",
                "review_flag", "review_priority", "review_reason"]


def _run(argv: list[str]) -> tuple[int, str, str]:
    """Chạy main và bắt stdout/stderr; trả về (mã thoát, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class TestRunCTT(unittest.TestCase):
    """main() trên dữ liệu nhỏ."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.data = root / "data"
        self.data.mkdir()
        (self.data / "responses.csv").write_text(RESPONSES_CSV, encoding="utf-8")
        (self.data / "key.csv").write_text(KEY_CSV, encoding="utf-8")
        self.out = root / "out" / "nested"
        self.base = ["--data-dir", str(self.data), "--out-dir", str(self.out)]

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _summary(self) -> dict:
        return json.loads((self.out / SUMMARY_FILE).read_text(encoding="utf-8"))

    def test_writes_items_and_summary(self) -> None:
        code, stdout, _ = _run(self.base)
        self.assertEqual(code, 0)
        items = pd.read_csv(self.out / ITEMS_FILE, dtype={"question_id": str})
        self.assertEqual(list(items.columns), ITEM_COLUMNS)
        self.assertEqual(list(items["question_id"]), ["q1", "q2", "q3", "q4"])
        summary = self._summary()
        self.assertEqual(summary["n_students"], 10)
        self.assertEqual(summary["n_blank_students"], 2)
        self.assertFalse(summary["thresholds"]["exclude_blank_students"])
        expected = dataset_hash([self.data / "responses.csv", self.data / "key.csv"])
        self.assertEqual(summary["dataset_hash"], expected)
        self.assertEqual(len(stdout.strip().splitlines()), 5)
        self.assertFalse((self.out / COMPARE_FILE).exists())

    def test_exclude_blank_overrides(self) -> None:
        self.assertEqual(_run(self.base + ["--exclude-blank"])[0], 0)
        summary = self._summary()
        self.assertEqual(summary["n_students"], 8)
        self.assertTrue(summary["thresholds"]["exclude_blank_students"])

    def test_keep_blank_overrides_config(self) -> None:
        config = Path(self._tmp.name) / "config.toml"
        config.write_text("[ctt]\nexclude_blank_students = true\n", encoding="utf-8")
        self.assertEqual(_run(self.base + ["--config", str(config)])[0], 0)
        self.assertEqual(self._summary()["n_students"], 8)
        self.assertEqual(_run(self.base + ["--config", str(config), "--keep-blank"])[0], 0)
        self.assertEqual(self._summary()["n_students"], 10)

    def test_blank_flags_mutually_exclusive(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as ctx:
            build_parser().parse_args(self.base + ["--exclude-blank", "--keep-blank"])
        self.assertEqual(ctx.exception.code, 2)

    def test_compare_lists_only_changed_flags(self) -> None:
        code, stdout, _ = _run(self.base + ["--compare"])
        self.assertEqual(code, 0)
        diff = pd.read_csv(self.out / COMPARE_FILE, dtype={"question_id": str})
        self.assertEqual(list(diff.columns), list(COMPARE_COLUMNS))
        self.assertEqual(list(diff["question_id"]), ["q3", "q4"])
        self.assertFalse(diff["flag_keep"].any())
        self.assertTrue(diff["flag_exclude"].all())
        self.assertIn("2", stdout.strip().splitlines()[-1])
        # file chính vẫn theo chế độ chính (giữ thí sinh bỏ trống)
        self.assertEqual(self._summary()["n_students"], 10)

    def test_compare_with_exclude_mode_keeps_column_meaning(self) -> None:
        self.assertEqual(_run(self.base + ["--compare", "--exclude-blank"])[0], 0)
        diff = pd.read_csv(self.out / COMPARE_FILE, dtype={"question_id": str})
        self.assertEqual(list(diff["question_id"]), ["q3", "q4"])
        self.assertTrue(diff["flag_exclude"].all())
        self.assertEqual(self._summary()["n_students"], 8)

    def test_missing_file_exits_1(self) -> None:
        (self.data / "key.csv").unlink()
        code, _, stderr = _run(self.base)
        self.assertEqual(code, 1)
        self.assertIn("key.csv", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_invalid_data_exits_1(self) -> None:
        (self.data / "key.csv").write_text("question_id,correct_answer\nq1,a\n", encoding="utf-8")
        code, _, stderr = _run(self.base)
        self.assertEqual(code, 1)
        self.assertTrue(stderr.startswith("Lỗi"))
        self.assertNotIn("Traceback", stderr)

    def test_summary_lines(self) -> None:
        summary = {"n_students": 3, "n_blank_students": 0, "blank_students_excluded": False,
                   "kr20": 0.5, "n_flagged": 1, "n_items": 2,
                   "difficulty_counts": {"too_hard": 0, "suitable": 2, "too_easy": 0}}
        lines = summary_lines(summary)
        self.assertEqual(len(lines), 5)
        self.assertEqual(lines[0], "N: 3")


def _infoseccat_dir() -> Path:
    env = os.environ.get("INFOSECCAT_DIR")
    return Path(env) if env else load_config().data_dir / "raw" / "infoseccat"


@unittest.skipUnless((_infoseccat_dir() / "responses.csv").exists(), "chưa có dữ liệu InfoSecCAT")
class TestRunCTTInfoSecCAT(unittest.TestCase):
    """main() trên dữ liệu thật InfoSecCAT."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name)
        self.base = ["--data-dir", str(_infoseccat_dir()), "--out-dir", str(self.out)]

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _n_students(self) -> int:
        return json.loads((self.out / SUMMARY_FILE).read_text(encoding="utf-8"))["n_students"]

    def test_default_keeps_276(self) -> None:
        self.assertEqual(_run(self.base)[0], 0)
        self.assertEqual(self._n_students(), 276)

    def test_exclude_blank_gives_273(self) -> None:
        self.assertEqual(_run(self.base + ["--exclude-blank"])[0], 0)
        self.assertEqual(self._n_students(), 273)

    def test_compare_lists_17_items(self) -> None:
        self.assertEqual(_run(self.base + ["--compare"])[0], 0)
        diff = pd.read_csv(self.out / COMPARE_FILE)
        self.assertEqual(len(diff), 17)


if __name__ == "__main__":
    unittest.main()
