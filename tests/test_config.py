"""Test cho core.config."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core.config import ENV_DATA_DIR, AppConfig, CTTThresholds, load_config


class TestCTTThresholds(unittest.TestCase):
    """Kiểm tra giá trị mặc định và ràng buộc của ngưỡng."""

    def test_defaults_match_paper(self) -> None:
        t = CTTThresholds()
        self.assertEqual((t.p_min, t.p_max, t.group_fraction), (0.3, 0.7, 0.27))
        self.assertEqual((t.r_pbis_acceptable, t.r_pbis_good), (0.2, 0.4))
        self.assertTrue(t.exclude_item_from_total)
        self.assertFalse(t.exclude_blank_students)

    def test_invalid_values_raise(self) -> None:
        for kwargs in ({"p_min": 0.8}, {"group_fraction": 0.5}, {"r_pbis_good": 0.1}):
            with self.assertRaises(ValueError):
                CTTThresholds(**kwargs)


class TestLoadConfig(unittest.TestCase):
    """Kiểm tra load_config."""

    def test_defaults_without_file(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ENV_DATA_DIR, None)
            cfg = load_config()
        self.assertIsInstance(cfg, AppConfig)
        self.assertEqual(cfg.data_dir, Path.cwd() / "data")

    def test_toml_relative_to_file_and_custom_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.toml"
            p.write_text('[paths]\ndata_dir = "dd"\n[ctt]\np_min = 0.2\n', encoding="utf-8")
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(ENV_DATA_DIR, None)
                cfg = load_config(p)
            self.assertEqual(cfg.data_dir, Path(d).resolve() / "dd")
            self.assertEqual(cfg.ctt.p_min, 0.2)

    def test_exclude_blank_students_from_toml(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.toml"
            p.write_text("[ctt]\nexclude_blank_students = true\n", encoding="utf-8")
            self.assertTrue(load_config(p).ctt.exclude_blank_students)
            p.write_text("[ctt]\np_min = 0.2\n", encoding="utf-8")
            self.assertFalse(load_config(p).ctt.exclude_blank_students)

    def test_repo_config_toml_matches_defaults(self) -> None:
        cfg = load_config(Path(__file__).resolve().parent.parent / "config.toml")
        self.assertEqual(cfg.ctt, CTTThresholds())

    def test_env_overrides_data_dir(self) -> None:
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {ENV_DATA_DIR: d}):
            self.assertEqual(load_config().data_dir, Path(d))

    def test_unknown_ctt_key_raises(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.toml"
            p.write_text("[ctt]\nfoo = 1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(p)


if __name__ == "__main__":
    unittest.main()
