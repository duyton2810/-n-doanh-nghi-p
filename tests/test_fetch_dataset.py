"""Test cho scripts.fetch_dataset (dùng file:// nên không cần mạng)."""
import tempfile
import unittest
from pathlib import Path

from scripts.fetch_dataset import main


class TestFetchDataset(unittest.TestCase):
    """main() của script tải dữ liệu."""

    def test_downloads_requested_files(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            src, dest = Path(d) / "src", Path(d) / "dest"
            src.mkdir()
            (src / "a.csv").write_text("x", encoding="utf-8")
            (src / "b.csv").write_text("y", encoding="utf-8")
            code = main(["--base-url", src.as_uri() + "/", "--dest", str(dest),
                         "--files", "a.csv", "b.csv"])
            self.assertEqual(code, 0)
            self.assertEqual((dest / "a.csv").read_text(encoding="utf-8"), "x")
            self.assertEqual((dest / "b.csv").read_text(encoding="utf-8"), "y")


if __name__ == "__main__":
    unittest.main()
