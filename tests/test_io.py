"""Test cho core.io."""
import tempfile
import unittest
from pathlib import Path

from core.io import (dataset_hash, download_file, load_key, load_responses,
                     sha256_file)

KEY_OK = "question_id,correct_answer,num_choices\r\n1,d,4\r\n2,A,4\r\n"
RESP_OK = "student,1,2\nS001,d,a\nS002, D ,\nS003,,B\n"


def _write(dirpath: str, name: str, text: str) -> Path:
    path = Path(dirpath) / name
    path.write_bytes(text.encode("utf-8"))
    return path


class TestHash(unittest.TestCase):
    """sha256_file và dataset_hash."""

    def test_sha256_known_vector(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = _write(d, "a.txt", "abc")
            self.assertEqual(
                sha256_file(p),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )

    def test_dataset_hash_deterministic_and_order_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            a, b = _write(d, "a", "1"), _write(d, "b", "2")
            self.assertEqual(dataset_hash([a, b]), dataset_hash([a, b]))
            self.assertNotEqual(dataset_hash([a, b]), dataset_hash([b, a]))


class TestLoadKey(unittest.TestCase):
    """load_key."""

    def test_ok_and_normalizes(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            key = load_key(_write(d, "k.csv", KEY_OK))
        self.assertEqual(list(key.index), ["1", "2"])
        self.assertEqual(list(key["correct_answer"]), ["d", "a"])
        self.assertEqual(list(key["num_choices"]), [4, 4])

    def test_invalid_inputs_raise(self) -> None:
        bad = {
            "missing col": "question_id,correct_answer\n1,a\n",
            "dup id": "question_id,correct_answer,num_choices\n1,a,4\n1,b,4\n",
            "empty id": "question_id,correct_answer,num_choices\n,a,4\n",
            "answer out of range": "question_id,correct_answer,num_choices\n1,e,4\n",
            "non-int choices": "question_id,correct_answer,num_choices\n1,a,x\n",
            "choices too small": "question_id,correct_answer,num_choices\n1,a,1\n",
        }
        with tempfile.TemporaryDirectory() as d:
            for name, text in bad.items():
                with self.subTest(name), self.assertRaises(ValueError):
                    load_key(_write(d, "k.csv", text))


class TestLoadResponses(unittest.TestCase):
    """load_responses."""

    def test_ok_strips_lowercases_keeps_blank(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            r = load_responses(_write(d, "r.csv", RESP_OK))
        self.assertEqual(list(r.index), ["S001", "S002", "S003"])
        self.assertEqual(list(r.columns), ["1", "2"])
        self.assertEqual(r.loc["S002", "1"], "d")
        self.assertEqual(r.loc["S002", "2"], "")
        self.assertEqual(r.loc["S003", "2"], "b")
        self.assertEqual(r.index.name, "student")

    def test_invalid_inputs_raise(self) -> None:
        bad = {
            "no student col": "id,1\nS1,a\n",
            "dup student": "student,1\nS1,a\nS1,b\n",
            "empty student": "student,1\n,a\n",
            "no items": "student\nS1\n",
        }
        with tempfile.TemporaryDirectory() as d:
            for name, text in bad.items():
                with self.subTest(name), self.assertRaises(ValueError):
                    load_responses(_write(d, "r.csv", text))


class TestDownloadFile(unittest.TestCase):
    """download_file (dùng file:// để không cần mạng)."""

    def test_download_with_correct_hash(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            src = _write(d, "src.txt", "abc")
            dest = Path(d) / "sub" / "dir" / "out.txt"
            out = download_file(src.as_uri(), dest, sha256_file(src))
            self.assertEqual(out.read_bytes(), b"abc")
            self.assertFalse(dest.with_name("out.txt.part").exists())

    def test_hash_mismatch_raises_and_leaves_no_file(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            src = _write(d, "src.txt", "abc")
            dest = Path(d) / "out.txt"
            with self.assertRaises(ValueError):
                download_file(src.as_uri(), dest, "0" * 64)
            self.assertFalse(dest.exists())
            self.assertFalse(dest.with_name("out.txt.part").exists())

    def test_disallowed_scheme_raises(self) -> None:
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
            download_file("http://example.com/x", Path(d) / "x")


if __name__ == "__main__":
    unittest.main()
