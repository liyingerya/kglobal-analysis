from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from kglobal_analysis.index import parse_filename, scan_case


class FilenameTests(unittest.TestCase):
    def test_supported_names(self):
        for name, kind, suffix, variable in [
            ("movie.bx.000", "movie", "000", "bx"),
            ("movie.ni.015", "movie", "015", "ni"),
            ("movie.pehpar.015", "movie", "015", "pehpar"),
            ("movie.ni2.1000", "movie", "1000", "ni2"),
            ("movie.log.015", "log", "015", None),
            ("p3d.stdout.015", "stdout", "015", None),
        ]:
            with self.subTest(name=name):
                record = parse_filename(name)
                self.assertIsNotNone(record)
                self.assertEqual((record.kind, record.suffix, record.variable),
                                 (kind, suffix, variable))

    def test_unrelated_or_partial_names_are_ignored(self):
        for name in ("movie.bx.000~", "movie.bx.000.bak", "movie.bx.x", "movie..001",
                     "movie.bx.-1", "p3d-001.015", "bx", "movie.bx.000\n",
                     "nested/movie.bx.000", "movie.bx.000.extra"):
            with self.subTest(name=name):
                self.assertIsNone(parse_filename(name))

    def test_discovery_is_nonrecursive_and_supports_gaps(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            for name in ("movie.bx.000", "movie.ni.015", "movie.bx.015",
                         "movie.log.015", "p3d.stdout.015", "p3d.stdout.100", "movie.log.042",
                         "movie.bx.003~"):
                (root / name).touch()
            (root / "movie.bz.999").mkdir()
            (root / "movie.bz.999" / "movie.bx.020").touch()
            index = scan_case(root)
            self.assertEqual(index.suffixes, ("000", "015", "042", "100"))
            self.assertEqual(index.variables, ("bx", "ni"))
            self.assertEqual(index.segments["000"].variables, ("bx",))
            self.assertIsNone(index.segments["000"].movie_log)
            self.assertEqual(index.segments["015"].movie_log, root / "movie.log.015")
            self.assertEqual(index.segments["015"].stdout, root / "p3d.stdout.015")
            self.assertEqual(index.segments["100"].variables, ())
            self.assertIsNone(index.segments["100"].movie_log)
            self.assertEqual(index.segments["100"].stdout, root / "p3d.stdout.100")
            self.assertEqual(index.segments["042"].variables, ())
            self.assertEqual(index.segments["042"].movie_log, root / "movie.log.042")
            self.assertIsNone(index.segments["042"].stdout)

    def test_suffix_sorting_preserves_spelling(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            for suffix in ("10", "2", "002"):
                (root / f"movie.bx.{suffix}").touch()
            self.assertEqual(scan_case(root).suffixes, ("002", "2", "10"))


if __name__ == "__main__":
    unittest.main()
