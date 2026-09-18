"""Real 005/006 timeline via temporary links; validation-data stays read-only."""

from pathlib import Path
from tempfile import TemporaryDirectory
import os
import unittest
from unittest.mock import patch

import numpy as np

from kglobal_analysis import KGlobalCase


DATA = Path(__file__).resolve().parents[2] / "validation-data"
FIRST = Path(os.environ.get("KGLOBAL_VALIDATION_CASE", DATA / "hcs_large_005"))
SECOND = Path(os.environ.get("KGLOBAL_MULTI_VALIDATION_CASE", DATA / "hcs_large_multi"))


@unittest.skipUnless(FIRST.is_dir() and SECOND.is_dir(), "real 005/006 fixture directories unavailable")
class RealSeriesTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "param_hcs_large").symlink_to((FIRST / "param_hcs_large").resolve())
        for suffix, directory in (("005", FIRST), ("006", SECOND)):
            for name in (f"movie.bx.{suffix}", f"movie.log.{suffix}", f"p3d.stdout.{suffix}"):
                source = (directory / name).resolve(strict=True)
                (self.root / name).symlink_to(source)
        self.case = KGlobalCase(self.root)

    def test_real_global_timeline_is_lazy_and_continuous(self):
        original_open = Path.open

        def guard(path, *args, **kwargs):
            self.assertTrue(path.name.startswith("p3d.stdout."))
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", new=guard), \
                patch("kglobal_analysis.bx.np.fromfile", side_effect=AssertionError("binary read")):
            bx = self.case.bx(byteorder="little")
            self.assertEqual(bx.suffixes, ("005", "006"))
            self.assertEqual(bx.frame_count, 40)
            self.assertEqual(len(bx.times), 40)
            self.assertAlmostEqual(bx.times[0], 5.05, places=6)
            self.assertAlmostEqual(bx.times[-1], 7., places=6)
            np.testing.assert_allclose(np.diff(bx.times), .05, rtol=0, atol=2e-9)
            self.assertEqual(bx.locate_time(6.), ("005", 19))
            self.assertEqual(bx.locate_time(6.05), ("006", 0))
            self.assertEqual(bx.locate_time(7.), ("006", 19))

    def test_real_boundary_reads_are_one_frame_and_match_direct_reads(self):
        bx = self.case.bx(byteorder="little")
        original_fromfile = np.fromfile
        for time, suffix, local_index in ((6., "005", 19), (6.05, "006", 0)):
            with self.subTest(time=time):
                calls = []

                def bounded_read(stream, *, dtype, count):
                    self.assertEqual(Path(stream.name).name, f"movie.bx.{suffix}")
                    self.assertEqual(stream.tell(), local_index * 67108864)
                    self.assertEqual(count, 8192 * 4096)
                    self.assertEqual(np.dtype(dtype).itemsize, 2)
                    data = original_fromfile(stream, dtype=dtype, count=count)
                    self.assertEqual(stream.tell(), (local_index + 1) * 67108864)
                    self.assertEqual(data.nbytes, 67108864)
                    calls.append(data.nbytes)
                    return data

                with patch("kglobal_analysis.bx.np.fromfile", side_effect=bounded_read):
                    by_time = bx.read_time(time)
                self.assertEqual(calls, [67108864])
                direct = self.case.read_bx_frame(suffix, local_index, byteorder="little")
                self.assertTrue(np.array_equal(by_time, direct))
                del by_time, direct


if __name__ == "__main__":
    unittest.main()
