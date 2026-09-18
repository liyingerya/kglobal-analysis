from pathlib import Path
from tempfile import TemporaryDirectory
import struct
import unittest
from unittest.mock import patch

from kglobal_analysis import BxSeries, KGlobalCase, TimeMetadataError


PARAM = '''#define nx 2
#define ny 2
#define nz 1
#define pex 1
#define pey 1
#define pez 1
#define double_byte
#define movie_header "movie_kglobal3.0.h"
#define dt .025
#define n_movieout 2
'''


class SeriesTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "param").write_text(PARAM)
        # Deliberately reverse suffix order relative to physical time.
        self.add_segment("090", [10.0, 10.05])
        self.add_segment("002", [10.1, 10.15])

    def add_segment(self, suffix, times):
        (self.root / f"movie.bx.{suffix}").write_bytes(struct.pack("<4h", 0, 1, 2, 3) * len(times))
        (self.root / f"movie.log.{suffix}").write_text("-1 1\n" * 18 * len(times))
        (self.root / f"p3d.stdout.{suffix}").write_text("two-byte movie output\n" + "".join(
            f" movie output, t= {time}\n" for time in times))

    def series(self):
        return KGlobalCase(self.root).bx(byteorder="little")

    def test_order_count_and_global_mapping(self):
        bx = self.series()
        self.assertIsInstance(bx, BxSeries)
        self.assertEqual(bx.suffixes, ("090", "002"))
        self.assertEqual(bx.frame_count, 4)
        self.assertEqual(bx.times, (10., 10.05, 10.1, 10.15))
        for time, location in zip(bx.times, [("090", 0), ("090", 1), ("002", 0), ("002", 1)]):
            self.assertEqual(bx.locate_time(time), location)

    def test_exact_boundary_crossing_and_no_nearest(self):
        bx = self.series()
        self.assertEqual(bx.locate_time(10.05), ("090", 1))
        self.assertEqual(bx.locate_time(10.10), ("002", 0))
        for time in (9., 10.075, 10.10001, 11.):
            with self.subTest(time=time), self.assertRaises(KeyError):
                bx.locate_time(time)
        for time in (float("nan"), float("inf")):
            with self.subTest(time=time), self.assertRaises(ValueError):
                bx.locate_time(time)

    def test_duplicate_and_overlapping_segments(self):
        for times in ([10.05, 10.1], [10.025, 10.075], [10., 10.05]):
            with self.subTest(times=times):
                self.add_segment("002", times)
                with self.assertRaisesRegex(TimeMetadataError, "Overlapping segments or duplicate"):
                    self.series()

    def test_nonmonotonic_segment_not_repaired_by_global_sort(self):
        self.add_segment("002", [10.15, 10.1])
        with self.assertRaisesRegex(TimeMetadataError, "strictly increasing"):
            self.series()

    def test_gap(self):
        self.add_segment("002", [10.2, 10.25])
        with self.assertRaisesRegex(TimeMetadataError, "Gap at 090 -> 002"):
            self.series()

    def test_boundary_shorter_than_cadence(self):
        self.add_segment("002", [10.075, 10.125])
        with self.assertRaisesRegex(TimeMetadataError, "Cadence discontinuity"):
            self.series()

    def test_boundary_roundoff_and_time_equality_tolerance(self):
        self.add_segment("090", [9.9999999, 10.049999899])
        self.add_segment("002", [10.099999898, 10.149999897])
        bx = self.series()
        self.assertEqual(bx.locate_time(10.05), ("090", 1))
        self.assertEqual(bx.locate_time(10.1), ("002", 0))
        self.assertEqual(bx.times[2], 10.099999898)

    def test_missing_and_invalid_stdout(self):
        stdout = self.root / "p3d.stdout.002"
        for contents in (None, "", "movie output, t= nan\n", "movie output, t= 10.1\n"):
            with self.subTest(contents=contents):
                if contents is None:
                    stdout.unlink()
                else:
                    stdout.write_text(contents)
                with self.assertRaisesRegex(TimeMetadataError, "Bx segment 002"):
                    self.series()

    def test_missing_nominal_cadence_errors_instead_of_guessing_gaps(self):
        (self.root / "param").write_text(PARAM.replace("#define dt .025\n", ""))
        with self.assertRaisesRegex(TimeMetadataError, "required to validate Bx segment boundaries"):
            self.series()

    def test_construction_and_times_never_open_binary_or_log(self):
        case = KGlobalCase(self.root)
        original_open = Path.open
        opened = []

        def guard(path, *args, **kwargs):
            self.assertTrue(path.name.startswith("p3d.stdout."))
            opened.append(path.name)
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", new=guard), \
                patch("kglobal_analysis.bx.np.fromfile", side_effect=AssertionError("binary read")):
            bx = case.bx(byteorder="little")
            self.assertEqual(bx.frame_count, 4)
            self.assertEqual(len(bx.times), 4)
            self.assertEqual(bx.locate_time(10.1), ("002", 0))
        self.assertEqual(sorted(opened), ["p3d.stdout.002", "p3d.stdout.090"])

    def test_read_time_delegates_exactly_one_frame(self):
        case = KGlobalCase(self.root)
        bx = case.bx(byteorder="little")
        result = object()
        with patch.object(case, "read_bx_frame", return_value=result) as reader:
            self.assertIs(bx.read_time(10.1), result)
            reader.assert_called_once_with("002", 0, byteorder="little")
            reader.reset_mock()
            with self.assertRaises(KeyError):
                bx.read_time(10.075)
            reader.assert_not_called()

    def test_companion_only_segments_do_not_participate(self):
        (self.root / "movie.log.999").touch()
        (self.root / "p3d.stdout.888").write_text("invalid")
        self.assertEqual(self.series().suffixes, ("090", "002"))

    def test_empty_series(self):
        for path in self.root.glob("movie.bx.*"):
            path.unlink()
        with self.assertRaisesRegex(ValueError, "No Bx segments"):
            self.series()

    def test_single_frame_segments_use_global_tolerance(self):
        (self.root / "param").write_text(PARAM.replace("#define dt .025", "#define dt 5e-9"))
        self.add_segment("090", [0.0])
        self.add_segment("002", [1e-8])
        bx = self.series()
        self.assertEqual(bx.locate_time(1e-8), ("002", 0))
        with self.assertRaises(KeyError):
            bx.locate_time(5e-9)


if __name__ == "__main__":
    unittest.main()
