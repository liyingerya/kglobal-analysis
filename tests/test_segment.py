from pathlib import Path
from tempfile import TemporaryDirectory
import struct
import unittest
from unittest.mock import patch

import numpy as np

from kglobal_analysis import BxSegment, KGlobalCase, TimeMetadataError
from kglobal_analysis.times import read_movie_times


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


class SegmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.param = self.root / "param"
        self.param.write_text(PARAM)
        (self.root / "movie.bx.005").write_bytes(struct.pack("<12h", *range(12)))
        (self.root / "movie.log.005").write_text("-2 2\n" * 54)
        self.stdout = self.root / "p3d.stdout.005"
        self.write_times([5.05, 5.10, 5.15])
        self.case = KGlobalCase(self.root)

    def write_times(self, times):
        self.stdout.write_text("two-byte movie output\n" + "".join(
            f" movie output, t= {time}\n" for time in times))

    def segment(self):
        return KGlobalCase(self.root).bx_segment("005", byteorder="little")

    def test_parsing_actual_events_only_and_fortran_numbers(self):
        self.stdout.write_text('''two-byte movie output
There will be movie output, t= 999
"movie output, t= 999"
  movie output, t= +5.05D+00
movie output, t= 5.1e0
movie output, t= 5.15
''')
        self.assertEqual(read_movie_times(self.stdout), (5.05, 5.1, 5.15))

    def test_suffix_count_times_and_cadence(self):
        segment = self.segment()
        self.assertIsInstance(segment, BxSegment)
        self.assertEqual(segment.suffix, "005")
        self.assertEqual(segment.frame_count, 3)
        self.assertEqual(segment.times, (5.05, 5.1, 5.15))
        self.assertEqual(segment.cadence, .05)

    def test_time_metadata_access_does_not_read_binary(self):
        with patch("kglobal_analysis.bx.np.fromfile", side_effect=AssertionError("binary read")):
            segment = self.segment()
            self.assertEqual(segment.times, (5.05, 5.1, 5.15))
            self.assertEqual(segment.frame_index_at_time(5.1), 1)

    def test_timestamp_count_mismatch(self):
        for times in ([], [5.05], [5.05, 5.1, 5.15, 5.2]):
            with self.subTest(times=times):
                self.write_times(times)
                segment = self.segment()
                with self.assertRaisesRegex(TimeMetadataError, "timestamps; Bx has 3 frames"):
                    segment.read_time(5.05)
                # Time metadata errors do not disable index-based reading.
                self.assertEqual(segment.read_frame(0).shape, (2, 2))

    def test_nonmonotonic_and_duplicate_times(self):
        for times in ([5.05, 5.05, 5.1], [5.1, 5.05, 5.15]):
            with self.subTest(times=times):
                self.write_times(times)
                with self.assertRaisesRegex(TimeMetadataError, "strictly increasing"):
                    _ = self.segment().times

    def test_nonfinite_and_malformed_timestamp(self):
        for value in ("nan", "inf", "-inf", "1e999", "garbage", "", "5.05 trailing junk"):
            with self.subTest(value=value):
                self.write_times([value, 5.1, 5.15])
                with self.assertRaises(TimeMetadataError):
                    _ = self.segment().times

    def test_cadence_mismatch(self):
        self.write_times([5.05, 5.1, 5.2])
        with self.assertRaisesRegex(TimeMetadataError, "cadence mismatch"):
            _ = self.segment().times

    def test_cadence_roundoff_allowed_without_replacing_times(self):
        times = (5.0499998724262696, 5.0999998711631633, 5.1499998699000571)
        self.write_times(times)
        segment = self.segment()
        self.assertEqual(segment.times, times)
        self.assertEqual(segment.frame_index_at_time(5.05), 0)
        self.assertEqual(segment.frame_index_at_time(5.1), 1)

    def test_missing_cadence_metadata_does_not_invent_times(self):
        for field in ("#define dt .025\n", "#define n_movieout 2\n"):
            with self.subTest(field=field):
                self.param.write_text(PARAM.replace(field, ""))
                self.write_times([7.0, 7.2, 7.9])
                segment = self.segment()
                self.assertIsNone(segment.cadence)
                self.assertEqual(segment.times, (7.0, 7.2, 7.9))
                self.assertEqual(segment.frame_index_at_time(7.2), 1)

    def test_exact_time_lookup_and_missing_time(self):
        segment = self.segment()
        for index, time in enumerate(segment.times):
            self.assertEqual(segment.frame_index_at_time(time), index)
        for time in (0.0, 5.0, 5.075, 5.10001, 6.0):
            with self.subTest(time=time), self.assertRaises(KeyError):
                segment.frame_index_at_time(time)
        for time in (float("nan"), float("inf")):
            with self.subTest(time=time), self.assertRaises(ValueError):
                segment.frame_index_at_time(time)

    def test_lookup_tolerance_cannot_select_neighbor_for_small_cadence(self):
        self.param.write_text(PARAM.replace("#define dt .025", "#define dt 5e-9"))
        self.write_times([0.0, 1e-8, 2e-8])
        segment = self.segment()
        self.assertEqual(segment.frame_index_at_time(1e-8), 1)
        with self.assertRaises(KeyError):
            segment.frame_index_at_time(1.5e-8)

    def test_missing_stdout_errors_only_on_time_access(self):
        self.stdout.unlink()
        segment = self.segment()
        with self.assertRaisesRegex(TimeMetadataError, "No p3d.stdout.005"):
            _ = segment.times
        with self.assertRaises(TimeMetadataError):
            segment.read_time(5.05)
        self.assertEqual(segment.read_frame(0).shape, (2, 2))
        self.assertEqual(self.case.read_bx_frame("005", 0, byteorder="little").shape, (2, 2))

    def test_stdout_deleted_after_case_discovery(self):
        segment = self.case.bx_segment("005", byteorder="little")
        self.stdout.unlink()
        with self.assertRaisesRegex(TimeMetadataError, "Cannot read movie timestamps"):
            _ = segment.times

    def test_read_time_delegates_correct_frame(self):
        segment = self.case.bx_segment("005", byteorder="little")
        expected = object()
        with patch.object(self.case, "read_bx_frame", return_value=expected) as reader:
            self.assertIs(segment.read_time(5.1), expected)
            reader.assert_called_once_with("005", 1, byteorder="little")
            reader.reset_mock()
            with self.assertRaises(KeyError):
                segment.read_time(5.075)
            reader.assert_not_called()

    def test_index_and_time_reads_equal(self):
        segment = self.segment()
        np.testing.assert_array_equal(segment.read_time(5.15), segment.read_frame(2))

    def test_invalid_byteorder(self):
        with self.assertRaises(ValueError):
            self.case.bx_segment("005", byteorder="native")


if __name__ == "__main__":
    unittest.main()
