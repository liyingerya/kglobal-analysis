"""Read-only integration validation; skipped when the external fixture is absent."""

from pathlib import Path
import os
import re
import struct
import unittest
from unittest.mock import patch

import numpy as np

from kglobal_analysis import KGlobalCase


DEFAULT_CASE = Path(__file__).resolve().parents[2] / "validation-data" / "hcs_large_005"
REAL_CASE = Path(os.environ.get("KGLOBAL_VALIDATION_CASE", DEFAULT_CASE))
POINTS = ((0, 0), (1, 0), (0, 1), (4096, 2048), (8191, 0), (0, 4095), (8191, 4095))


@unittest.skipUnless(REAL_CASE.is_dir(), "real hcs_large_005 fixture is not available")
class RealBxTests(unittest.TestCase):
    def test_real_first_and_last_frame_use_bounded_reads(self):
        case = KGlobalCase(REAL_CASE, parameter_file="param_hcs_large")
        self.assertEqual((case.parameters.Nx, case.parameters.Ny, case.parameters.Nz), (8192, 4096, 1))
        self.assertEqual(case.bx_frame_count("005"), 20)
        movie = case.index.segments["005"].movies["bx"]
        self.assertEqual(movie.stat().st_size, 1342177280)
        original_fromfile = np.fromfile
        samples = 8192 * 4096
        frame_bytes = samples * 2

        for index, (low, high) in ((0, (-1.3824, 1.50651)), (19, (-1.4281, 1.486))):
            with self.subTest(index=index):
                calls = []

                def bounded_read(stream, *, dtype, count):
                    self.assertEqual(count, samples)
                    self.assertEqual(dtype, np.dtype("<i2"))
                    self.assertEqual(stream.tell(), index * frame_bytes)
                    data = original_fromfile(stream, dtype=dtype, count=count)
                    self.assertEqual(stream.tell(), (index + 1) * frame_bytes)
                    self.assertEqual(data.nbytes, frame_bytes)
                    calls.append(count)
                    return data

                with patch("kglobal_analysis.bx.np.fromfile", side_effect=bounded_read), \
                        patch.object(Path, "read_bytes", side_effect=AssertionError("unbounded read")):
                    values = case.read_bx_frame("005", index, byteorder="little")
                self.assertEqual(calls, [samples])
                self.assertEqual(values.shape, (8192, 4096))
                self.assertEqual(values.dtype, np.dtype("float64"))
                self.assertTrue(values.flags.f_contiguous)
                self.assertTrue(values.flags.owndata)
                self.assertAlmostEqual(float(values.min()), low, places=12)
                self.assertAlmostEqual(float(values.max()), high, places=12)
                # Independent two-byte reads check layout and offsets without
                # using NumPy reshape or the production log/decoder functions.
                with movie.open("rb") as stream:
                    for x, y in POINTS:
                        stream.seek(index * frame_bytes + 2 * (x + 8192 * y))
                        q = struct.unpack("<h", stream.read(2))[0]
                        expected = low + (q + 32768) * (high - low) / 65535.0
                        self.assertAlmostEqual(values[x, y], expected, places=12)
                del values

    def test_real_stdout_has_twenty_actual_movie_events(self):
        # Independently validate the fixture's raw stdout event lines.
        text = (REAL_CASE / "p3d.stdout.005").read_text()
        times = [float(value) for value in re.findall(
            r"^\s*movie output, t=\s*([0-9.eE+-]+)\s*$", text, re.MULTILINE)]
        self.assertEqual(len(times), 20)
        self.assertAlmostEqual(times[0], 5.05, places=6)
        self.assertAlmostEqual(times[-1], 6.0, places=6)
        np.testing.assert_allclose(np.diff(times), 0.05, rtol=0, atol=2e-9)

    def test_real_segment_times_and_lookup(self):
        case = KGlobalCase(REAL_CASE, parameter_file="param_hcs_large")
        segment = case.bx_segment("005", byteorder="little")
        self.assertEqual(segment.suffix, "005")
        self.assertEqual(segment.frame_count, 20)
        self.assertEqual(len(segment.times), 20)
        self.assertAlmostEqual(segment.times[0], 5.05, places=6)
        self.assertAlmostEqual(segment.times[-1], 6.0, places=6)
        self.assertEqual(segment.cadence, 0.05)
        np.testing.assert_allclose(np.diff(segment.times), .05, rtol=0, atol=2e-9)
        for time, index in ((5.05, 0), (5.1, 1), (5.5, 9), (5.75, 14), (6.0, 19)):
            with self.subTest(time=time):
                self.assertEqual(segment.frame_index_at_time(time), index)
                self.assertEqual(segment.frame_index_at_time(segment.times[index]), index)
        with self.assertRaises(KeyError):
            segment.frame_index_at_time(5.075)

    def test_real_time_reads_match_index_reads(self):
        case = KGlobalCase(REAL_CASE, parameter_file="param_hcs_large")
        segment = case.bx_segment("005", byteorder="little")
        for time, index in ((5.05, 0), (6.0, 19)):
            with self.subTest(time=time):
                by_time = segment.read_time(time)
                by_index = case.read_bx_frame("005", index, byteorder="little")
                self.assertTrue(np.array_equal(by_time, by_index))
                del by_time, by_index


if __name__ == "__main__":
    unittest.main()
