from pathlib import Path
from tempfile import TemporaryDirectory
import struct
import unittest
from unittest.mock import patch

import numpy as np

from kglobal_analysis import KGlobalCase
from kglobal_analysis.bx import frame_count_from_size, read_bx_minmax, _inverse_quantization


PARAM = '''#define nx 3
#define ny 2
#define nz 1
#define pex 1
#define pey 1
#define pez 1
#define movie_header "movie_kglobal3.0.h"
#define double_byte
'''


def log_text(ranges):
    return "".join(f"{low:.6E} {high:.6E}\n" if entry == 4 else "-9.0 9.0\n"
                   for low, high in ranges for entry in range(18))


class BxTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "param").write_text(PARAM)
        self.movie = self.root / "movie.bx.005"
        self.log = self.root / "movie.log.005"
        self.codes = (-32768, -100, 0, 100, 12345, 32767,
                      32767, 1000, 0, -1000, -12345, -32768)
        self.movie.write_bytes(struct.pack("<12h", *self.codes))
        self.log.write_text(log_text([(-2.0, 3.0), (-5.0, 7.0)]))

    def test_frame_count_and_remainder(self):
        self.assertEqual(frame_count_from_size(1342177280, 8192, 4096), 20)
        self.assertEqual(KGlobalCase(self.root).bx_frame_count("005"), 2)
        for size in (0, -1, 1342177281, 67108863):
            with self.subTest(size=size), self.assertRaises(ValueError):
                frame_count_from_size(size, 8192, 4096)
        with self.assertRaises(ValueError):
            frame_count_from_size(12, 0, 2)

    def test_frame_index_validation_before_data_read(self):
        case = KGlobalCase(self.root)
        with patch("kglobal_analysis.bx.np.fromfile", side_effect=AssertionError("read attempted")):
            for index in (-1, 2, 19):
                with self.subTest(index=index), self.assertRaises(IndexError):
                    case.read_bx_frame("005", index, byteorder="little")
            for index in (0.5, "0", True):
                with self.subTest(index=index), self.assertRaises(TypeError):
                    case.read_bx_frame("005", index, byteorder="little")

    def test_log_bx_entry_and_frame_selection(self):
        self.assertEqual(read_bx_minmax(self.log, 0, 2), (-2.0, 3.0))
        self.assertEqual(read_bx_minmax(self.log, 1, 2), (-5.0, 7.0))
        self.log.write_text(self.log.read_text().replace("E", "D"))
        self.assertEqual(read_bx_minmax(self.log, 1, 2), (-5.0, 7.0))

    def test_fixed_width_log_without_separator(self):
        lines = [" 0.0 1.0\n"] * 18
        lines[4] = "-0.123456E+100-0.100000E+100\n"
        self.assertEqual(len(lines[4].rstrip()), 28)
        self.log.write_text("".join(lines))
        self.assertEqual(read_bx_minmax(self.log, 0, 1), (-0.123456e100, -0.1e100))

    def test_log_rejects_wrong_count_and_invalid_ranges(self):
        valid = log_text([(-2.0, 3.0), (-5.0, 7.0)])
        for text in ("", "".join(valid.splitlines(keepends=True)[:-1]), valid + "0 1\n",
                     valid.replace("-9.0 9.0", "nan 9.0", 1),
                     valid.replace("-9.0 9.0", "9.0 -9.0", 1),
                     valid.replace("-9.0 9.0", "broken", 1),
                     valid.replace("-9.0 9.0", "0 inf", 1)):
            with self.subTest(text=text[:30]):
                self.log.write_text(text)
                with self.assertRaises(ValueError):
                    read_bx_minmax(self.log, 0, 2)

    def test_inverse_quantization_and_constant_field(self):
        encoded = np.array([-32768, -1, 0, 32767], dtype=np.int16)
        values = _inverse_quantization(encoded, -2.0, 3.0)
        expected = [-2.0 + (int(q) + 32768) * 5.0 / 65535.0 for q in encoded]
        np.testing.assert_allclose(values, expected, rtol=0, atol=1e-15)
        self.assertEqual(values.dtype, np.dtype("float64"))
        self.assertEqual(values[0], -2.0)
        self.assertEqual(values[-1], 3.0)
        np.testing.assert_array_equal(_inverse_quantization(encoded, 7.0, 7.0), [7.0] * 4)

    def test_order_shape_and_nonzero_frame_offset(self):
        case = KGlobalCase(self.root)
        for index, limits in enumerate([(-2.0, 3.0), (-5.0, 7.0)]):
            values = case.read_bx_frame("005", index, byteorder="little")
            self.assertEqual(values.shape, (3, 2))
            self.assertTrue(values.flags.f_contiguous)
            self.assertTrue(values.flags.owndata)
            low, high = limits
            for x in range(3):
                for y in range(2):
                    q = self.codes[index * 6 + x + 3 * y]
                    self.assertAlmostEqual(values[x, y], low + (q + 32768) * (high - low) / 65535)

    def test_byteorder_is_explicit(self):
        case = KGlobalCase(self.root)
        with self.assertRaises(TypeError):
            case.read_bx_frame("005", 0)
        with self.assertRaises(ValueError):
            case.read_bx_frame("005", 0, byteorder="native")
        little = case.read_bx_frame("005", 0, byteorder="little")
        self.movie.write_bytes(struct.pack(">12h", *self.codes))
        np.testing.assert_array_equal(case.read_bx_frame("005", 0, byteorder="big"), little)

    def test_unsupported_layouts(self):
        for text in (PARAM.replace("#define nz 1", "#define nz 2"),
                     PARAM.replace("#define double_byte", "#define four_byte"),
                     PARAM + "#define four_byte\n", PARAM + "#define heatfluxmovies\n",
                     PARAM + "#define mult_species\n", PARAM + '#define movie_header2 "other.h"\n',
                     PARAM.replace("movie_kglobal3.0.h", "movie_pic3.0.h"),
                     PARAM.replace("#define nx 3", "")):
            with self.subTest(text=text):
                (self.root / "param").write_text(text)
                with self.assertRaises(ValueError):
                    KGlobalCase(self.root).read_bx_frame("005", 0, byteorder="little")

    def test_missing_companions(self):
        with self.assertRaises(KeyError):
            KGlobalCase(self.root).read_bx_frame("015", 0, byteorder="little")
        self.log.unlink()
        with self.assertRaises(FileNotFoundError):
            KGlobalCase(self.root).read_bx_frame("005", 0, byteorder="little")
        self.movie.unlink()
        self.log.touch()
        with self.assertRaises(FileNotFoundError):
            KGlobalCase(self.root).read_bx_frame("005", 0, byteorder="little")

    def test_short_read_detected(self):
        case = KGlobalCase(self.root)
        with patch("kglobal_analysis.bx.np.fromfile", return_value=np.array([], dtype=np.int16)):
            with self.assertRaisesRegex(ValueError, "Short Bx frame"):
                case.read_bx_frame("005", 0, byteorder="little")


if __name__ == "__main__":
    unittest.main()
