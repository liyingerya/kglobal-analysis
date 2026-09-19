"""One-frame xarray integration checks using read-only real movie fixtures."""
from pathlib import Path
from tempfile import TemporaryDirectory
import os
import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr

from kglobal_analysis import KGlobalCase

DATA = Path(__file__).resolve().parents[2] / 'validation-data'
FIRST = Path(os.environ.get('KGLOBAL_VALIDATION_CASE', DATA / 'hcs_large_005'))
MULTI = Path(os.environ.get('KGLOBAL_MULTI_VALIDATION_CASE', DATA / 'hcs_large_multi'))


@unittest.skipUnless(FIRST.is_dir() and MULTI.is_dir(), 'real fixture directories unavailable')
class RealFrameXarrayTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        (root / 'param_hcs_large').symlink_to((FIRST / 'param_hcs_large').resolve(strict=True))
        for suffix, directory, variable in (('004', MULTI, 'jihpar'),
                                             ('005', FIRST, 'bx'), ('006', MULTI, 'bx')):
            for name in (f'movie.{variable}.{suffix}', f'movie.log.{suffix}', f'p3d.stdout.{suffix}'):
                (root / name).symlink_to((directory / name).resolve(strict=True))
        self.case = KGlobalCase(root)

    def check_frame(self, variable, time, suffix, local_index, global_index):
        series = self.case.movie(variable, byteorder='little')
        original = np.fromfile
        calls = []

        def bounded(stream, *, dtype, count):
            self.assertEqual(Path(stream.name).name, f'movie.{variable}.{suffix}')
            self.assertEqual(stream.tell(), local_index*67108864)
            self.assertEqual(count, 8192*4096)
            self.assertEqual(dtype, np.dtype('<i2'))
            result = original(stream, dtype=dtype, count=count)
            self.assertEqual(result.nbytes, 67108864)
            self.assertEqual(stream.tell(), (local_index+1)*67108864)
            calls.append(result.nbytes)
            return result

        with patch('kglobal_analysis.movie.np.fromfile', side_effect=bounded), \
                patch.object(Path, 'read_bytes', side_effect=AssertionError('unbounded read')):
            da = series.read_time_xarray(time)
            self.assertEqual(calls, [67108864])
            direct = self.case.read_movie_frame(variable, suffix, local_index, byteorder='little')
            self.assertEqual(calls, [67108864, 67108864])
        self.assertIsInstance(da, xr.DataArray)
        self.assertEqual(da.name, variable)
        self.assertEqual(da.dims, ('x', 'y'))
        self.assertEqual(da.shape, direct.shape)
        self.assertEqual(da.shape, (8192, 4096))
        self.assertEqual(da.dtype, np.dtype('float64'))
        self.assertEqual(set(da.coords), {'time'})
        self.assertEqual(da.coords['time'].dims, ())
        self.assertEqual(da.coords['time'].item(), series.times[global_index])
        self.assertAlmostEqual(da.coords['time'].item(), time, places=6)
        self.assertEqual(da.attrs['source_segment'], suffix)
        self.assertEqual(da.attrs['local_frame_index'], local_index)
        self.assertEqual(da.attrs['global_frame_index'], global_index)
        self.assertEqual(da.attrs['storage_name'], variable)
        np.testing.assert_array_equal(da.values, direct)
        self.assertEqual(da.isel(x=100, y=50).item(), direct[100, 50])
        self.assertIsInstance(da.data, np.ndarray)
        self.assertIsNone(da.chunks)

    def test_real_bx_frame_after_segment_boundary(self):
        self.check_frame('bx', 6.05, '006', 0, 20)

    def test_real_jihpar_frame_with_nonzero_offset(self):
        self.check_frame('jihpar', 5., '004', 19, 19)
