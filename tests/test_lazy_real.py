"""Never materialize the full real series: compute only one selected frame."""
from pathlib import Path
from tempfile import TemporaryDirectory
import os
import unittest
from unittest.mock import patch

import dask.array as dsa
import numpy as np

from kglobal_analysis import KGlobalCase

DATA = Path(__file__).resolve().parents[2] / 'validation-data'
FIRST = Path(os.environ.get('KGLOBAL_VALIDATION_CASE', DATA / 'hcs_large_005'))
MULTI = Path(os.environ.get('KGLOBAL_MULTI_VALIDATION_CASE', DATA / 'hcs_large_multi'))


@unittest.skipUnless(FIRST.is_dir() and MULTI.is_dir(), 'real fixture directories unavailable')
class RealLazyMovieTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        (root/'param_hcs_large').symlink_to((FIRST/'param_hcs_large').resolve(strict=True))
        for suffix, directory in (('004', MULTI), ('005', FIRST), ('006', MULTI)):
            names = [f'movie.bx.{suffix}', f'movie.log.{suffix}', f'p3d.stdout.{suffix}']
            if suffix == '004':
                names.append('movie.jihpar.004')
            for name in names:
                (root/name).symlink_to((directory/name).resolve(strict=True))
        self.case = KGlobalCase(root)

    def compute_one(self, selected, variable, suffix, local_index):
        original = np.fromfile
        calls = []
        def bounded(stream, *, dtype, count):
            self.assertEqual(Path(stream.name).name, f'movie.{variable}.{suffix}')
            self.assertEqual(stream.tell(), local_index*67108864)
            self.assertEqual(count, 8192*4096)
            self.assertEqual(dtype, np.dtype('<i2'))
            data = original(stream, dtype=dtype, count=count)
            self.assertEqual(stream.tell(), (local_index+1)*67108864)
            self.assertEqual(data.nbytes, 67108864)
            calls.append(data.nbytes)
            return data
        self.assertEqual(selected.dims, ('x','y'))  # prohibit accidental full-series compute
        with patch('kglobal_analysis.movie.np.fromfile', side_effect=bounded), \
                patch.object(Path, 'read_bytes', side_effect=AssertionError('unbounded read')):
            result = selected.compute(scheduler='synchronous')
        self.assertEqual(calls, [67108864])
        return result

    def test_real_bx_sixty_frame_metadata_and_one_boundary_compute(self):
        original_open = Path.open
        def metadata_only(path, *args, **kwargs):
            self.assertTrue(path.name.startswith('p3d.stdout.'))
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', new=metadata_only), \
                patch('kglobal_analysis.movie.np.fromfile', side_effect=AssertionError('sample read')):
            series = self.case.movie('bx', byteorder='little')
            self.assertEqual(series.suffixes, ('004','005','006'))
            self.assertEqual(series.frame_count, 60)
            location = series.locate_time(6.05)
            self.assertEqual(location, ('006',0))
            # Derive the index from the actual constructed timeline, then verify 40.
            index = next(i for i,t in enumerate(series.times) if series.locate_time(t) == location)
            self.assertEqual(index, 40)
            da = series.to_xarray()
            self.assertIsInstance(da.data, dsa.Array)
            self.assertEqual(da.shape, (60,8192,4096))
            self.assertEqual(da.dims, ('time','x','y'))
            self.assertEqual(da.chunks, ((1,)*60,(8192,),(4096,)))
            np.testing.assert_array_equal(da.time.values, series.times)
            selected = da.sel(time=series.times[index])
            self.assertIsInstance(selected.data, dsa.Array)
            self.assertEqual(selected.source_segment.item(), '006')
            self.assertEqual(selected.local_frame_index.item(), 0)
            self.assertEqual(selected.global_frame_index.item(), index)
            with self.assertRaises(KeyError):
                da.sel(time=6.05)
        result = self.compute_one(selected, 'bx', '006', 0)
        eager = series.read_time_xarray(6.05)
        np.testing.assert_array_equal(result.data, eager.data)
        self.assertEqual(result.time.item(), eager.time.item())
        self.assertIsInstance(da.data, dsa.Array)

    def test_real_jihpar_uses_same_lazy_architecture(self):
        with patch('kglobal_analysis.movie.np.fromfile', side_effect=AssertionError('sample read')):
            series = self.case.movie('jihpar', byteorder='little')
            self.assertEqual(series.suffixes, ('004',))
            da = series.to_xarray()
            self.assertEqual(da.shape, (20,8192,4096))
            self.assertEqual(da.chunks, ((1,)*20,(8192,),(4096,)))
            np.testing.assert_array_equal(da.time.values, series.times)
            self.assertEqual(da.name, 'jihpar')
            selected = da.isel(time=19)
            self.assertIsInstance(selected.data, dsa.Array)
        result = self.compute_one(selected, 'jihpar', '004', 19)
        direct = self.case.read_movie_frame('jihpar', '004', 19, byteorder='little')
        np.testing.assert_array_equal(result.data, direct)
        self.assertEqual(result.source_segment.item(), '004')
        self.assertEqual(result.local_frame_index.item(), 19)
