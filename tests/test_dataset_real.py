"""Real 004 dataset: only selected frames are ever materialized."""
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import os
import unittest
from unittest.mock import patch

import dask.array as dsa
import numpy as np

from kglobal_analysis import KGlobalCase

DATA = Path(__file__).resolve().parents[2]/'validation-data'
FIRST = Path(os.environ.get('KGLOBAL_VALIDATION_CASE',DATA/'hcs_large_005'))
MULTI = Path(os.environ.get('KGLOBAL_MULTI_VALIDATION_CASE',DATA/'hcs_large_multi'))


@unittest.skipUnless(FIRST.is_dir() and MULTI.is_dir(),'real fixture directories unavailable')
class RealDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        (root/'param_hcs_large').symlink_to((FIRST/'param_hcs_large').resolve(strict=True))
        for name in ('movie.bx.004','movie.jihpar.004','movie.log.004','p3d.stdout.004'):
            (root/name).symlink_to((MULTI/name).resolve(strict=True))
        self.case = KGlobalCase(root)

    def construct(self):
        original = Path.open
        def guard(path,*args,**kwargs):
            self.assertTrue(path.name.startswith('p3d.stdout.'))
            return original(path,*args,**kwargs)
        with patch.object(Path,'open',new=guard), \
                patch('kglobal_analysis.movie.np.fromfile',side_effect=AssertionError('sample read')):
            ds = self.case.movie_dataset(['bx','jihpar'],byteorder='little')
            self.assertEqual(list(ds.data_vars),['bx','jihpar'])
            self.assertEqual(dict(ds.sizes),{'time':20,'x':8192,'y':4096})
            for name in ds.data_vars:
                self.assertEqual(ds[name].shape,(20,8192,4096))
                self.assertIsInstance(ds[name].data,dsa.Array)
                self.assertEqual(ds[name].chunks,((1,)*20,(8192,),(4096,)))
            times = self.case.movie('bx',byteorder='little').times
            np.testing.assert_array_equal(ds.time.values,times)
            self.assertEqual(ds.time[19].item(),times[19])
            selected = ds.isel(time=19)
            self.assertIsInstance(selected.bx.data,dsa.Array)
        return ds

    def compute_selected(self, selected, names):
        self.assertNotIn('time',selected.dims)  # Guard against full real dataset compute.
        original = np.fromfile
        calls = []
        def bounded(stream,*,dtype,count):
            filename = Path(stream.name).name
            self.assertIn(filename,[f'movie.{name}.004' for name in names])
            self.assertEqual(stream.tell(),19*67108864)
            self.assertEqual(count,8192*4096)
            self.assertEqual(dtype,np.dtype('<i2'))
            result = original(stream,dtype=dtype,count=count)
            self.assertEqual(result.nbytes,67108864)
            self.assertEqual(stream.tell(),20*67108864)
            calls.append(filename)
            return result
        with patch('kglobal_analysis.movie.np.fromfile',side_effect=bounded), \
                patch.object(Path,'read_bytes',side_effect=AssertionError('unbounded read')):
            result = selected.compute(scheduler='synchronous')
        self.assertEqual(Counter(calls),Counter(f'movie.{name}.004' for name in names))
        return result

    def test_real_two_variables_one_time(self):
        ds = self.construct()
        result = self.compute_selected(ds.isel(time=19),['bx','jihpar'])
        self.assertEqual(result.source_segment.item(),'004')
        self.assertEqual(result.local_frame_index.item(),19)
        self.assertAlmostEqual(result.time.item(),5.,places=6)
        for name in ('bx','jihpar'):
            direct = self.case.read_movie_frame(name,'004',19,byteorder='little')
            self.assertTrue(np.array_equal(result[name].data,direct))
            del direct

    def test_real_bx_selection_never_reads_jihpar(self):
        ds = self.construct()
        result = self.compute_selected(ds['bx'].isel(time=19),['bx'])
        direct = self.case.read_movie_frame('bx','004',19,byteorder='little')
        self.assertTrue(np.array_equal(result.data,direct))
