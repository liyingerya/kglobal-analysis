"""Strict alignment and variable/time culling on small synthetic movies."""
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import struct
import unittest
from unittest.mock import patch

import dask.array as dsa
import numpy as np
import xarray as xr

from kglobal_analysis import KGlobalCase, MovieAlignmentError
from kglobal_analysis.series import MovieSeries

PARAM = '''#define nx 3
#define ny 2
#define nz 1
#define pex 1
#define pey 1
#define pez 1
#define movie_header "movie_kglobal3.0.h"
#define double_byte
#define dt .025
#define n_movieout 2
'''


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'param').write_text(PARAM)
        self.add_segment('090', [1.000000001,1.050000001], 0)
        self.add_segment('002', [1.100000001,1.150000001], 100)

    def add_segment(self, suffix, times, base, variables=('bx','jihpar'), samples=6):
        for name in variables:
            shift = 1000 if name == 'jihpar' else 0
            (self.root/f'movie.{name}.{suffix}').write_bytes(struct.pack(
                f'<{samples*len(times)}h', *range(base+shift,base+shift+samples*len(times))))
        (self.root/f'movie.log.{suffix}').write_text('-32768 32767\n'*18*len(times))
        (self.root/f'p3d.stdout.{suffix}').write_text(''.join(
            f'movie output, t= {time}\n' for time in times))

    def dataset(self, variables=('bx','jihpar')):
        return KGlobalCase(self.root).movie_dataset(variables, byteorder='little')

    def counted_compute(self, selected):
        original = np.fromfile
        calls = []
        def read(stream, *, dtype, count):
            calls.append((Path(stream.name).name,stream.tell(),count))
            return original(stream,dtype=dtype,count=count)
        with patch('kglobal_analysis.movie.np.fromfile',side_effect=read):
            result = selected.compute(scheduler='synchronous')
        return result, Counter(calls)

    def test_zero_read_construction_order_shared_coordinates_and_attrs(self):
        case = KGlobalCase(self.root)
        original = Path.open
        def guard(path,*args,**kwargs):
            self.assertTrue(path.name.startswith('p3d.stdout.'))
            return original(path,*args,**kwargs)
        with patch.object(Path,'open',new=guard), patch('kglobal_analysis.movie.np.fromfile',side_effect=AssertionError('read')):
            ds = case.movie_dataset(iter(['jihpar','bx']),byteorder='little')
            self.assertIsInstance(ds,xr.Dataset)
            self.assertEqual(list(ds.data_vars),['jihpar','bx'])
            self.assertEqual(dict(ds.sizes),{'time':4,'x':3,'y':2})
            selected = ds.isel(time=2)
            for name in ds.data_vars:
                self.assertIsInstance(selected[name].data,dsa.Array)
                self.assertEqual(ds[name].dims,('time','x','y'))
                self.assertEqual(ds[name].chunks,((1,1,1,1),(3,),(2,)))
                self.assertEqual(ds[name].attrs['storage_name'],name)
            self.assertEqual(set(ds.attrs),{'movie_header','encoding','storage_order'})
            self.assertEqual(set(ds.coords),{'time','source_segment','local_frame_index','global_frame_index'})
            for coord in ds.coords.values():
                self.assertNotIsInstance(coord.data,dsa.Array)
            np.testing.assert_array_equal(ds.source_segment.values,['090','090','002','002'])
            np.testing.assert_array_equal(ds.local_frame_index.values,[0,1,0,1])
            self.assertEqual(selected.source_segment.item(),'002')
            self.assertEqual(ds.time[0].item(),1.000000001)
            with self.assertRaises(KeyError):
                ds.sel(time=1.1)
            self.assertEqual(ds.sel(time=ds.time[2].item()).global_frame_index.item(),2)

    def test_one_time_reads_one_frame_per_variable(self):
        result,calls = self.counted_compute(self.dataset().isel(time=2))
        self.assertEqual(calls,Counter([('movie.bx.002',0,6),('movie.jihpar.002',0,6)]))
        self.assertEqual(result.bx[0,0].item(),100.)
        self.assertEqual(result.jihpar[0,0].item(),1100.)

    def test_unselected_variable_is_culled(self):
        result,calls = self.counted_compute(self.dataset()['bx'].isel(time=3))
        self.assertEqual(calls,Counter([('movie.bx.002',12,6)]))
        self.assertEqual(result[0,0].item(),106.)

    def test_two_variables_two_nonadjacent_times(self):
        result,calls = self.counted_compute(self.dataset()[['jihpar','bx']].isel(time=[3,0]))
        self.assertEqual(calls,Counter([(f'movie.{name}.{suffix}',offset,6)
                                      for name in ('bx','jihpar') for suffix,offset in (('002',12),('090',0))]))
        self.assertEqual(result.jihpar[0,0,0].item(),1106.)
        self.assertEqual(result.bx[1,2,1].item(),5.)

    def test_variable_input_validation(self):
        for names, error in (([],ValueError),(['bx','bx'],ValueError),('bx',TypeError),
                             ({'bx','jihpar'},TypeError),([1],TypeError),(['unknown'],ValueError)):
            with self.subTest(names=names), patch('kglobal_analysis.movie.np.fromfile',side_effect=AssertionError('read')):
                with self.assertRaises(error):
                    self.dataset(names)
        self.assertEqual(list(self.dataset(['bx']).data_vars),['bx'])
        with self.assertRaisesRegex(MovieAlignmentError,"'ni'.*no segments"):
            self.dataset(['bx','ni'])

    def test_missing_segment_rejected_before_graph_or_samples(self):
        (self.root/'movie.jihpar.002').unlink()
        with patch.object(MovieSeries,'to_xarray',side_effect=AssertionError('graph constructed')), \
                patch('kglobal_analysis.movie.np.fromfile',side_effect=AssertionError('read')):
            with self.assertRaisesRegex(MovieAlignmentError,"Frame count mismatch.*'bx'.*'jihpar'"):
                self.dataset()

    def test_frame_count_stdout_mismatch_is_alignment_error(self):
        (self.root/'movie.jihpar.002').write_bytes(struct.pack('<6h',*range(6)))
        with patch('kglobal_analysis.movie.np.fromfile',side_effect=AssertionError('read')):
            with self.assertRaisesRegex(MovieAlignmentError,"jihpar.*timestamps"):
                self.dataset()

    def test_exact_time_and_source_mapping_mismatches(self):
        # With shared suffix stdout, variables share timestamps. Separate suffixes
        # let each have independently valid but different time/provenance metadata.
        for path in self.root.glob('movie.jihpar.*'):
            path.unlink()
        for shift, reason in ((1e-9,'timestamp'),(0.,'provenance')):
            self.add_segment('800',[1.000000001+shift,1.050000001+shift],0,('jihpar',))
            self.add_segment('801',[1.100000001+shift,1.150000001+shift],100,('jihpar',))
            with patch.object(MovieSeries,'to_xarray',side_effect=AssertionError('graph constructed')), \
                    patch('kglobal_analysis.movie.np.fromfile',side_effect=AssertionError('read')):
                with self.assertRaisesRegex(MovieAlignmentError,reason):
                    self.dataset()

    def test_local_frame_mapping_is_also_checked(self):
        case = KGlobalCase(self.root)
        bx = case.movie('bx',byteorder='little')
        other = case.movie('jihpar',byteorder='little')
        # Such a local-index change cannot arise from the current shared stdout
        # files; inject it to independently check the provenance comparison.
        other._locations = (('090',1),) + other._locations[1:]
        with patch.object(case,'movie',side_effect=[bx,other]), \
                patch.object(MovieSeries,'to_xarray',side_effect=AssertionError('graph constructed')):
            with self.assertRaisesRegex(MovieAlignmentError,'provenance'):
                case.movie_dataset(['bx','jihpar'],byteorder='little')

    def test_synthetic_volume_and_single_variable_culling(self):
        (self.root/'param').write_text(PARAM.replace('#define nz 1','#define nz 2'))
        self.add_segment('090',[1.000000001,1.050000001],0,samples=12)
        self.add_segment('002',[1.100000001,1.150000001],100,samples=12)
        with patch('kglobal_analysis.movie.np.fromfile',side_effect=AssertionError('read')):
            ds = self.dataset()
            self.assertEqual(dict(ds.sizes),{'time':4,'x':3,'y':2,'z':2})
            for name in ds.data_vars:
                self.assertIsInstance(ds[name].data,dsa.Array)
                self.assertEqual(ds[name].dims,('time','x','y','z'))
            self.assertTrue({'x','y','z'}.isdisjoint(ds.coords))
            selected = ds['jihpar'].isel(time=3)
        result,calls = self.counted_compute(selected)
        self.assertEqual(calls,Counter([('movie.jihpar.002',24,12)]))
        for x,y,z in ((0,0,0),(1,0,0),(0,1,0),(0,0,1),(2,1,1)):
            self.assertEqual(result[x,y,z].item(),1112+x+3*(y+2*z))

    def test_spatial_point_still_reads_whole_frame(self):
        result,calls = self.counted_compute(self.dataset()['bx'].isel(time=3,x=1,y=1))
        self.assertEqual(calls,Counter([('movie.bx.002',12,6)]))
        self.assertEqual(result.item(),110.)
