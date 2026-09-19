"""Time-lazy graphs, task culling and unchanged geometry/time semantics."""
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import struct
import unittest
from unittest.mock import patch

import dask
import dask.array as dsa
import numpy as np

from kglobal_analysis import KGlobalCase

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


class LazyMovieTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'param').write_text(PARAM)
        self.write_movies()
        self.case = KGlobalCase(self.root)

    def write_movies(self, samples=6):
        for suffix, start, base in (('090', 1.000000001, 0), ('002', 1.100000001, 100)):
            for variable, shift in (('bx', 0), ('jihpar', 1000)):
                codes = range(base + shift, base + shift + 2*samples)
                (self.root / f'movie.{variable}.{suffix}').write_bytes(
                    struct.pack(f'<{samples*2}h', *codes))
            (self.root / f'movie.log.{suffix}').write_text('-32768 32767\n'*36)
            (self.root / f'p3d.stdout.{suffix}').write_text(
                f'two-byte movie output\nmovie output, t= {start}\nmovie output, t= {start+.05}\n')

    def lazy(self, variable='bx'):
        return self.case.movie(variable, byteorder='little').to_xarray()

    def test_construction_indexing_and_metadata_have_zero_sample_io(self):
        original_open = Path.open
        def metadata_only(path, *args, **kwargs):
            self.assertTrue(path.name.startswith('p3d.stdout.'))
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', new=metadata_only), \
                patch('kglobal_analysis.movie.np.fromfile', side_effect=AssertionError('sample read')):
            series = self.case.movie('bx', byteorder='little')
            da = series.to_xarray()
            selected = da.isel(time=2)
            self.assertIsInstance(da.data, dsa.Array)
            self.assertIsInstance(selected.data, dsa.Array)
            self.assertEqual(da.shape, (4, 3, 2))
            self.assertEqual(da.dims, ('time', 'x', 'y'))
            self.assertEqual(da.chunks, ((1,1,1,1), (3,), (2,)))
            np.testing.assert_array_equal(da.time.values, series.times)
            self.assertEqual(selected.source_segment.item(), '002')
            self.assertEqual(selected.local_frame_index.item(), 0)
            self.assertEqual(selected.global_frame_index.item(), 2)

    def test_one_time_compute_executes_exactly_one_frame(self):
        da = self.lazy()
        original = np.fromfile
        calls = []
        def bounded(stream, *, dtype, count):
            calls.append((Path(stream.name).name, stream.tell(), count))
            return original(stream, dtype=dtype, count=count)
        with patch('kglobal_analysis.movie.np.fromfile', side_effect=bounded):
            selected = da.isel(time=2)
            self.assertEqual(calls, [])
            result = selected.compute(scheduler='synchronous')
        self.assertEqual(calls, [('movie.bx.002', 0, 6)])
        np.testing.assert_array_equal(result.data, np.arange(100.,106.).reshape((3,2), order='F'))
        self.assertIsInstance(da.data, dsa.Array)

    def test_two_nonadjacent_times_cull_all_other_tasks(self):
        da = self.lazy('jihpar')
        original = np.fromfile
        calls = []
        def bounded(stream, *, dtype, count):
            calls.append((Path(stream.name).name, stream.tell(), count))
            return original(stream, dtype=dtype, count=count)
        with patch('kglobal_analysis.movie.np.fromfile', side_effect=bounded):
            selected = da.isel(time=[3,0])
            self.assertEqual(calls, [])
            result = selected.compute(scheduler='synchronous')
        self.assertEqual(Counter(calls), Counter([('movie.jihpar.002', 12, 6), ('movie.jihpar.090', 0, 6)]))
        np.testing.assert_array_equal(result.global_frame_index.values, [3,0])
        self.assertEqual(result.isel(time=0,x=0,y=0).item(), 1106.)
        self.assertEqual(result.isel(time=1,x=2,y=1).item(), 1005.)

    def test_load_and_values_only_materialize_selected_time(self):
        for action in ('load', 'values'):
            with self.subTest(action=action):
                da = self.lazy()
                with patch('kglobal_analysis.movie.np.fromfile', wraps=np.fromfile) as reader:
                    selected = da.isel(time=1)
                    reader.assert_not_called()
                    with dask.config.set(scheduler='synchronous'):
                        values = selected.load().data if action == 'load' else selected.values
                self.assertEqual(reader.call_count, 1)
                self.assertEqual(values[0,0], 6.)
                self.assertIsInstance(da.data, dsa.Array)

    def test_exact_labels_and_explicit_bounded_nearest(self):
        series = self.case.movie('bx', byteorder='little')
        with patch('kglobal_analysis.movie.np.fromfile', side_effect=AssertionError('read')):
            da = series.to_xarray()
            self.assertEqual(da.sel(time=series.times[2]).global_frame_index.item(), 2)
            self.assertNotEqual(series.times[2], 1.10)
            with self.assertRaises(KeyError):
                da.sel(time=1.10)
            tolerance = min(1e-6, min(np.diff(series.times))/1000)
            bounded = da.sel(time=1.10, method='nearest', tolerance=tolerance)
            self.assertEqual(bounded.time.item(), series.times[2])
            with self.assertRaises(KeyError):
                da.sel(time=1.075, method='nearest', tolerance=tolerance)
            self.assertEqual(series.locate_time(1.10), ('002',0))
            with self.assertRaises(KeyError):
                series.locate_time(1.075)

    def test_provenance_is_per_time_and_no_spatial_coordinates(self):
        da = self.lazy('jihpar')
        self.assertEqual(da.name, 'jihpar')
        self.assertEqual(set(da.coords), {'time','source_segment','local_frame_index','global_frame_index'})
        for coord in da.coords.values():
            self.assertEqual(coord.dims, ('time',))
            self.assertNotIsInstance(coord.data, dsa.Array)
        np.testing.assert_array_equal(da.source_segment.values, ['090','090','002','002'])
        np.testing.assert_array_equal(da.local_frame_index.values, [0,1,0,1])
        self.assertEqual(set(da.attrs), {'storage_name','movie_header','encoding','storage_order'})
        self.assertEqual(da.attrs['encoding'], 'double_byte')
        self.assertEqual(da.dtype, np.dtype('float64'))

    def test_synthetic_3d_lazy_order_and_one_read(self):
        (self.root/'param').write_text(PARAM.replace('#define nz 1','#define nz 2'))
        self.write_movies(samples=12)
        case = KGlobalCase(self.root)
        with patch('kglobal_analysis.movie.np.fromfile', side_effect=AssertionError('read')):
            series = case.movie('jihpar', byteorder='little')
            da = series.to_xarray()
            selected = da.isel(time=3)
            self.assertEqual(da.dims, ('time','x','y','z'))
            self.assertEqual(da.shape, (4,3,2,2))
            self.assertEqual(da.chunks, ((1,1,1,1),(3,),(2,),(2,)))
            self.assertTrue(set(('x','y','z')).isdisjoint(da.coords))
        with patch('kglobal_analysis.movie.np.fromfile', wraps=np.fromfile) as reader:
            result = selected.compute(scheduler='synchronous')
        self.assertEqual(reader.call_count, 1)
        self.assertEqual(reader.call_args.kwargs['count'], 12)
        for x,y,z in ((0,0,0),(1,0,0),(0,1,0),(0,0,1),(2,1,1)):
            self.assertEqual(result.isel(x=x,y=y,z=z).item(), 1100+12+x+3*(y+2*z))

    def test_bx_compatibility_uses_existing_reader(self):
        series = self.case.bx(byteorder='little')
        with patch.object(self.case, 'read_bx_frame', wraps=self.case.read_bx_frame) as reader:
            da = series.to_xarray()
            reader.assert_not_called()
            result = da.isel(time=2).compute(scheduler='synchronous')
        reader.assert_called_once_with('002', 0, byteorder='little')
        self.assertEqual(result.data[0,0], 100.)

    def test_spatial_selection_still_reads_whole_frame(self):
        da = self.lazy()
        with patch('kglobal_analysis.movie.np.fromfile', wraps=np.fromfile) as reader:
            value = da.isel(time=3, x=1, y=1).compute(scheduler='synchronous')
        self.assertEqual(reader.call_count, 1)
        self.assertEqual(reader.call_args.kwargs['count'], 6)
        self.assertEqual(value.item(), 110.)
