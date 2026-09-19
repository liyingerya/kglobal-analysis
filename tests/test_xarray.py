"""Conservative labels for one frame, using real decoders on small fixtures."""
from pathlib import Path
from tempfile import TemporaryDirectory
import struct
import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr

from kglobal_analysis import KGlobalCase, STANDARD_MOVIE_FORMAT, TimeMetadataError

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


class FrameXarrayTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'param').write_text(PARAM)
        # Physical order deliberately disagrees with suffix spelling.
        self.add_segment('090', [1.000000001, 1.050000001], 0)
        self.add_segment('002', [1.100000001, 1.150000001], 100)
        self.case = KGlobalCase(self.root)

    def add_segment(self, suffix, times, base, samples=6):
        for spec in STANDARD_MOVIE_FORMAT.variables:
            codes = range(base + spec.log_index * 100, base + spec.log_index * 100 + samples*2)
            (self.root / f'movie.{spec.storage_name}.{suffix}').write_bytes(
                struct.pack(f'<{samples*2}h', *codes))
        (self.root / f'movie.log.{suffix}').write_text('-32768 32767\n' * 36)
        (self.root / f'p3d.stdout.{suffix}').write_text('two-byte movie output\n' + ''.join(
            f'movie output, t= {time}\n' for time in times))

    def test_all_variables_are_named_from_storage(self):
        for spec in STANDARD_MOVIE_FORMAT.variables:
            with self.subTest(variable=spec.storage_name):
                series = self.case.movie(spec.storage_name, byteorder='little')
                da = series.read_frame_xarray(0)
                self.assertIsInstance(da, xr.DataArray)
                self.assertEqual(da.name, spec.storage_name)
                self.assertEqual(da.attrs['storage_name'], spec.storage_name)
                self.assertEqual(da.dims, ('x', 'y'))
                self.assertEqual(da.shape, (3, 2))
                self.assertEqual(da.isel(x=1, y=0).item(), spec.log_index*100+1)

    def test_global_index_uses_physical_order_and_exactly_one_reader(self):
        series = self.case.movie('jihpar', byteorder='little')
        values = np.arange(6., dtype=np.float64).reshape((3, 2), order='F')
        with patch.object(self.case, 'read_movie_frame', return_value=values) as reader:
            da = series.read_frame_xarray(2)
        reader.assert_called_once_with('jihpar', '002', 0, byteorder='little')
        self.assertIs(da.data, values)  # no copy/concatenation by the wrapper
        self.assertEqual(da.attrs['source_segment'], '002')
        self.assertEqual(da.attrs['local_frame_index'], 0)
        self.assertEqual(da.attrs['global_frame_index'], 2)
        self.assertEqual(da.coords['time'].item(), series.times[2])
        self.assertEqual(da.coords['time'].dims, ())
        self.assertEqual(da.attrs['movie_header'], 'movie_kglobal3.0.h')
        self.assertEqual(da.attrs['encoding'], 'double_byte')
        self.assertIn('x fastest', da.attrs['storage_order'])

    def test_time_lookup_uses_stored_time_and_boundary_mapping(self):
        series = self.case.movie('bx', byteorder='little')
        for time, global_index, suffix, local_index in ((1.05, 1, '090', 1), (1.10, 2, '002', 0)):
            with patch.object(self.case, 'read_movie_frame', wraps=self.case.read_movie_frame) as reader:
                da = series.read_time_xarray(time)
            reader.assert_called_once_with('bx', suffix, local_index, byteorder='little')
            self.assertEqual(da.coords['time'].item(), series.times[global_index])
            self.assertNotEqual(da.coords['time'].item(), time)  # request is not the timestamp
            self.assertEqual(da.attrs['global_frame_index'], global_index)

    def test_no_spatial_coordinates_or_unverified_physics_metadata(self):
        da = self.case.movie('bx', byteorder='little').read_frame_xarray(0)
        self.assertEqual(set(da.coords), {'time'})
        self.assertEqual(len(da.indexes), 0)
        self.assertNotIn('time', da.dims)
        self.assertEqual(set(da.attrs), {'storage_name', 'movie_header', 'encoding',
                                       'source_segment', 'local_frame_index',
                                       'global_frame_index', 'storage_order'})
        self.assertEqual(da.coords['time'].attrs, {})
        self.assertIsInstance(da.data, np.ndarray)
        self.assertIsNone(da.chunks)
        self.assertEqual(da.isel(x=2, y=1).item(), 405.)
        self.assertEqual((da + 2).isel(x=2, y=1).item(), 407.)

    def test_invalid_global_indices_fail_before_binary_read(self):
        series = self.case.movie('bx', byteorder='little')
        with patch.object(self.case, 'read_movie_frame', side_effect=AssertionError('read')):
            for index in (-1, 4, 100):
                with self.subTest(index=index), self.assertRaises(IndexError):
                    series.read_frame_xarray(index)
            for index in (True, np.bool_(False), '0', 1., None):
                with self.subTest(index=index), self.assertRaises(TypeError):
                    series.read_frame_xarray(index)
        self.assertEqual(series.read_frame_xarray(np.int64(3)).attrs['global_frame_index'], 3)

    def test_missing_times_have_no_nearest_fallback(self):
        series = self.case.movie('bx', byteorder='little')
        with patch.object(self.case, 'read_movie_frame', side_effect=AssertionError('read')):
            for time in (0., 1.075, 1.10001, 2.):
                with self.assertRaises(KeyError):
                    series.read_time_xarray(time)
            for time in (float('nan'), float('inf')):
                with self.assertRaises(ValueError):
                    series.read_time_xarray(time)

    def test_local_segment_wrapping_and_unchanged_numpy_apis(self):
        segment = self.case.movie_segment('jihpar', '002', byteorder='little')
        da = segment.read_time_xarray(1.15)
        self.assertEqual(da.attrs['source_segment'], '002')
        self.assertEqual(da.attrs['local_frame_index'], 1)
        self.assertNotIn('global_frame_index', da.attrs)
        self.assertEqual(da.coords['time'].item(), segment.times[1])
        direct = segment.read_frame(1)
        self.assertIsInstance(direct, np.ndarray)
        np.testing.assert_array_equal(da.values, direct)
        series = self.case.movie('jihpar', byteorder='little')
        self.assertIsInstance(series.read_time(1.15), np.ndarray)
        with patch.object(self.case, 'read_movie_frame', side_effect=AssertionError('read')):
            with self.assertRaises(IndexError):
                segment.read_frame_xarray(2)

    def test_missing_stdout_blocks_xarray_but_not_numpy(self):
        (self.root / 'p3d.stdout.090').unlink()
        case = KGlobalCase(self.root)
        segment = case.movie_segment('bx', '090', byteorder='little')
        with patch.object(case, 'read_movie_frame', side_effect=AssertionError('read')):
            with self.assertRaises(TimeMetadataError):
                segment.read_frame_xarray(0)
        self.assertEqual(segment.read_frame(0).shape, (3, 2))

    def test_bx_compatibility_inherits_generic_wrapper(self):
        series = self.case.bx(byteorder='little')
        with patch.object(self.case, 'read_bx_frame', wraps=self.case.read_bx_frame) as reader:
            da = series.read_frame_xarray(2)
        reader.assert_called_once_with('002', 0, byteorder='little')
        self.assertEqual(da.name, 'bx')
        self.assertEqual(da.dims, ('x', 'y'))
        self.assertIsInstance(series.read_time(1.1), np.ndarray)

    def test_synthetic_volume_voxels_and_nonzero_frame_offset(self):
        (self.root / 'param').write_text(PARAM.replace('#define nz 1', '#define nz 2'))
        self.add_segment('090', [1.000000001, 1.050000001], 0, samples=12)
        self.add_segment('002', [1.100000001, 1.150000001], 100, samples=12)
        case = KGlobalCase(self.root)
        series = case.movie('by', byteorder='little')
        original = np.fromfile
        def bounded(stream, *, dtype, count):
            self.assertEqual(Path(stream.name).name, 'movie.by.002')
            self.assertEqual(stream.tell(), 24)
            self.assertEqual(count, 12)
            return original(stream, dtype=dtype, count=count)
        with patch('kglobal_analysis.movie.np.fromfile', side_effect=bounded) as reader:
            da = series.read_frame_xarray(3)
        self.assertEqual(reader.call_count, 1)
        self.assertEqual(da.dims, ('x', 'y', 'z'))
        self.assertEqual(da.shape, (3, 2, 2))
        self.assertEqual(set(da.coords), {'time'})
        self.assertEqual(da.attrs['source_segment'], '002')
        self.assertEqual(da.attrs['local_frame_index'], 1)
        for x, y, z in ((0,0,0), (1,0,0), (0,1,0), (0,0,1), (2,1,1)):
            self.assertEqual(da.isel(x=x, y=y, z=z).item(), 100+500+12+x+3*(y+2*z))

    def test_singleton_x_and_y_axes_are_not_squeezed(self):
        param = PARAM.replace('#define nx 3', '#define nx 1').replace('#define ny 2', '#define ny 1')
        (self.root / 'param').write_text(param)
        self.add_segment('090', [1., 1.05], 0, samples=1)
        self.add_segment('002', [1.1, 1.15], 100, samples=1)
        da = KGlobalCase(self.root).movie('ni', byteorder='little').read_frame_xarray(0)
        self.assertEqual(da.dims, ('x', 'y'))
        self.assertEqual(da.shape, (1, 1))
