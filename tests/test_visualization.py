"""Scientific selection/artist contracts, not pixel-perfect styling tests."""
import gc
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import weakref

import numpy as np
import xarray as xr
import dask.array as da
from dask import delayed

from kglobal_analysis.analysis import index_cut, spacetime
from kglobal_analysis.animation import iter_movie_frames, render_frame_sequence
from kglobal_analysis.energy import EnergySpectrum
from kglobal_analysis.distribution import ReducedDistribution
from kglobal_analysis.plotting import (plot_scalar_map, plot_spacetime, plot_energy_spectrum,
                                      plot_energy_overlay, plot_distribution)

HAS_PLOT = importlib.util.find_spec('matplotlib') is not None
if HAS_PLOT:
    import matplotlib
    matplotlib.use('Agg')  # Test-only backend choice.
    from matplotlib import pyplot as plt


def movie(calls, refs=None):
    def load(i):
        if refs is not None:
            gc.collect()
            if any(ref() is not None for ref in refs):
                raise AssertionError('previous full frame retained')
        calls.append(i)
        x, y = np.indices((4, 3))
        values = (10*i + x*x + (i+1)*y*x + y).astype(float)
        if refs is not None:
            refs.append(weakref.ref(values))
        return values
    values = da.stack([da.from_delayed(delayed(load)(i), shape=(4, 3), dtype=float)
                       for i in range(4)])
    return xr.DataArray(values, dims=('time', 'x', 'y'), name='bx',
                        coords={'time': [5.05, 5.10, 5.15, 5.20],
                                'source_segment': ('time', ['005']*4)},
                        attrs={'storage_name': 'bx'})


def energy():
    edges = tuple(np.geomspace(.01, 10, 201))
    return EnergySpectrum('energy_spectrum', 'electron', 'xenergylog', '016',
                          Path('xenergylog.016'), Path('vd2dgyro.016'),
                          (999., 0., -1.) + tuple(range(1, 199)), .01, 10., edges)


def distribution(position=None, species='electron'):
    par = np.arange(-200, 201)
    if position is None:
        shape, names, indices = (401, 201), ('v_parallel', 'v_perp'), (par, np.arange(201))
    else:
        shape, names, indices = (101, 401), ('position', 'v_parallel'), (np.arange(101), par)
    scale = 2 if species == 'electron' else .2
    axes = tuple(a.astype(float)*scale for a in indices)
    x, y = np.indices(shape)
    values = (x*x + x*y + 3*y).astype(float)
    return ReducedDistribution('provisional_quantity', species, 'regular', 'legacy_name', position,
                               '016', Path('legacy_name.016'), Path('vd2dgyro.016'), 0, 'box',
                               {'ZMIN': .25, 'ZMAX': .75}, 'normalized_bin_mass', values,
                               names, indices, axes, {})


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.frame = xr.DataArray(np.arange(12).reshape(4, 3), dims=('x', 'y'),
                                  coords={'time': 5.05}, attrs={'storage_name': 'bx'})

    def test_index_cuts_both_axes_preserve_provenance(self):
        for axis, fixed in [('x', {'y': 0}), ('y', {'x': 2})]:
            cut = index_cut(self.frame, line_axis=axis, fixed_indices=fixed)
            np.testing.assert_array_equal(cut, self.frame.isel(fixed))
            self.assertEqual(cut.dims, (axis,))
            self.assertEqual(cut.time.item(), 5.05)
            self.assertEqual(cut.attrs['fixed_indices'], fixed)
            self.assertEqual(cut.attrs['storage_name'], 'bx')
        self.assertNotIn('fixed_indices', self.frame.attrs)

    def test_future_volume_line(self):
        volume = xr.DataArray(np.arange(24).reshape(4, 3, 2), dims=('x', 'y', 'z'))
        cut = index_cut(volume, line_axis='z', fixed_indices={'x': 1, 'y': 2})
        np.testing.assert_array_equal(cut, volume.values[1, 2, :])

    def test_cut_bad_selection(self):
        for axis, fixed in [('x', {}), ('z', {'x': 0}), ('x', {'y': -1}),
                            ('x', {'y': 3}), ('x', {'y': True}), ('x', {'y': 0, 'z': 1})]:
            with self.subTest(axis=axis, fixed=fixed), self.assertRaises((ValueError, TypeError, IndexError)):
                index_cut(self.frame, line_axis=axis, fixed_indices=fixed)
        with self.assertRaises(ValueError):
            index_cut(movie([]), line_axis='x', fixed_indices={'y': 0})

    def test_spacetime_exact_subset_order_and_bounded_frames(self):
        calls, refs = [], []
        source = movie(calls, refs)
        original = xr.DataArray.compute
        def guarded(obj, **kwargs):
            self.assertNotIn('time', obj.dims)
            return original(obj, **kwargs)
        with patch.object(xr.DataArray, 'compute', guarded):
            result = spacetime(source, line_axis='x', fixed_indices={'y': 1}, times=[5.2, 5.1])
        self.assertEqual(calls, [1, 3])
        self.assertEqual(result.dims, ('time', 'x'))
        np.testing.assert_allclose(result.time, [5.1, 5.2])
        for row, i in enumerate([1, 3]):
            x = np.arange(4)
            np.testing.assert_array_equal(result[row], 10*i+x*x+(i+1)*x+1)
        self.assertEqual(result.attrs['fixed_indices'], {'y': 1})
        self.assertLessEqual(result.nbytes, 8*8)
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))

    def test_no_nearest_or_invented_times(self):
        source = movie([])
        for times in ([5.12], [np.nan], [5.1, 5.1], []):
            with self.assertRaises((KeyError, ValueError)):
                spacetime(source, line_axis='x', fixed_indices={'y': 0}, times=times)
        for bad in (source.drop_vars('time'), source.assign_coords(time=[1, 1, 2, 3]),
                    source.assign_coords(time=[1, 2, np.nan, 4])):
            with self.assertRaises(ValueError):
                list(iter_movie_frames(bad))

    def test_tolerance_and_stride(self):
        calls = []
        frames = iter_movie_frames(movie(calls), times=[5.05000001, 5.10, 5.20], stride=2)
        first = next(frames)
        self.assertEqual(first.time, 5.05)
        del first
        second = next(frames)
        self.assertEqual(second.time, 5.2)
        del second
        with self.assertRaises(StopIteration):
            next(frames)
        self.assertEqual(calls, [0, 3])

    def test_iterator_does_not_retain_prior_frame(self):
        calls, refs = [], []
        iterator = iter_movie_frames(movie(calls, refs), stride=2)
        first = next(iterator)
        self.assertEqual(first.source_index, 0)
        del first
        second = next(iterator)
        self.assertEqual(second.source_index, 2)
        del second
        with self.assertRaises(StopIteration):
            next(iterator)
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))

    def test_aligned_dataset_event_compute_shared_tasks(self):
        calls = []
        base = movie(calls)
        ds = xr.Dataset({'base': base, 'overlay': base*2})
        original = xr.Dataset.compute
        def guarded(obj, **kwargs):
            self.assertNotIn('time', obj.dims)
            return original(obj, **kwargs)
        with patch.object(xr.Dataset, 'compute', guarded):
            iterator = iter_movie_frames(ds, variable='base', contour_variable='overlay', times=[5.1])
            result = next(iterator)
            np.testing.assert_array_equal(result.contours, result.data*2)
            self.assertEqual(calls, [1])

    def test_time_chunks_and_dataset_selection_validation(self):
        source = movie([])
        for bad in (source.chunk({'time': 2}), xr.Dataset({'bx': source})):
            with self.assertRaises(ValueError):
                list(iter_movie_frames(bad))
        for stride in (0, -1, True):
            with self.assertRaises((ValueError, TypeError)):
                list(iter_movie_frames(source, stride=stride))


@unittest.skipUnless(HAS_PLOT, 'Install kglobal-analysis[plot] for visualization tests')
class PlottingTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(plt.close, 'all')
        self.data = xr.DataArray([[1., 2., 3.], [10., 20., 30.]], dims=('x', 'y'), name='bx')

    def test_nonsquare_orientation_and_indices(self):
        before = self.data.copy(deep=True)
        result = plot_scalar_map(self.data, horizontal='x', vertical='y')
        array = result.artist.get_array().reshape(3, 2)
        np.testing.assert_array_equal(array, self.data.values.T)
        self.assertEqual(array[0, 0], 1)
        self.assertEqual(array[-1, -1], 30)
        self.assertEqual(array[0, -1], 10)
        self.assertEqual(result.axes.get_xlabel(), 'x index')
        self.assertEqual(result.axes.get_ylabel(), 'y index')
        self.assertEqual(result.colorbar.ax.get_ylabel(), 'bx')
        xr.testing.assert_identical(self.data, before)

    def test_explicit_axes_and_coordinates(self):
        figure, ax = plt.subplots()
        result = plot_scalar_map(self.data, horizontal='y', vertical='x', ax=ax,
                                 coordinates={'x': [3, 7], 'y': [10, 20, 40]},
                                 vmin=0, vmax=100, title='chosen', xlabel='y supplied')
        self.assertIs(result.axes, ax)
        self.assertEqual(result.axes.get_title(), 'chosen')
        self.assertEqual(result.artist.norm.vmin, 0)
        self.assertEqual(result.artist.norm.vmax, 100)
        np.testing.assert_array_equal(result.artist.get_array().reshape(2, 3), self.data)

    def test_bad_coordinates_and_volumes(self):
        for coordinates in ({'x': [0]}, {'x': [0, np.nan]}, {'x': [1, 1]},
                            {'x': [[1, 2]]}, {'z': [1]}, {'y': [0, 2, 1]}):
            with self.subTest(coords=coordinates), self.assertRaises(ValueError):
                plot_scalar_map(self.data, horizontal='x', vertical='y', coordinates=coordinates)
        with self.assertRaises(ValueError):
            plot_scalar_map(self.data.expand_dims(z=[0]), horizontal='x', vertical='y')
        with self.assertRaises(ValueError):
            plot_scalar_map(self.data.chunk(), horizontal='x', vertical='y')

    def test_existing_coordinate_units_and_descending(self):
        data = self.data.assign_coords(x=[5., 2.], y=[1., 2., 3.])
        data.x.attrs['units'] = 'user unit'
        result = plot_scalar_map(data, horizontal='x', vertical='y')
        self.assertEqual(result.axes.get_xlabel(), 'x [user unit]')

    def test_log_masks_and_signed_normalization(self):
        data = self.data.copy(deep=True)
        data.values[0] = [0, -1, np.nan]
        original = data.copy(deep=True)
        result = plot_scalar_map(data, horizontal='x', vertical='y', normalization='log', vmin=1, vmax=40)
        mask = np.ma.getmaskarray(result.artist.get_array()).reshape(3, 2)
        self.assertTrue(mask[:, 0].all())
        self.assertFalse(mask[:, 1].any())
        xr.testing.assert_identical(data, original)
        signed = plot_scalar_map(data, horizontal='x', vertical='y', normalization='signed')
        self.assertEqual(signed.artist.norm.vcenter, 0)
        self.assertEqual(signed.artist.norm.vmin, -signed.artist.norm.vmax)

    def test_empty_log_constant_and_bad_limits(self):
        with self.assertRaisesRegex(ValueError, 'positive'):
            plot_scalar_map(self.data*0, horizontal='x', vertical='y', normalization='log')
        for policy in ('linear', 'log', 'signed'):
            result = plot_scalar_map(self.data*0+2, horizontal='x', vertical='y', normalization=policy)
            self.assertLess(result.artist.norm.vmin, result.artist.norm.vmax)
        for kwargs in ({'vmin': 5, 'vmax': 1}, {'vmin': np.nan},
                       {'normalization': 'log', 'vmin': 0}, {'normalization': 'bad'}):
            with self.assertRaises(ValueError):
                plot_scalar_map(self.data, horizontal='x', vertical='y', **kwargs)

    def test_contour_alignment_and_no_mutation(self):
        data = self.data.assign_coords(x=[0., 1.], y=[0., 1., 2.], time=5.)
        other = data*2
        before = other.copy(deep=True)
        result = plot_scalar_map(data, horizontal='x', vertical='y', contours=other, contour_levels=[5, 20])
        self.assertIsNotNone(result.contours)
        self.assertEqual(result.metadata['overlay'], 'supplied scalar contours')
        xr.testing.assert_identical(other, before)
        for bad in (other.isel(y=slice(0, 2)), other.transpose(),
                    other.assign_coords(x=[1., 2.]), other.assign_coords(time=6.),
                    other.drop_vars('x')):
            with self.assertRaises(ValueError):
                plot_scalar_map(data, horizontal='x', vertical='y', contours=bad)

    def test_auxiliary_contour_coordinate_mismatch(self):
        base = self.data.assign_coords(position=('x', [1., 2.]))
        other = base.assign_coords(position=('x', [2., 3.]))
        with self.assertRaisesRegex(ValueError, 'coordinates'):
            plot_scalar_map(base, horizontal='x', vertical='y', contours=other)

    def test_spacetime_renderer_orientation(self):
        data = spacetime(movie([]), line_axis='x', fixed_indices={'y': 1}, times=[5.1, 5.2])
        result = plot_spacetime(data, line_axis='x')
        np.testing.assert_array_equal(result.artist.get_array().reshape(2, 4), data)
        self.assertEqual(result.axes.get_aspect(), 'auto')
        self.assertEqual(result.axes.get_ylabel(), 'time')

    def test_energy_semantics_and_overlays(self):
        spectrum = energy()
        original = spectrum.storage_values
        result = plot_energy_spectrum(spectrum, xscale='log', yscale='log', label='user label')
        self.assertEqual(len(result.artist.get_xdata()), 200)
        np.testing.assert_array_equal(result.artist.get_xdata().data, spectrum.centers)
        self.assertTrue(np.ma.getmaskarray(result.artist.get_ydata())[:2].all())
        self.assertEqual(result.axes.get_ylabel(), 'legacy energy estimator')
        self.assertEqual(spectrum.storage_values, original)
        results = plot_energy_overlay({'one': spectrum, 'two': spectrum}, yscale='linear')
        self.assertEqual([r.artist.get_label() for r in results], ['one', 'two'])
        self.assertIs(results[0].axes, results[1].axes)
        np.testing.assert_array_equal(results[0].artist.get_ydata(), spectrum.values)

    def test_distribution_all_families_species_and_modes(self):
        for position in (None, 'x', 'y'):
            for species in ('electron', 'ion'):
                dist = distribution(position, species)
                for mode in ('index', 'centers'):
                    result = plot_distribution(dist, coordinate_mode=mode, normalization='log')
                    drawn = result.artist.get_array().reshape(dist.shape[::-1])
                    np.testing.assert_array_equal(drawn.data, dist.values.T)
                    self.assertEqual(result.colorbar.ax.get_ylabel(), 'normalized bin mass')
                    self.assertEqual(result.metadata['region_metadata']['ZMIN'], .25)
                    self.assertEqual(result.metadata['checkpoint_suffix'], '016')
                    self.assertEqual(result.metadata['species'], species)
                    self.assertEqual(dist.values[0, 0], 0)
                    self.assertEqual(drawn.size, dist.values.size)
                    if mode == 'centers':
                        # QuadMesh centers recover supplied species-specific axes.
                        edges = result.artist.get_coordinates()[0, :, 0]
                        np.testing.assert_allclose((edges[1:]+edges[:-1])/2, dist.axis_values[0])
                    plt.close(result.figure)

    def test_detached_coordinate_example_arithmetic(self):
        dist = energy()
        x = np.array(dist.centers, copy=True)
        transformed = x / 2**3
        self.assertFalse(np.shares_memory(x, transformed))
        np.testing.assert_array_equal(dist.centers, x)
        self.assertEqual(dist.storage_values[0], 999)


@unittest.skipUnless(HAS_PLOT, 'Install kglobal-analysis[plot] for visualization tests')
class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'frames'
        self.options = dict(horizontal='x', vertical='y', dpi=30, figsize=(2, 2))

    def test_fixed_order_times_png_no_frame_retention(self):
        calls, refs = [], []
        before = plt.get_fignums()
        records = render_frame_sequence(movie(calls, refs), self.path, times=[5.2, 5.05],
                                         vmin=0, vmax=80, **self.options)
        self.assertEqual(calls, [0, 3])
        self.assertEqual([r.path.name for r in records], ['frame_000000.png', 'frame_000001.png'])
        self.assertEqual([r.time for r in records], [5.05, 5.2])
        for record in records:
            self.assertEqual((record.vmin, record.vmax), (0, 80))
            self.assertTrue(record.path.read_bytes().startswith(b'\x89PNG'))
            self.assertGreater(record.path.stat().st_size, 100)
        self.assertEqual(plt.get_fignums(), before)
        self.assertTrue(all(ref() is None for ref in refs))

    def test_global_two_pass_order_and_single_frame_compute(self):
        calls, refs = [], []
        source = movie(calls, refs)
        original = xr.DataArray.compute
        def guard(obj, **kwargs):
            self.assertNotIn('time', obj.dims)
            return original(obj, **kwargs)
        with patch.object(xr.DataArray, 'compute', guard):
            records = render_frame_sequence(source, self.path, stride=2, color_policy='global', **self.options)
        self.assertEqual(calls, [0, 2, 0, 2])
        self.assertEqual(len({(r.vmin, r.vmax) for r in records}), 1)
        self.assertEqual(records[0].vmin, 0)
        self.assertEqual(records[0].vmax, 49)

    def test_explicit_per_frame_limits(self):
        records = render_frame_sequence(movie([]), self.path, stride=2, color_policy='per_frame', **self.options)
        self.assertNotEqual(records[0].vmin, records[1].vmin)

    def test_overwrite_and_preflight(self):
        calls = []
        source = movie(calls)
        with self.assertRaises(ValueError):
            render_frame_sequence(source, self.path, **self.options)
        self.assertFalse(self.path.exists())
        self.path.mkdir()
        existing = self.path/'frame_000000.png'
        existing.write_bytes(b'original')
        with self.assertRaises(FileExistsError):
            render_frame_sequence(source, self.path, vmin=0, vmax=80, **self.options)
        self.assertEqual(calls, [])
        self.assertEqual(existing.read_bytes(), b'original')
        render_frame_sequence(source, self.path, times=[5.05], vmin=0, vmax=80,
                              overwrite=True, **self.options)
        self.assertTrue(existing.read_bytes().startswith(b'\x89PNG'))

    def test_global_log_empty_constant_and_masked(self):
        zero = xr.DataArray(np.zeros((2, 2, 3)), dims=('time', 'x', 'y'), coords={'time': [4., 5.]})
        with self.assertRaisesRegex(ValueError, 'positive'):
            render_frame_sequence(zero, self.path, color_policy='global', normalization='log', **self.options)
        self.assertFalse(self.path.exists())
        records = render_frame_sequence(zero+3, self.path, color_policy='global', normalization='log', **self.options)
        self.assertEqual([(r.vmin, r.vmax) for r in records], [(1.5, 6.), (1.5, 6.)])
        zero.values[:] = 2
        zero.values[:, 0, 0] = np.nan
        render_frame_sequence(zero, self.path, color_policy='global', overwrite=True, **self.options)

    def test_dataset_contours_export(self):
        calls = []
        base = movie(calls)
        ds = xr.Dataset({'base': base, 'contour': base*2})
        render_frame_sequence(ds, self.path, variable='base', contour_variable='contour',
                              times=[5.10], vmin=0, vmax=80, **self.options)
        self.assertEqual(calls, [1])


class OptionalDependencyTests(unittest.TestCase):
    def test_core_and_modules_do_not_import_matplotlib(self):
        script = '''
import sys
import kglobal_analysis
import kglobal_analysis.analysis
import kglobal_analysis.plotting
import kglobal_analysis.animation
assert not any(n == 'matplotlib' or n.startswith('matplotlib.') for n in sys.modules)
'''
        subprocess.run([sys.executable, '-c', script], check=True, env=os.environ.copy())

    def test_missing_plot_dependency_actionable(self):
        import kglobal_analysis.plotting as plotting
        with patch.object(plotting.importlib, 'import_module', side_effect=ImportError('missing')):
            with self.assertRaisesRegex(ImportError, r'kglobal-analysis\[plot\]'):
                plotting._matplotlib()


if __name__ == '__main__':
    unittest.main()
