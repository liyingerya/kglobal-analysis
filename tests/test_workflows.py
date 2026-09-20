"""End-to-end orchestration, exact dependencies and shared event tasks."""
from collections import Counter
from dataclasses import replace
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import dask.array as da
from dask import delayed
import numpy as np
import xarray as xr

import kglobal_analysis as kga
from kglobal_analysis import derived, workflows as w
from kglobal_analysis.geometry import MovieGeometry, GeometryError
from test_geometry_flux import PARAM

NR = derived.ParticleMomentProfile(.04, False)
VALUES = dict(nih=2, neh=2, pihpar=6, pehpar=6, pihperp=6, pehperp=6,
              jihpar=0, jhpar=0, bx=2, by=1, bz=1, ni=90, pi=90, pc=90)
FIRE = set(VALUES)-{'ni', 'pi', 'pc'}


class SyntheticCase:
    def __init__(self, profile=NR):
        self.profile = profile
        self.requests = []
        self.reads = []
        self.geometry_calls = 0
        def load(name, event):
            self.reads.append((name, event))
            return np.full((8, 6), VALUES[name], dtype=float)
        self.ds = xr.Dataset({name: (('time', 'x', 'y'), da.stack([
            da.from_delayed(delayed(load)(name, i), shape=(8, 6), dtype=float)
            for i in range(3)])) for name in VALUES},
            coords={'time': [5.05, 5.1, 5.15], 'source_segment': ('time', ['005']*3),
                    'local_frame_index': ('time', [0, 1, 2])}, attrs={'selection': 'full'})

    def particle_moment_profile(self):
        return self.profile

    def movie_geometry(self):
        self.geometry_calls += 1
        return MovieGeometry(8, 6, 1, 4., 9., 1.)

    def movie_dataset(self, names, *, byteorder):
        self.requests.append((tuple(names), byteorder))
        return self.ds[list(names)]


class WorkflowTests(unittest.TestCase):
    def assert_reads(self, case, names, event=1):
        self.assertEqual(Counter(case.reads), Counter((n, event) for n in names))

    def test_catalog_nr_and_relativistic(self):
        for rel in (False, True):
            catalog = w.particle_quantity_catalog(replace(NR, relativistic=rel))
            self.assertEqual(len(catalog), 7)
            for item in catalog:
                self.assertEqual(item.population, 'particle')
                self.assertEqual(item.supported, not (rel and item.requires_nr))
                self.assertEqual(item.reason is not None, not item.supported)

    def test_all_quantities_delegate_and_dependencies(self):
        expected = {
            'number_density': ({'nih'}, 'particle_number_density'),
            'raw_parallel_stress': ({'pihpar'}, 'raw_parallel_stress'),
            'perpendicular_pressure': ({'pihperp'}, 'particle_perpendicular_pressure'),
            'perpendicular_temperature': ({'pihperp', 'nih'}, 'particle_perpendicular_temperature'),
            'parallel_pressure': ({'pihpar', 'jihpar', 'nih'}, 'particle_parallel_pressure'),
            'parallel_temperature': ({'pihpar', 'jihpar', 'nih'}, 'particle_parallel_temperature'),
            'total_temperature': ({'pihpar', 'jihpar', 'nih', 'pihperp'}, 'particle_total_temperature'),
        }
        for key, (names, operation) in expected.items():
            with self.subTest(key=key):
                case = SyntheticCase()
                fn = getattr(derived, operation)
                with patch.object(derived, operation, wraps=fn) as spy:
                    result = w.particle_quantity_series(case, key, species='ion', byteorder='big')
                    spy.assert_called_once()
                self.assertEqual(case.reads, [])
                self.assertEqual(case.geometry_calls, 0)
                self.assertEqual(set(case.requests[0][0]), names)
                self.assertEqual(case.requests[0][1], 'big')
                actual = result.isel(time=1).compute(scheduler='synchronous')
                self.assert_reads(case, names)
                self.assertEqual(actual.source_segment.item(), '005')
                self.assertEqual(actual.local_frame_index.item(), 1)
                self.assertEqual(actual.attrs['population'], 'particle')

    def test_invalid_key_and_species(self):
        for key in ('density', 'ion_temperature', 'nih', 'firehose_parameter', None):
            case = SyntheticCase()
            with self.assertRaises(w.UnknownQuantityError):
                w.particle_quantity_series(case, key, species='ion')
            self.assertEqual(case.requests, [])
        with self.assertRaises(derived.DerivedSemanticsError):
            w.particle_quantity_series(SyntheticCase(), 'number_density', species='unknown')

    def test_relativistic_rejected_before_dataset(self):
        for fn, args in ((w.particle_quantity_series, ('parallel_pressure',)),
                         (w.particle_quantity_series, ('parallel_temperature',)),
                         (w.particle_map_series, ('total_temperature',)),
                         (w.firehose_series, ()), (w.firehose_map_series, ())):
            case = SyntheticCase(replace(NR, relativistic=True))
            with self.assertRaises(derived.DerivedSemanticsError):
                fn(case, *args, **({'species': 'ion'} if args else {}))
            self.assertEqual(case.requests, [])
            self.assertEqual(case.reads, [])

    def test_relativistic_supported_quantities(self):
        for key in ('number_density', 'raw_parallel_stress', 'perpendicular_pressure', 'perpendicular_temperature'):
            case = SyntheticCase(replace(NR, relativistic=True))
            self.assertIsInstance(w.particle_quantity_series(case, key, species='electron').data, da.Array)
            self.assertEqual(case.reads, [])

    def test_particle_map_union_one_dataset(self):
        case = SyntheticCase()
        result = w.particle_map_series(case, 'total_temperature', species='ion')
        self.assertEqual(len(case.requests), 1)
        self.assertEqual(case.reads, [])
        names = {'pihpar', 'jihpar', 'nih', 'pihperp', 'bx', 'by'}
        self.assertEqual(set(case.requests[0][0]), names)
        frame = result.isel(time=1).compute(scheduler='synchronous')
        self.assert_reads(case, names)
        np.testing.assert_allclose(frame.field, 3)
        np.testing.assert_array_equal(frame.x, case.movie_geometry().x)
        self.assertEqual(frame.attrs['selection'], 'full')
        self.assertEqual(frame.field.attrs['scientific_quantity'], 'total_temperature')
        self.assertEqual(frame.psi.attrs['scientific_quantity'], 'magnetic_flux_function')

    def test_density_map_union(self):
        case = SyntheticCase()
        result = w.particle_map_series(case, 'number_density', species='ion')
        result.isel(time=1).compute(scheduler='synchronous')
        self.assert_reads(case, {'nih', 'bx', 'by'})

    def test_map_coordinate_attrs_may_differ(self):
        original = w.flux.magnetic_flux_2d
        def different_attrs(*args, **kwargs):
            psi = original(*args, **kwargs)
            return psi.assign_coords({axis: psi[axis].assign_attrs(
                units='code_normalized', long_name='derived coordinate') for axis in ('x', 'y')})
        case = SyntheticCase()
        with patch.object(w.flux, 'magnetic_flux_2d', side_effect=different_attrs):
            result = w.particle_map_series(case, 'number_density', species='ion')
        self.assertEqual(case.reads, [])
        self.assertEqual(result.x.attrs, {'units': 'code_normalized'})
        result.isel(time=1).compute(scheduler='synchronous')
        self.assert_reads(case, {'nih', 'bx', 'by'})

    def test_map_coordinate_values_must_match_exactly(self):
        original = w.flux.magnetic_flux_2d
        for axis in ('x', 'y'):
            def changed(*args, **kwargs):
                psi = original(*args, **kwargs)
                return psi.assign_coords({axis: psi[axis] + 1e-12})
            with self.subTest(axis=axis), patch.object(w.flux, 'magnetic_flux_2d', side_effect=changed):
                with self.assertRaises(derived.DerivedAlignmentError):
                    w.particle_map_series(SyntheticCase(), 'number_density', species='ion')

    def test_map_coordinate_dimensions_must_match(self):
        original = w.flux.magnetic_flux_2d
        for axis in ('x', 'y'):
            def changed(*args, **kwargs):
                psi = original(*args, **kwargs)
                values = np.broadcast_to(psi[axis].values[:, None] if axis == 'x'
                                         else psi[axis].values[None, :], (8, 6))
                return psi.drop_vars(axis).assign_coords({axis: (('x', 'y'), values)})
            with self.subTest(axis=axis), patch.object(w.flux, 'magnetic_flux_2d', side_effect=changed):
                with self.assertRaises(derived.DerivedAlignmentError):
                    w.particle_map_series(SyntheticCase(), 'number_density', species='ion')

    def test_maps_without_flux(self):
        for key in ('number_density', 'perpendicular_temperature', 'total_temperature'):
            case = SyntheticCase()
            ds = w.particle_map_series(case, key, species='ion', include_flux=False)
            self.assertEqual(set(ds), {'field'})
            self.assertEqual(case.geometry_calls, 1)
            self.assertNotIn('bx', case.requests[0][0])
        with self.assertRaises(w.WorkflowError):
            w.particle_map_series(SyntheticCase(), 'number_density', species='ion', include_flux='yes')

    def test_firehose_series(self):
        case = SyntheticCase()
        with patch.object(derived, 'firehose_parameter', wraps=derived.firehose_parameter) as spy:
            result = w.firehose_series(case)
            spy.assert_called_once()
        self.assertEqual(case.reads, [])
        self.assertEqual(len(case.requests), 1)
        result.isel(time=1).compute(scheduler='synchronous')
        self.assert_reads(case, FIRE)
        self.assertEqual(case.geometry_calls, 0)

    def test_firehose_map_shared_b_reads(self):
        case = SyntheticCase()
        result = w.firehose_map_series(case)
        self.assertEqual(len(case.requests), 1)
        self.assertEqual(len(case.requests[0][0]), 11)
        self.assertEqual(case.reads, [])
        result.isel(time=1).compute(scheduler='synchronous')
        self.assert_reads(case, FIRE)

    def test_flux_has_no_particle_profile_dependency(self):
        case = SyntheticCase()
        with patch.object(case, 'particle_moment_profile', side_effect=AssertionError('particle dependency')):
            result = w.magnetic_flux_series(case)
        self.assertEqual(case.reads, [])
        self.assertEqual(case.requests[0][0], ('bx', 'by'))
        result.isel(time=1).compute(scheduler='synchronous')
        self.assert_reads(case, {'bx', 'by'})

    def test_geometry_boundary(self):
        case = SyntheticCase()
        with patch.object(case, 'movie_geometry', side_effect=GeometryError('unsupported')):
            w.particle_quantity_series(case, 'number_density', species='ion')
            for fn, args in ((w.particle_map_series, ('number_density',)), (w.firehose_map_series, ()), (w.magnetic_flux_series, ())):
                with self.assertRaises(GeometryError):
                    fn(case, *args, **({'species': 'ion'} if args else {}))

    def test_mapping_independence(self):
        case = SyntheticCase()
        case.ds = case.ds.rename({'nih': 'mapped_density'})
        original = derived.ParticleMomentProfile.storage_for
        def mapping(profile, quantity, *, species, population='particle'):
            if (quantity, species, population) == ('number_density', 'ion', 'particle'):
                return 'mapped_density'
            return original(profile, quantity, species=species, population=population)
        with patch.object(derived.ParticleMomentProfile, 'storage_for', new=mapping):
            result = w.particle_quantity_series(case, 'number_density', species='ion')
        self.assertEqual(case.requests[0][0], ('mapped_density',))
        self.assertEqual(result.attrs['source_storage_names'], ('mapped_density',))

    def test_local_algebra_remains_rank_independent(self):
        case = SyntheticCase()
        case.ds = case.ds.expand_dims(z=[0, 1]).transpose('time', 'x', 'y', 'z')
        with patch.object(case, 'movie_geometry', side_effect=AssertionError('geometry requested')):
            result = w.particle_quantity_series(case, 'total_temperature', species='ion')
        self.assertEqual(result.dims, ('time', 'x', 'y', 'z'))
        np.testing.assert_allclose(result.isel(time=1).compute(scheduler='synchronous'), 3)
        self.assert_reads(case, {'nih', 'pihpar', 'jihpar', 'pihperp'})

    def test_map_calls_existing_science_on_same_dataset(self):
        case = SyntheticCase()
        with patch.object(derived, 'particle_number_density', wraps=derived.particle_number_density) as scalar, \
             patch.object(w.flux, 'magnetic_flux_2d', wraps=w.flux.magnetic_flux_2d) as psi:
            w.particle_map_series(case, 'number_density', species='ion')
        scalar.assert_called_once()
        psi.assert_called_once()
        self.assertIs(scalar.call_args.args[0], psi.call_args.args[0])
        self.assertEqual(case.reads, [])

    def test_firehose_map_without_flux(self):
        case = SyntheticCase()
        result = w.firehose_map_series(case, include_flux=False)
        self.assertEqual(set(result.data_vars), {'field'})
        self.assertEqual(set(case.requests[0][0]), FIRE)
        result.isel(time=1).compute(scheduler='synchronous')
        self.assert_reads(case, FIRE)

    def test_alignment_error_preserved(self):
        case = SyntheticCase()
        with patch.object(case, 'movie_dataset', side_effect=kga.MovieAlignmentError('coverage')):
            with self.assertRaises(kga.MovieAlignmentError):
                w.particle_map_series(case, 'number_density', species='ion')

    def test_root_exports(self):
        for name in ('particle_quantity_catalog', 'particle_quantity_series', 'particle_map_series',
                     'firehose_series', 'firehose_map_series', 'magnetic_flux_series'):
            self.assertIs(getattr(kga, name), getattr(w, name))
            self.assertIn(name, kga.__all__)

    def test_spacetime_and_exact_events(self):
        from kglobal_analysis.analysis import spacetime
        case = SyntheticCase()
        workflow = w.particle_map_series(case, 'number_density', species='ion')
        xt = spacetime(workflow.field, line_axis='x', fixed_indices={'y': 2}, times=[5.05, 5.15])
        np.testing.assert_array_equal(xt.time, [5.05, 5.15])
        self.assertEqual(Counter(case.reads), Counter([('nih', 0), ('nih', 2)]))
        case.reads.clear()
        for index in (0, 2):
            workflow.isel(time=index).compute(scheduler='synchronous')
        self.assertEqual(Counter(case.reads), Counter((n, i) for n in ('nih', 'bx', 'by') for i in (0, 2)))

    def test_iteration_culls_flux(self):
        from kglobal_analysis.animation import iter_movie_frames
        case = SyntheticCase()
        workflow = w.particle_map_series(case, 'number_density', species='ion')
        for frame in iter_movie_frames(workflow.field, times=[5.05, 5.15]):
            self.assertEqual(frame.data.dims, ('x', 'y'))
        self.assertEqual(Counter(case.reads), Counter([('nih', 0), ('nih', 2)]))

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'optional plotting unavailable')
    def test_plot_and_png_composition(self):
        from kglobal_analysis.plotting import plot_scalar_map
        from kglobal_analysis.animation import render_frame_sequence
        case = SyntheticCase()
        workflow = w.particle_map_series(case, 'number_density', species='ion')
        frame = workflow.isel(time=1).compute(scheduler='synchronous')
        plot = plot_scalar_map(frame.field, horizontal='x', vertical='y', contours=frame.psi)
        np.testing.assert_array_equal(plot.artist.get_array(), frame.field.values.T)
        plot.figure.clear()
        case.reads.clear()
        with TemporaryDirectory() as root:
            records = render_frame_sequence(workflow.field, root, horizontal='x', vertical='y',
                                            times=[5.05], vmin=1, vmax=3)
            self.assertEqual(len(records), 1)
        self.assert_reads(case, {'nih'}, event=0)

    def test_real_metadata_catalog(self):
        root = Path(__file__).resolve().parents[2]/'validation-data/hcs_large_005'
        if not root.exists():
            self.skipTest('private fixture unavailable')
        with patch('numpy.fromfile', side_effect=AssertionError('sample read')):
            case = kga.KGlobalCase(root)
            catalog = w.particle_quantity_catalog(case.particle_moment_profile())
            self.assertEqual(sum(item.supported for item in catalog), 4)
            with self.assertRaises(derived.DerivedSemanticsError):
                w.firehose_series(case)

    def test_decoder_shared_graph_and_missing_coverage(self):
        with TemporaryDirectory() as root:
            path = Path(root)
            (path/'param').write_text(PARAM+'#define m_e .04\n')
            for name, value in VALUES.items():
                (path/f'movie.{name}.005').write_bytes(np.full(48*3, value, dtype='<i2').tobytes())
            (path/'movie.log.005').write_text('-32768 32767\n'*18*3)
            (path/'p3d.stdout.005').write_text(''.join(f'movie output, t= {t}\n' for t in (5.05, 5.1, 5.15)))
            with patch('numpy.fromfile', side_effect=AssertionError('construction read')):
                result = w.firehose_map_series(kga.KGlobalCase(path))
            reads = []
            original = np.fromfile
            def read(stream, *, dtype, count):
                reads.append((Path(stream.name).name, stream.tell(), count))
                return original(stream, dtype=dtype, count=count)
            with patch('numpy.fromfile', side_effect=read):
                result.isel(time=1).compute(scheduler='synchronous')
            self.assertEqual(Counter(reads), Counter((f'movie.{n}.005', 96, 48) for n in FIRE))
            (path/'movie.by.005').unlink()
            with self.assertRaises(kga.MovieAlignmentError):
                w.particle_map_series(kga.KGlobalCase(path), 'number_density', species='ion')
