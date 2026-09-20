"""Source-derived moment contracts; synthetic physics, no publication claims."""
from collections import Counter
from dataclasses import FrozenInstanceError, replace
import importlib.util
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr
import dask.array as da
from dask import delayed

from kglobal_analysis import KGlobalCase, parse_parameters
from kglobal_analysis.derived import (
    ParticleMomentProfile, DerivedQuantityError, DerivedSemanticsError,
    DerivedAlignmentError, particle_number_density, raw_parallel_stress,
    particle_perpendicular_pressure, particle_perpendicular_temperature,
    particle_parallel_pressure, particle_parallel_temperature,
    particle_total_temperature, firehose_parameter, validity_mask, _center_parallel,
)

PARAM = '''#define movie_header "movie_kglobal3.0.h"
#define double_byte
#define m_e .04
'''
NR = ParticleMomentProfile(.04, False)
NR_OPS = (particle_parallel_pressure, particle_parallel_temperature, particle_total_temperature)
ALL_SPECIES_OPS = (particle_number_density, raw_parallel_stress, particle_perpendicular_pressure,
                   particle_perpendicular_temperature) + NR_OPS


def moments(shape=(3, 2), dims=('x', 'y'), **changes):
    values = dict(nih=2., neh=2., pihpar=6., pehpar=6., pihperp=6., pehperp=6.,
                  jihpar=0., jhpar=0., bx=2., by=0., bz=0., ni=99., pi=91., pc=92.)
    values.update(changes)
    return xr.Dataset({key: (dims, np.broadcast_to(value, shape).copy()) for key, value in values.items()})


class ProfileTests(unittest.TestCase):
    def test_parsed_profiles_and_case_entry(self):
        for mass in ('.04', '.01', '( 4D-2 )'):
            with TemporaryDirectory() as root:
                path = Path(root)/'param'
                path.write_text(PARAM.replace('.04', mass))
                profile = KGlobalCase(root).particle_moment_profile()
                self.assertFalse(profile.relativistic)
                self.assertEqual(profile.parameter_source, str(path.resolve()))
                self.assertGreater(profile.electron_mass_ratio, 0)
                self.assertEqual(profile.mass_charge('ion'), (1., 1))
                self.assertEqual(profile.mass_charge('electron'), (profile.electron_mass_ratio, -1))

    def test_flags_and_ifdef_zero(self):
        for flags, expected in [('', 0), ('#define doublesmooth\n', 0),
                                ('#define smooth_part_mom\n', 1),
                                ('#define smooth_part_mom\n#define doublesmooth\n', 2)]:
            p = ParticleMomentProfile.from_parameters(parse_parameters(PARAM + flags))
            self.assertEqual(p.smoothing_passes, expected)
        p = ParticleMomentProfile.from_parameters(parse_parameters(PARAM + '#define relativistic 0\n'))
        self.assertTrue(p.relativistic)

    def test_mass_missing_invalid_expression(self):
        for mass in ('0', '-1', 'nan', 'inf', '1e999', '1/25', 'MASS', '(.04)+0'):
            with self.subTest(mass=mass), self.assertRaises(DerivedSemanticsError):
                ParticleMomentProfile.from_parameters(parse_parameters(PARAM.replace('.04', mass)))
        with self.assertRaises(DerivedSemanticsError):
            ParticleMomentProfile.from_parameters(parse_parameters(PARAM.replace('#define m_e .04', '')))

    def test_unsupported_and_conflicting_profile(self):
        for extra in ('#define mult_species\n', '#define four_byte\n', '#define heatfluxmovies\n',
                      '#define movie_header "unknown.h"\n'):
            with self.assertRaises(DerivedSemanticsError):
                ParticleMomentProfile.from_parameters(parse_parameters(PARAM + extra))
        p = parse_parameters(PARAM)
        for modified in (replace(p, movie_header='unknown'), replace(p, double_byte=False),
                         replace(p, definitions={'m_e': '.04'})):
            with self.assertRaises(DerivedSemanticsError):
                ParticleMomentProfile.from_parameters(modified)
        with self.assertRaises(DerivedSemanticsError):
            replace(NR, producer_id='unknown')
        with self.assertRaises(ValueError):
            parse_parameters(PARAM + '#ifdef MAYBE\n#define relativistic\n#endif\n')

    def test_explicit_profile_validation_frozen(self):
        for mass in (True, None, '.04', 0, -1, np.nan, np.inf, 1j):
            with self.assertRaises(DerivedSemanticsError):
                ParticleMomentProfile(mass, False)
        for value in (None, 0, 'False'):
            with self.assertRaises(DerivedSemanticsError):
                ParticleMomentProfile(.04, value)
        with self.assertRaises(FrozenInstanceError):
            NR.relativistic = True
        with self.assertRaises(DerivedSemanticsError):
            NR.mass_charge('helium')

    def test_population_mapping(self):
        self.assertEqual(NR.storage_for('number_density', species='ion'), 'nih')
        self.assertEqual(NR.storage_for('number_density', species='electron'), 'neh')
        self.assertEqual(NR.storage_for('number_density', species='ion', population='fluid'), 'ni')
        for population in ('total', 'unknown'):
            with self.assertRaises(DerivedSemanticsError):
                NR.storage_for('number_density', species='ion', population=population)

    def test_testmovie_is_provenance_not_particle_override(self):
        profile = ParticleMomentProfile.from_parameters(parse_parameters(PARAM + '#define testmovie\n'))
        result = firehose_parameter(moments(pc=-123), profile=profile)
        np.testing.assert_array_equal(result, 1)
        self.assertTrue(result.attrs['testmovie'])
        self.assertNotIn('pc', result.attrs['source_storage_names'])

    def test_real_metadata_only_relativistic_gate(self):
        root = Path(__file__).resolve().parents[2]/'validation-data/hcs_large_005'
        if not root.is_dir():
            self.skipTest('private metadata fixture unavailable')
        with patch('numpy.fromfile', side_effect=AssertionError('binary read')):
            profile = KGlobalCase(root).particle_moment_profile()
            self.assertTrue(profile.relativistic)
            self.assertEqual(profile.electron_mass_ratio, .04)
            with self.assertRaises(DerivedSemanticsError):
                firehose_parameter(moments(), profile=profile)


class ScienceTests(unittest.TestCase):
    def test_isotropic_and_population_labels(self):
        ds = moments()
        for species in ('electron', 'ion'):
            n = particle_number_density(ds, species=species, profile=NR)
            np.testing.assert_array_equal(n, 2)
            self.assertEqual(n.attrs['population'], 'particle')
            self.assertEqual(n.attrs['scientific_quantity'], 'number_density')
            self.assertNotIn('total', n.name)
            for fn in (particle_parallel_temperature, particle_perpendicular_temperature, particle_total_temperature):
                np.testing.assert_array_equal(fn(ds, species=species, profile=NR), 3)
        np.testing.assert_array_equal(firehose_parameter(ds, profile=NR), 1)

    def test_anisotropic_single_contribution(self):
        ds = moments(pihpar=8., pihperp=2., bx=2., by=2., bz=2.)
        np.testing.assert_array_equal(particle_total_temperature(ds, species='ion', profile=NR), 2)
        np.testing.assert_array_equal(firehose_parameter(ds, profile=NR), .5)

    def test_drift_invariance(self):
        ds = moments(pihpar=56., jihpar=10.)
        np.testing.assert_array_equal(raw_parallel_stress(ds, species='ion', profile=NR), 56)
        np.testing.assert_array_equal(particle_parallel_pressure(ds, species='ion', profile=NR), 6)
        np.testing.assert_array_equal(particle_parallel_temperature(ds, species='ion', profile=NR), 3)

    def test_electron_mass_is_not_fixed(self):
        for mass in (.04, .01):
            ds = moments(pehpar=6 + mass*2*25, jhpar=-10.)
            np.testing.assert_allclose(particle_parallel_temperature(ds, species='electron',
                                       profile=replace(NR, electron_mass_ratio=mass)), 3)
        wrong = particle_parallel_temperature(ds, species='electron', profile=NR)
        self.assertFalse(np.allclose(wrong, 3))

    def test_general_algebra_only_not_species_support(self):
        n, j, m = (xr.DataArray(v) for v in (2., 20., 206.))
        self.assertEqual(_center_parallel(m, j, n, 4., 2.).item(), 6.)
        with self.assertRaises(DerivedSemanticsError):
            particle_number_density(moments(), species='He2+', profile=NR)

    def test_relativistic_cold_beam_and_gate(self):
        # c=1, m=1, n=2, u=.6: gamma=1.25, M=.9, J=1.2.
        # NR subtraction leaves .18 although the beam has no thermal spread.
        ds = moments(pihpar=.9, jihpar=1.2, pihperp=0.)
        np.testing.assert_allclose(_center_parallel(ds.pihpar, ds.jihpar, ds.nih, 1, 1), .18)
        profile = replace(NR, relativistic=True)
        for fn in NR_OPS:
            with self.assertRaisesRegex(DerivedSemanticsError, 'nonrelativistic'):
                fn(ds, species='ion', profile=profile)
        with self.assertRaises(DerivedSemanticsError):
            firehose_parameter(ds, profile=profile)
        for fn in ALL_SPECIES_OPS[:4]:
            fn(ds, species='ion', profile=profile)

    def test_raw_and_perp_semantics(self):
        profile = replace(NR, relativistic=True, smooth_part_mom=True, doublesmooth=True)
        ds = moments()
        raw = raw_parallel_stress(ds, species='ion', profile=profile)
        q = particle_perpendicular_pressure(ds, species='ion', profile=profile)
        np.testing.assert_array_equal(q, 6)
        self.assertFalse(raw.attrs['centered'])
        self.assertEqual(raw.attrs['moment_model'], 'gamma_weighted')
        self.assertEqual(q.attrs['smoothing_passes'], 2)
        np.testing.assert_array_equal(particle_perpendicular_temperature(ds, species='ion', profile=profile), 3)

    def test_nonsquare_and_volume_preserve_values_coordinates(self):
        for shape, dims in (((3, 2), ('x', 'y')), ((3, 2, 4), ('x', 'y', 'z'))):
            q = np.arange(np.prod(shape)).reshape(shape).astype(float)
            ds = moments(shape, dims, pihperp=q).assign_coords({d: np.arange(s)*.2 for d, s in zip(dims, shape)})
            ds = ds.assign_coords(time=5.05, source_segment='005', local_frame_index=0)
            result = particle_perpendicular_temperature(ds, species='ion', profile=NR)
            self.assertEqual(result.dims, dims)
            np.testing.assert_array_equal(result, q/2)
            for coord in ds.coords:
                xr.testing.assert_identical(result[coord], ds[coord])

    def test_terminology_is_output_only(self):
        ds = moments()
        original = particle_number_density(ds, species='ion', profile=NR)
        renamed = original.rename('provisional_new_term')
        renamed.attrs['scientific_quantity'] = 'provisional_new_term'
        np.testing.assert_array_equal(renamed, original)
        self.assertEqual(NR.storage_for('number_density', species='ion'), 'nih')
        self.assertEqual(renamed.attrs['source_storage_names'], ('nih',))

    def test_functions_resolve_mapping_through_profile(self):
        # Probe the boundary only, not a claim to support another on-disk schema.
        original = ParticleMomentProfile.storage_for
        def mapped(profile, quantity, **kwargs):
            return 'probe_' + original(profile, quantity, **kwargs)
        ds = moments().rename({key: 'probe_' + key for key in moments().data_vars})
        with patch.object(ParticleMomentProfile, 'storage_for', mapped):
            for fn in ALL_SPECIES_OPS:
                fn(ds, species='ion', profile=NR)
            np.testing.assert_array_equal(firehose_parameter(ds, profile=NR), 1)

    def test_invalid_density_and_perpendicular_moments_no_mutation(self):
        ds = moments((7,), ('x',), nih=[0, -1, 2, 2, 2, np.inf, np.nan],
                     pihperp=[6, 6, 0, -1, np.nan, 1, 1])
        before = ds.copy(deep=True)
        density = particle_number_density(ds, species='ion', profile=NR)
        temp = particle_perpendicular_temperature(ds, species='ion', profile=NR)
        np.testing.assert_array_equal(validity_mask(density), [True, False, True, True, True, False, False])
        np.testing.assert_array_equal(validity_mask(temp), [False, False, True, False, False, False, False])
        self.assertEqual(temp[2], 0)
        xr.testing.assert_identical(ds, before)

    def test_corrected_zero_negative_and_tiny_negative(self):
        ds = moments((6,), ('x',), pihpar=[0, -1, 1-1e-14, 1, np.inf, 1],
                     jihpar=[0, 0, np.sqrt(2), np.nan, 0, np.inf])
        result = particle_parallel_pressure(ds, species='ion', profile=NR)
        np.testing.assert_array_equal(validity_mask(result), [True, False, False, False, False, False])
        self.assertEqual(result[0], 0)
        total = particle_total_temperature(ds, species='ion', profile=NR)
        self.assertTrue(np.isnan(total[2]))  # Positive perp cannot hide invalid par.

    def test_firehose_null_tiny_negative_and_invalid_components(self):
        ds = moments((6,), ('x',), bx=[0, 1e-10, 1, np.nan, np.inf, 1],
                     pihpar=[8, 8, 8, 8, 8, -1], pihperp=2)
        result = firehose_parameter(ds, profile=NR)
        np.testing.assert_array_equal(validity_mask(result), [False, True, True, False, False, False])
        self.assertLess(result[1], -1e19)
        self.assertEqual(result[2], -5.)
        self.assertNotIn('pi', result.attrs['source_storage_names'])
        self.assertNotIn('pc', result.attrs['source_storage_names'])

    def test_overflow_and_nonfinite_sources(self):
        ds = moments(jihpar=1e308)
        with np.errstate(over='ignore', invalid='ignore'):
            result = particle_parallel_pressure(ds, species='ion', profile=NR)
        self.assertTrue(np.isnan(result).all())
        for value in (np.nan, np.inf, -np.inf, -1):
            self.assertTrue(np.isnan(raw_parallel_stress(moments(pihpar=value), species='ion', profile=NR)).all())

    def test_profile_input_conflicts(self):
        for key, value in [('relativistic', True), ('electron_mass_ratio', .01),
                           ('producer_id', 'unknown'), ('units_status', 'SI')]:
            ds = moments()
            ds.attrs[key] = value
            with self.assertRaises(DerivedSemanticsError):
                particle_number_density(ds, species='ion', profile=NR)
        with self.assertRaises(DerivedSemanticsError):
            particle_number_density(moments(), species='ion', profile=None)

    def test_source_scientific_identity_conflicts(self):
        for key, value in [('species', 'electron'), ('population', 'fluid'),
                           ('scientific_quantity', 'parallel_pressure'), ('mass_ratio', .04),
                           ('centered', True)]:
            ds = moments()
            ds.pihpar.attrs[key] = value
            with self.assertRaises(DerivedSemanticsError):
                raw_parallel_stress(ds, species='ion', profile=NR)

    def test_validity_helper_requires_derived_output(self):
        with self.assertRaises(DerivedQuantityError):
            validity_mask(moments().nih)
        ds = moments(nih=np.array([[0, 2], [-1, np.nan], [np.inf, 4]]))
        result = particle_number_density(ds, species='ion', profile=NR)
        np.testing.assert_array_equal(validity_mask(result), np.isfinite(result))

    def test_region_selection_provenance_detached(self):
        ds = moments()
        ds.attrs['region'] = 'externally selected box'
        ds.pihpar.attrs['selection'] = ds.jihpar.attrs['selection'] = ds.nih.attrs['selection'] = {'label': 'chosen'}
        field = particle_parallel_temperature(ds, species='ion', profile=NR)
        self.assertEqual(field.attrs['region'], 'externally selected box')
        self.assertEqual(field.attrs['selection'], {'label': 'chosen'})
        field.attrs['selection']['label'] = 'changed'
        self.assertEqual(ds.pihpar.attrs['selection']['label'], 'chosen')


class AlignmentTests(unittest.TestCase):
    def test_missing_variable_and_dataset_requirement(self):
        for source in (moments().drop_vars('nih'), moments().nih):
            with self.assertRaises(DerivedAlignmentError):
                particle_perpendicular_temperature(source, species='ion', profile=NR)

    def test_dimensions_spatial_and_time_mismatch(self):
        for dims in (('y', 'x'), ('x2', 'y'), ('time', 'y')):
            ds = moments()
            ds['pihperp'] = xr.DataArray(np.ones((2, 3) if dims == ('y', 'x') else (3, 2)), dims=dims)
            with self.assertRaises(DerivedAlignmentError):
                particle_perpendicular_temperature(ds, species='ion', profile=NR)

    def test_observable_external_coordinate_stamps(self):
        for key, first, second in [('time', 5.05, 5.0500000001), ('source_segment', '005', '006'),
                                    ('local_frame_index', 0, 1), ('x', [0, 1, 2], [0, 1, 3])]:
            ds = moments()
            ds.nih.attrs[key] = first
            ds.pihperp.attrs[key] = second
            with self.assertRaises(DerivedAlignmentError):
                particle_perpendicular_temperature(ds, species='ion', profile=NR)

    def test_scalar_stamp_conflicts_with_shared_coordinates(self):
        ds = moments().assign_coords(time=5.05)
        ds.nih.attrs['time'] = ds.pihperp.attrs['time'] = 6.
        with self.assertRaises(DerivedAlignmentError):
            particle_perpendicular_temperature(ds, species='ion', profile=NR)

    def test_time_and_provenance_validation(self):
        for times in ([1, 1, 2], [1, 0, 2], [1, np.nan, 2]):
            ds = moments((3, 2), ('time', 'x')).assign_coords(time=times)
            with self.assertRaises(DerivedAlignmentError):
                particle_number_density(ds, species='ion', profile=NR)
        ds = moments((3, 2), ('time', 'x'))
        with self.assertRaises(DerivedAlignmentError):
            particle_number_density(ds, species='ion', profile=NR)
        for key, vals in [('local_frame_index', [0., np.nan, 2.]), ('source_segment', ['005', '', '006'])]:
            ds = ds.assign_coords(time=[1, 2, 3], **{key: ('time', vals)})
            with self.assertRaises(DerivedAlignmentError):
                particle_number_density(ds, species='ion', profile=NR)
            ds = ds.drop_vars(key)

    def test_no_alignment_or_sample_compute(self):
        ds = moments()
        with patch.object(xr, 'align', side_effect=AssertionError('align')), \
                patch.object(xr.DataArray, 'compute', side_effect=AssertionError('compute')):
            particle_total_temperature(ds, species='ion', profile=NR)

    def test_lazy_coordinate_rejected_without_computing(self):
        def fail():
            raise AssertionError('coordinate task ran')
        coord = da.from_delayed(delayed(fail)(), shape=(3,), dtype=float)
        ds = moments().assign_coords(external_coordinate=('x', coord))
        with self.assertRaises(DerivedAlignmentError):
            particle_perpendicular_temperature(ds, species='ion', profile=NR)

    def test_region_conflict_and_storage_identity(self):
        ds = moments()
        ds.nih.attrs['region'] = 'left'
        ds.pihperp.attrs['region'] = 'right'
        with self.assertRaises(DerivedAlignmentError):
            particle_perpendicular_temperature(ds, species='ion', profile=NR)
        ds = moments()
        ds.nih.attrs['storage_name'] = 'ni'
        with self.assertRaises(DerivedSemanticsError):
            particle_number_density(ds, species='ion', profile=NR)


class LazyAndCompositionTests(unittest.TestCase):
    def source(self):
        calls = []
        def read(name, event):
            calls.append((name, event))
            return moments()[name].values + (event if name in ('pihpar', 'pehpar') else 0)
        ds = xr.Dataset({name: (('time', 'x', 'y'), da.stack([
            da.from_delayed(delayed(read)(name, i), shape=(3, 2), dtype=float) for i in range(3)]))
            for name in moments().data_vars}, coords={'time': [5.05, 5.1, 5.15],
                'source_segment': ('time', ['005']*3), 'local_frame_index': ('time', [0, 1, 2])})
        return ds, calls

    def test_lazy_required_variables_one_event(self):
        ds, calls = self.source()
        temp = particle_total_temperature(ds, species='ion', profile=NR)
        self.assertEqual(calls, [])
        self.assertIsInstance(temp.data, da.Array)
        selected = temp.isel(time=1).compute(scheduler='synchronous')
        self.assertEqual(Counter(calls), Counter((name, 1) for name in ('nih', 'jihpar', 'pihpar', 'pihperp')))
        self.assertEqual(selected.source_segment.item(), '005')
        self.assertEqual(selected.local_frame_index.item(), 1)
        self.assertEqual(selected.dims, ('x', 'y'))
        np.testing.assert_allclose(selected, 19/6)

    def test_firehose_reads_exactly_eleven_event_fields(self):
        ds, calls = self.source()
        result = firehose_parameter(ds, profile=NR)
        mask = validity_mask(result)
        self.assertEqual(calls, [])
        result.isel(time=2).compute(scheduler='synchronous')
        names = ('neh', 'nih', 'pehpar', 'pihpar', 'pehperp', 'pihperp', 'jhpar', 'jihpar', 'bx', 'by', 'bz')
        self.assertEqual(Counter(calls), Counter((name, 2) for name in names))
        self.assertIsInstance(mask.data, da.Array)

    def test_spacetime_composition(self):
        from kglobal_analysis.analysis import spacetime
        ds, calls = self.source()
        field = particle_total_temperature(ds, species='ion', profile=NR)
        result = spacetime(field, line_axis='x', fixed_indices={'y': 1}, times=[5.05, 5.15])
        self.assertEqual(result.dims, ('time', 'x'))
        np.testing.assert_allclose(result[:, 0], [3, 10/3])
        self.assertEqual(Counter(calls), Counter((name, event) for name in ('nih','jihpar','pihpar','pihperp') for event in (0,2)))

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'optional plot extra unavailable')
    def test_map_composition(self):
        from kglobal_analysis.plotting import plot_scalar_map
        ds, calls = self.source()
        field = particle_total_temperature(ds, species='ion', profile=NR).isel(time=0).compute(scheduler='synchronous')
        result = plot_scalar_map(field, horizontal='x', vertical='y')
        np.testing.assert_array_equal(result.artist.get_array(), field.values.T)
        self.assertEqual(result.metadata['population'], 'particle')
        result.figure.clear()

    def test_no_plot_import_in_derived(self):
        subprocess.run([sys.executable, '-c',
                        'import sys; import kglobal_analysis.derived; assert "matplotlib" not in sys.modules'], check=True)


class MovieIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'param').write_text(PARAM + '''#define nx 3
#define ny 2
#define nz 1
#define pex 1
#define pey 1
#define pez 1
#define dt .025
#define n_movieout 2
''')
        for name in moments().data_vars:
            frames = [moments()[name].values + (i if name == 'pihpar' else 0) for i in range(3)]
            (self.root/f'movie.{name}.005').write_bytes(b''.join(
                frame.astype('<i2').tobytes(order='F') for frame in frames))
        (self.root/'movie.log.005').write_text('-32768 32767\n'*18*3)
        (self.root/'p3d.stdout.005').write_text(''.join(f'movie output, t= {t}\n' for t in (5.05, 5.1, 5.15)))

    def test_actual_decoder_culling_and_zero_sample_construction(self):
        case = KGlobalCase(self.root)
        with patch('numpy.fromfile', side_effect=AssertionError('sample read at construction')):
            profile = case.particle_moment_profile()
            ds = case.movie_dataset(list(moments().data_vars), byteorder='little')
            field = firehose_parameter(ds, profile=profile)
        calls = []
        original = np.fromfile
        def read(stream, *, dtype, count):
            calls.append((Path(stream.name).name, stream.tell(), count))
            return original(stream, dtype=dtype, count=count)
        with patch('numpy.fromfile', side_effect=read):
            selected = field.isel(time=1).compute(scheduler='synchronous')
        expected = [f'movie.{name}.005' for name in field.attrs['source_storage_names']]
        self.assertEqual(Counter(calls), Counter((name, 12, 6) for name in expected))
        np.testing.assert_allclose(selected, .75)

    def test_missing_movie_coverage_still_rejected_by_dataset(self):
        from kglobal_analysis import MovieAlignmentError
        (self.root/'movie.nih.005').unlink()
        with patch('numpy.fromfile', side_effect=AssertionError('sample read')):
            with self.assertRaises(MovieAlignmentError):
                KGlobalCase(self.root).movie_dataset(['pihpar', 'nih'], byteorder='little')
