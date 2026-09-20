"""Declared geometry, discrete science, and event-bounded reconstruction."""
from collections import Counter
from dataclasses import replace, FrozenInstanceError
import importlib.util
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import dask.array as da
from dask import delayed
import numpy as np
import xarray as xr

from kglobal_analysis import KGlobalCase, parse_parameters
from kglobal_analysis.geometry import MovieGeometry, GeometryError, attach_movie_geometry
from kglobal_analysis.flux import magnetic_flux_2d, magnetic_flux_diagnostics, FluxError
from kglobal_analysis.derived import ParticleMomentProfile, particle_number_density

PARAM = '''#define nx 4
#define ny 3
#define nz 1
#define pex 2
#define pey 2
#define pez 1
#define lx 4.
#define ly 9.
#define lz 1.
#define init_scheme initrecon
#define boundary_condition periodic
#define movie_header "movie_kglobal3.0.h"
#define double_byte
#define dt .025
#define n_movieout 2
'''
G = MovieGeometry(8, 6, 1, 4., 9., 1.)


def field(bx=0., by=0., geometry=G):
    shape = (geometry.Nx, geometry.Ny)
    return xr.Dataset({name: (('x', 'y'), np.broadcast_to(value, shape).copy())
                       for name, value in (('bx', bx), ('by', by))})


def derivative(a, axis, spacing):
    return (np.roll(a, -1, axis)-np.roll(a, 1, axis))/(2*spacing)


def mode(g=G):
    x = 2*np.pi*np.arange(g.Nx)[:, None]/g.Nx
    y = 2*np.pi*np.arange(g.Ny)[None, :]/g.Ny
    return np.sin(x)*np.cos(y) + .3*np.cos(2*x+y)


def from_psi(psi, g=G):
    return field(derivative(psi, 1, g.dy), -derivative(psi, 0, g.dx), g)


def lazy_source():
    calls = []
    def load(name, event):
        calls.append((name, event))
        if name == 'nih':
            return np.full((G.Nx, G.Ny), 2.+event)
        if name == 'bz':
            raise AssertionError('Bz must never be read')
        return from_psi((event+1)*mode())[name].values
    ds = xr.Dataset({name: (('time', 'x', 'y'), da.stack([
        da.from_delayed(delayed(load)(name, i), shape=(G.Nx, G.Ny), dtype=float)
        for i in range(3)])) for name in ('bx', 'by', 'bz', 'nih')},
        coords={'time': [5.05, 5.10, 5.15], 'source_segment': ('time', ['005']*3),
                'local_frame_index': ('time', [0, 1, 2])}, attrs={'selection': {'kind': 'full'}})
    return ds, calls


class GeometryTests(unittest.TestCase):
    def test_parameter_resolution_and_nonsquare_centers(self):
        g = MovieGeometry.from_parameters(parse_parameters(PARAM))
        self.assertEqual((g.Nx, g.Ny, g.Nz), (8, 6, 1))
        self.assertEqual((g.dx, g.dy), (.5, 1.5))
        np.testing.assert_array_equal(g.x, (np.arange(8)+.5)*.5)
        np.testing.assert_array_equal(g.y, (np.arange(6)+.5)*1.5)
        self.assertTrue(g.degenerate_z)
        self.assertFalse(hasattr(g, 'z'))
        self.assertFalse(hasattr(g, 'dz'))

    def test_case_entry(self):
        with TemporaryDirectory() as root:
            (Path(root)/'param').write_text(PARAM)
            g = KGlobalCase(root).movie_geometry()
            self.assertEqual(g.Nx, 8)
            self.assertEqual(g.parameter_source, str((Path(root)/'param').resolve()))

    def test_lengths_literal_only(self):
        for name in ('lx', 'ly', 'lz'):
            for value in ('0', '-1', 'nan', 'inf', '1e999', '2*pi', '4/2', 'LENGTH'):
                with self.subTest(name=name, value=value), self.assertRaises(GeometryError):
                    MovieGeometry.from_parameters(parse_parameters(PARAM+f'#define {name} {value}\n'))
        g = MovieGeometry.from_parameters(parse_parameters(PARAM+'#define lx ( 4D0 )\n'))
        self.assertEqual(g.lx, 4)

    def test_missing_metadata(self):
        for name in ('nx', 'ny', 'nz', 'pex', 'pey', 'pez', 'lx', 'ly', 'lz', 'init_scheme', 'boundary_condition'):
            with self.subTest(name=name), self.assertRaises(GeometryError):
                MovieGeometry.from_parameters(parse_parameters(PARAM+f'#undef {name}\n'))

    def test_unknown_initializer_boundary_and_layout(self):
        for statement in ('init_scheme unknown', 'boundary_condition gem_reconnection',
                          'boundary_condition 0', 'movie_header "unknown.h"', 'four_byte', 'mult_species'):
            with self.subTest(statement=statement), self.assertRaises(GeometryError):
                MovieGeometry.from_parameters(parse_parameters(PARAM+f'#define {statement}\n'))

    def test_conflicting_typed_metadata(self):
        p = parse_parameters(PARAM)
        for changed in (replace(p, nx=7), replace(p, double_byte=False), replace(p, ny=True),
                        replace(p, definitions={**p.definitions, 'double_byte': '', 'four_byte': ''})):
            with self.assertRaises(GeometryError):
                MovieGeometry.from_parameters(changed)

    def test_frozen_explicit_contract(self):
        with self.assertRaises(FrozenInstanceError):
            G.Nx = 2
        for changes in ({'geometry_profile': 'unknown'}, {'periodic_x': False}, {'periodic_y': 1},
                        {'origin_x': 1}, {'sample_offset_y': 0}, {'Nx': True}, {'Ny': 0},
                        {'lx': np.inf}, {'ly': 0}, {'lz': '1'}, {'parameter_source': ''}):
            with self.subTest(changes=changes), self.assertRaises(GeometryError):
                replace(G, **changes)

    def test_three_dimensional_metadata_rejected(self):
        for statement in ('nz 2', 'pez 2'):
            with self.assertRaises(GeometryError):
                MovieGeometry.from_parameters(parse_parameters(PARAM+f'#define {statement}\n'))
        with self.assertRaises(GeometryError):
            replace(G, Nz=2)

    def test_attach_dataarray_preserves_order_samples_attrs(self):
        a = field(2).bx.transpose('y', 'x')
        a.attrs['selection'] = {'box': [1, 2]}
        result = attach_movie_geometry(a, G)
        self.assertEqual(result.dims, ('y', 'x'))
        self.assertIs(result.data, a.data)
        self.assertNotIn('x', a.coords)
        result.attrs['selection']['box'][0] = 9
        self.assertEqual(a.attrs['selection']['box'][0], 1)
        self.assertEqual(result.attrs['coordinate_units'], 'code_normalized')
        self.assertNotIn('z', result.coords)

    def test_attach_dataset_lazy_and_provenance(self):
        ds, calls = lazy_source()
        result = attach_movie_geometry(ds, G)
        self.assertEqual(calls, [])
        self.assertIs(result.bx.data, ds.bx.data)
        for name in ('time', 'source_segment', 'local_frame_index'):
            xr.testing.assert_identical(result[name], ds[name])
        self.assertEqual(result.bx.dims, ds.bx.dims)
        self.assertNotIn('geometry_profile', ds.attrs)

    def test_conflicting_x(self):
        with self.assertRaisesRegex(GeometryError, 'x coordinates'):
            attach_movie_geometry(field().assign_coords(x=np.arange(G.Nx)), G)

    def test_conflicting_y(self):
        with self.assertRaisesRegex(GeometryError, 'y coordinates'):
            attach_movie_geometry(field().assign_coords(y=G.y+.1), G)

    def test_compatible_coordinates(self):
        ds = field().assign_coords(x=G.x, y=G.y)
        once = attach_movie_geometry(ds, G)
        xr.testing.assert_identical(once, attach_movie_geometry(once, G))

    def test_wrong_size_z_and_conflicting_attrs(self):
        for ds in (field().isel(x=slice(1, None)), field().expand_dims(z=[0]),
                   field().assign_coords(z=0), field().assign_attrs(periodic_x=False),
                   field().assign_attrs(grid_layout='staggered')):
            with self.assertRaises(GeometryError):
                attach_movie_geometry(ds, G)
        ds = field().assign_coords(x=G.x)
        ds.x.attrs['units'] = 'cm'
        with self.assertRaises(GeometryError):
            attach_movie_geometry(ds, G)

    def test_no_geometry_or_implicit_inference(self):
        with self.assertRaises(GeometryError):
            attach_movie_geometry(field(), None)

    def test_real_metadata_without_samples(self):
        root = Path(__file__).resolve().parents[2]/'validation-data/hcs_large_005'
        if not root.exists():
            self.skipTest('private metadata fixture unavailable')
        with patch('numpy.fromfile', side_effect=AssertionError('binary sample read')):
            g = KGlobalCase(root).movie_geometry()
        self.assertEqual((g.Nx, g.Ny, g.Nz), (8192, 4096, 1))
        self.assertEqual(g.dx, 6.2831855/8192)
        self.assertEqual(g.dy, 3.1415926/4096)
        self.assertEqual(g.lz, 1)


class FluxScienceTests(unittest.TestCase):
    def test_zero_field(self):
        result = magnetic_flux_diagnostics(field(), geometry=G)
        for a in result.data_vars.values():
            np.testing.assert_array_equal(a, 0)
        self.assertEqual(result.psi.dtype, np.float64)

    def test_constant_bx(self):
        psi = magnetic_flux_2d(field(bx=2), geometry=G)
        np.testing.assert_allclose(psi, np.broadcast_to(2*(G.y-G.y.mean()), psi.shape))

    def test_constant_by(self):
        psi = magnetic_flux_2d(field(by=3), geometry=G)
        np.testing.assert_allclose(psi, np.broadcast_to(-3*(G.x-G.x.mean())[:, None], psi.shape))

    def test_mean_diagnostics_have_no_wrap_seam(self):
        result = magnetic_flux_diagnostics(field(bx=2, by=3), geometry=G)
        np.testing.assert_array_equal(result.reconstructed_bx, 2)
        np.testing.assert_array_equal(result.reconstructed_by, 3)
        np.testing.assert_array_equal(result.residual_magnitude, 0)
        np.testing.assert_array_equal(result.reconstructed_divergence, 0)

    def test_single_mode_sign_and_orientation(self):
        original = np.sin(2*np.pi*np.arange(G.Nx)[:, None]/G.Nx)*np.cos(2*np.pi*np.arange(G.Ny)[None, :]/G.Ny)
        psi = magnetic_flux_2d(from_psi(original), geometry=G)
        self.assertEqual(psi.dims, ('x', 'y'))
        np.testing.assert_allclose(psi, original, atol=2e-15)

    def test_mixed_modes_unequal_spacing_nonsquare(self):
        result = magnetic_flux_diagnostics(from_psi(mode()), geometry=G)
        np.testing.assert_allclose(result.psi, mode(), atol=2e-15)
        np.testing.assert_allclose(result.residual_magnitude, 0, atol=2e-15)
        np.testing.assert_allclose(result.input_divergence, 0, atol=2e-15)
        np.testing.assert_allclose(result.reconstructed_divergence, 0, atol=2e-15)

    def test_odd_even_combinations(self):
        for nx, ny in ((7, 5), (7, 6), (8, 5), (8, 6)):
            g = replace(G, Nx=nx, Ny=ny)
            with self.subTest(shape=(nx, ny)):
                np.testing.assert_allclose(magnetic_flux_2d(from_psi(mode(g), g), geometry=g), mode(g), atol=3e-15)

    def test_additive_gauge(self):
        psi = magnetic_flux_2d(from_psi(mode()+19), geometry=G)
        np.testing.assert_allclose(psi, mode(), atol=5e-15)
        self.assertAlmostEqual(float(psi.mean()), 0, places=14)

    def test_mean_plus_periodic(self):
        ds = from_psi(mode())
        ds['bx'] = ds.bx + 2
        ds['by'] = ds.by - 3
        psi = magnetic_flux_2d(ds, geometry=G)
        expected = mode() + 2*(G.y-G.y.mean())[None, :] + 3*(G.x-G.x.mean())[:, None]
        np.testing.assert_allclose(psi, expected, atol=3e-15)
        self.assertAlmostEqual(float(psi.mean()), 0, places=14)

    def test_gradient_contamination_projection(self):
        ds = from_psi(mode())
        gradient_x = derivative(mode(), 0, G.dx)
        gradient_y = derivative(mode(), 1, G.dy)
        ds['bx'] = ds.bx + gradient_x
        ds['by'] = ds.by + gradient_y
        d = magnetic_flux_diagnostics(ds, geometry=G)
        np.testing.assert_allclose(d.psi, mode(), atol=3e-15)
        np.testing.assert_allclose(d.residual_bx, gradient_x, atol=3e-15)
        np.testing.assert_allclose(d.residual_by, gradient_y, atol=3e-15)
        self.assertGreater(float(abs(d.input_divergence).max()), .1)
        np.testing.assert_allclose(d.reconstructed_divergence, 0, atol=3e-15)

    def test_joint_nyquist_null_content(self):
        for bx in ((-1.)**np.arange(G.Nx)[:, None],
                   (-1.)**np.arange(G.Ny)[None, :],
                   (-1.)**(np.arange(G.Nx)[:, None]+np.arange(G.Ny)[None, :])):
            with np.errstate(all='raise'):
                d = magnetic_flux_diagnostics(field(bx=bx), geometry=G)
            np.testing.assert_allclose(d.psi, 0, atol=2e-15)
            np.testing.assert_allclose(d.residual_bx, np.broadcast_to(bx, (G.Nx, G.Ny)), atol=1e-15)
            np.testing.assert_allclose(d.null_magnitude, 1, atol=1e-15)

    def test_nyquist_one_axis_nonnull_other(self):
        original = (-1.)**np.arange(G.Nx)[:, None]*np.sin(2*np.pi*np.arange(G.Ny)[None, :]/G.Ny)
        d = magnetic_flux_diagnostics(from_psi(original), geometry=G)
        np.testing.assert_allclose(d.psi, original, atol=2e-15)
        np.testing.assert_allclose(d.residual_magnitude, 0, atol=2e-15)

    def test_nonfinite_entire_evaluated_event(self):
        for bad in (np.nan, np.inf, -np.inf):
            ds = from_psi(mode()).expand_dims(time=[1, 2]).copy(deep=True)
            ds.bx.values[1, 2, 3] = bad
            d = magnetic_flux_diagnostics(ds, geometry=G)
            for a in d.data_vars.values():
                self.assertTrue(np.isfinite(a.isel(time=0)).all())
                self.assertTrue(np.isnan(a.isel(time=1)).all())

    def test_arbitrary_input_dimension_order_preserved(self):
        ds = from_psi(mode()).expand_dims(time=[1, 2]).transpose('y', 'time', 'x')
        psi = magnetic_flux_2d(ds, geometry=G)
        self.assertEqual(psi.dims, ('y', 'time', 'x'))
        np.testing.assert_allclose(psi.isel(time=1).transpose('x', 'y'), mode(), atol=2e-15)

    def test_attrs_and_nonmutation(self):
        ds = field().assign_attrs(selection={'kind': 'whole'})
        before = ds.copy(deep=True)
        psi = magnetic_flux_2d(ds, geometry=G)
        for name in ('scientific_quantity', 'convention', 'reconstruction_method', 'derivative_operator',
                     'boundary_condition', 'gauge', 'mean_field_treatment', 'coordinate_units',
                     'source_variables', 'units_status'):
            self.assertIn(name, psi.attrs)
        self.assertEqual(psi.attrs['source_variables'], ('bx', 'by'))
        psi.attrs['selection']['kind'] = 'changed'
        xr.testing.assert_identical(ds, before)


class AlignmentTests(unittest.TestCase):
    def test_missing_fields_and_dimensions(self):
        for ds in (field().drop_vars('by'), field().bx, field().expand_dims(z=[0]),
                   field().expand_dims(extra=[0])):
            with self.assertRaises((FluxError, GeometryError)):
                magnetic_flux_2d(ds, geometry=G)
        ds = field()
        ds['by'] = ds.by.transpose('y', 'x')
        with self.assertRaises(FluxError):
            magnetic_flux_2d(ds, geometry=G)

    def test_conflicting_provenance_stamps(self):
        for key in ('x', 'y', 'time', 'source_segment', 'local_frame_index', 'region', 'selection', 'fixed_indices'):
            ds = field()
            ds.bx.attrs[key] = 'a'
            ds.by.attrs[key] = 'b'
            with self.subTest(key=key), self.assertRaises((FluxError, GeometryError)):
                magnetic_flux_2d(ds, geometry=G)

    def test_conflict_with_shared_coordinates(self):
        ds = field().expand_dims(time=[2])
        ds.bx.attrs['time'] = ds.by.attrs['time'] = [1]
        with self.assertRaises(FluxError):
            magnetic_flux_2d(ds, geometry=G)

    def test_time_metadata_failures(self):
        for t in ([1, 1], [2, 1], [1, np.nan]):
            with self.assertRaises(FluxError):
                magnetic_flux_2d(field().expand_dims(time=t), geometry=G)
        with self.assertRaises(FluxError):
            magnetic_flux_2d(field().expand_dims(time=2), geometry=G)

    def test_dataset_provenance_conflicts(self):
        ds = field().expand_dims(time=[2]).assign_attrs(time=[1])
        with self.assertRaises(FluxError):
            magnetic_flux_2d(ds, geometry=G)
        ds = field().assign_attrs(selection='whole')
        ds.bx.attrs['selection'] = ds.by.attrs['selection'] = 'subset'
        with self.assertRaises(FluxError):
            magnetic_flux_2d(ds, geometry=G)

    def test_lazy_metadata_rejected_without_evaluation(self):
        ds, calls = lazy_source()
        ds = ds.assign_coords(source_segment=('time', da.from_delayed(
            delayed(lambda: (_ for _ in ()).throw(AssertionError('metadata computed')))(),
            shape=(3,), dtype='U3')))
        with self.assertRaises(FluxError):
            magnetic_flux_2d(ds, geometry=G)
        self.assertEqual(calls, [])

    def test_storage_and_scientific_identity(self):
        for key, value in (('storage_name', 'bz'), ('scientific_quantity', 'pressure'),
                           ('units_status', 'SI'), ('grid_layout', 'staggered'), ('dx', 9)):
            ds = field()
            ds.by.attrs[key] = value
            with self.subTest(key=key), self.assertRaises((FluxError, GeometryError)):
                magnetic_flux_2d(ds, geometry=G)

    def test_reject_multiple_event_chunks(self):
        ds, calls = lazy_source()
        with self.assertRaisesRegex(FluxError, 'one-event'):
            magnetic_flux_2d(ds.assign({n: ds[n].chunk(time=3) for n in ('bx', 'by')}), geometry=G)
        self.assertEqual(calls, [])


class LazyTests(unittest.TestCase):
    def test_construction_no_samples_and_one_psi_event(self):
        ds, calls = lazy_source()
        attached = attach_movie_geometry(ds, G)
        psi = magnetic_flux_2d(attached, geometry=G)
        diagnostics = magnetic_flux_diagnostics(attached, geometry=G)
        self.assertEqual(calls, [])
        out = psi.isel(time=1).compute(scheduler='synchronous')
        self.assertEqual(Counter(calls), Counter([('bx', 1), ('by', 1)]))
        self.assertEqual(out.source_segment.item(), '005')
        self.assertEqual(out.local_frame_index.item(), 1)
        np.testing.assert_allclose(out, 2*mode(), atol=4e-15)
        self.assertIsInstance(diagnostics.psi.data, da.Array)

    def test_diagnostics_one_event(self):
        ds, calls = lazy_source()
        result = magnetic_flux_diagnostics(ds, geometry=G).isel(time=2).compute(scheduler='synchronous')
        self.assertEqual(Counter(calls), Counter([('bx', 2), ('by', 2)]))
        np.testing.assert_allclose(result.residual_magnitude, 0, atol=5e-15)

    def test_nonfinite_lazy_event_is_isolated(self):
        ds, calls = lazy_source()
        ds['bx'] = ds.bx.where(ds.time != 5.10)
        diagnostics = magnetic_flux_diagnostics(ds, geometry=G)
        self.assertEqual(calls, [])
        result = diagnostics.isel(time=1).compute(scheduler='synchronous')
        self.assertEqual(Counter(calls), Counter([('bx', 1), ('by', 1)]))
        for array in result.data_vars.values():
            self.assertTrue(np.isnan(array).all())

    def test_fft_kernel_receives_only_one_selected_plane(self):
        import kglobal_analysis.flux as flux
        ds, calls = lazy_source()
        shapes = []
        original = flux._plane
        def plane(bx, by, **kwargs):
            shapes.append((bx.shape, by.shape))
            return original(bx, by, **kwargs)
        with patch.object(flux, '_plane', new=plane):
            psi = magnetic_flux_2d(ds, geometry=G)
            self.assertEqual(shapes, [])
            psi.isel(time=0).compute(scheduler='synchronous')
        self.assertEqual(shapes, [((8, 6), (8, 6))])
        self.assertEqual(Counter(calls), Counter([('bx', 0), ('by', 0)]))

    def test_spatial_rechunk_only(self):
        ds, calls = lazy_source()
        ds = ds.assign({n: ds[n].chunk({'x': 2, 'y': 3}) for n in ('bx', 'by')})
        result = magnetic_flux_2d(ds, geometry=G)
        self.assertEqual(calls, [])
        self.assertEqual(result.chunksizes['time'], (1, 1, 1))
        result.isel(time=2).compute(scheduler='synchronous')
        self.assertEqual(Counter(calls), Counter([('bx', 2), ('by', 2)]))

    def test_m14_combined_event_culling(self):
        ds, calls = lazy_source()
        ds = attach_movie_geometry(ds, G)
        scalar = particle_number_density(ds, species='ion', profile=ParticleMomentProfile(.04, False))
        psi = magnetic_flux_2d(ds, geometry=G)
        combined = xr.Dataset(dict(density=scalar, psi=psi))
        self.assertEqual(calls, [])
        result = combined.isel(time=1).compute(scheduler='synchronous')
        self.assertEqual(Counter(calls), Counter([('bx', 1), ('by', 1), ('nih', 1)]))
        np.testing.assert_array_equal(result.density, 3)

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'optional plotting unavailable')
    def test_m13_m14_contour_composition_orientation(self):
        from kglobal_analysis.plotting import plot_scalar_map
        ds, calls = lazy_source()
        ds = attach_movie_geometry(ds, G)
        scalar = particle_number_density(ds, species='ion', profile=ParticleMomentProfile(.04, False))
        frame = xr.Dataset(dict(density=scalar, psi=magnetic_flux_2d(ds, geometry=G))).isel(time=1).compute(scheduler='synchronous')
        # Capture the actual contour arguments, as well as the pcolormesh artist.
        from matplotlib.axes import Axes
        original = Axes.contour
        seen = []
        def contour(ax, x, y, values, **kwargs):
            seen.append((np.array(x), np.array(y), np.array(values)))
            return original(ax, x, y, values, **kwargs)
        with patch.object(Axes, 'contour', new=contour):
            result = plot_scalar_map(frame.density, horizontal='x', vertical='y', contours=frame.psi)
        np.testing.assert_array_equal(result.artist.get_array(), frame.density.values.T)
        np.testing.assert_array_equal(seen[0][0], G.x)
        np.testing.assert_array_equal(seen[0][1], G.y)
        np.testing.assert_array_equal(seen[0][2], frame.psi.values.T)
        self.assertEqual(frame.psi.dims, ('x', 'y'))
        self.assertEqual(Counter(calls), Counter([('bx', 1), ('by', 1), ('nih', 1)]))
        result.figure.clear()

    def test_no_rendering_import(self):
        subprocess.run([sys.executable, '-c', 'import sys; import kglobal_analysis.geometry; import kglobal_analysis.flux; assert "matplotlib" not in sys.modules'], check=True)

    def test_decoder_backed_culling(self):
        with TemporaryDirectory() as root:
            path = Path(root)
            (path/'param').write_text(PARAM)
            for name in ('bx', 'by', 'bz'):
                frames = [np.full((8, 6), i+1, dtype='<i2') for i in range(3)]
                (path/f'movie.{name}.005').write_bytes(b''.join(a.tobytes(order='F') for a in frames))
            (path/'movie.log.005').write_text('-32768 32767\n'*18*3)
            (path/'p3d.stdout.005').write_text(''.join(f'movie output, t= {t}\n' for t in (5.05, 5.10, 5.15)))
            with patch('numpy.fromfile', side_effect=AssertionError('eager sample read')):
                case = KGlobalCase(path)
                g = case.movie_geometry()
                ds = attach_movie_geometry(case.movie_dataset(['bx', 'by', 'bz'], byteorder='little'), g)
                psi = magnetic_flux_2d(ds, geometry=g)
            calls = []
            original = np.fromfile
            def read(stream, *, dtype, count):
                calls.append((Path(stream.name).name, stream.tell(), count))
                return original(stream, dtype=dtype, count=count)
            with patch('numpy.fromfile', side_effect=read):
                out = psi.isel(time=1).compute(scheduler='synchronous')
            self.assertEqual(Counter(calls), Counter((f'movie.{name}.005', 96, 48) for name in ('bx', 'by')))
            expected = 2*(g.y-g.y.mean())[None, :]-2*(g.x-g.x.mean())[:, None]
            np.testing.assert_allclose(out, expected)
