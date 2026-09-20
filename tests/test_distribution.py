"""Synthetic contract tests for reduced distributions, independent of raw data."""
from dataclasses import FrozenInstanceError
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr

from kglobal_analysis import KGlobalCase, DistributionError, DistributionMetadataError
from kglobal_analysis.distribution import DISTRIBUTION_QUANTITIES, _read_metadata
from kglobal_analysis.reduced import DISTRIBUTION_STORAGE, parse_reduced_filename, scan_reduced_products

LOG = '''
 vmax_e/c: 0.5
 vmax_i/c: 0.1
 # of bins in v: 200
 # of bins in x: 100
 # of bins in y: 100
 (xmax-xmin)*lx: 4
 (ymax-ymin)*ly: 2
 xmin*lx: 3
 ymin*ly: -2
 Max velocity is normalized to: 10
 The ROI is between 2 ellipsoids if the following is 1, or in separatrix if 2: 0
 0.2 <= x/lx <= 0.6
 0.1 <= y/ly <= 0.3
 0.25 <= z/lz <= 0.75
'''


def read_spec(case, spec, checkpoint='016'):
    if spec.position_axis is None:
        return case.parallel_perpendicular_velocity_distribution(spec.species, checkpoint=checkpoint)
    return case.position_parallel_velocity_distribution(spec.species, position_axis=spec.position_axis,
                                                        checkpoint=checkpoint)


class DistributionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.log = self.root / 'vd2dgyro.016'
        self.log.write_text(LOG)
        self.case = KGlobalCase(self.root)

    def write(self, spec=DISTRIBUTION_STORAGE[0], values=None):
        if values is None:
            i, j = np.indices(spec.raw_shape)
            values = i * i + 3 * j + 7 * i * j + 0.25
        path = self.root / (spec.storage_name + '.016')
        np.savetxt(path, values.ravel(order='F'))
        return values

    def test_all_six_mappings_orientation_and_active_bins(self):
        expected = [('vdeparperp', 'electron', None), ('vdiparperp', 'ion', None),
                    ('xpepar', 'electron', 'x'), ('xpipar', 'ion', 'x'),
                    ('ypepar', 'electron', 'y'), ('ypipar', 'ion', 'y')]
        for spec, mapping in zip(DISTRIBUTION_STORAGE, expected):
            with self.subTest(storage=spec.storage_name):
                original = self.write(spec)
                dist = read_spec(self.case, spec)
                self.assertEqual((dist.storage_name, dist.species, dist.position_axis), mapping)
                self.assertEqual(dist.category, 'regular')
                self.assertEqual(dist.shape, spec.raw_shape)
                np.testing.assert_array_equal(dist.values, original)
                self.assertFalse(np.array_equal(dist.values, original.ravel(order='F').reshape(spec.raw_shape)))
                self.assertNotEqual(dist.shape, original.T.shape)
                self.assertEqual(dist.values.size, 80601 if spec.position_axis is None else 40501)
                for index, limits in zip(dist.axis_indices, spec.index_ranges):
                    np.testing.assert_array_equal(index, np.arange(limits[0], limits[1]+1))
                self.assertGreater(dist.values[0, 0], 0)
                self.assertEqual(dist.value_sum, original.sum())

    def test_species_scales_nonunit_electron_fraction(self):
        for spec in DISTRIBUTION_STORAGE[:2]:
            self.write(spec)
            dist = read_spec(self.case, spec)
            vmax = 10 if spec.species == 'electron' else 2
            np.testing.assert_allclose(dist.axis_values[0], np.arange(-200, 201)*vmax/200)
            np.testing.assert_allclose(dist.axis_values[1], np.arange(201)*vmax/200)
            self.assertEqual(dist.axis_values[0][200], 0)
            self.assertEqual(dist.axis_values[1][0], 0)

    def test_position_centers_and_endpoints(self):
        for spec in DISTRIBUTION_STORAGE[2:]:
            self.write(spec)
            dist = read_spec(self.case, spec)
            origin, extent = (3, 4) if spec.position_axis == 'x' else (-2, 2)
            np.testing.assert_allclose(dist.axis_values[0], origin+np.arange(101)*extent/100)
            self.assertEqual(dist.axis_values[0][0], origin)
            self.assertEqual(dist.axis_values[0][-1], origin+extent)

    def test_region_provenance_and_no_geometry_inference(self):
        self.write()
        dist = read_spec(self.case, DISTRIBUTION_STORAGE[0])
        self.assertEqual(dist.legacy_roi_mode, 0)
        self.assertEqual(dist.region_kind, 'box')
        self.assertEqual(dict(dist.region_metadata), dict(XMIN=.2, XMAX=.6, YMIN=.1, YMAX=.3, ZMIN=.25, ZMAX=.75))
        self.assertNotIn('simulation_dimensionality', dist.provenance)
        self.assertNotIn('ellipse', dist.quantity)

    def test_optional_bounds_absent(self):
        self.log.write_text('\n'.join(line for line in LOG.splitlines() if '<=' not in line))
        self.write()
        self.assertEqual(dict(read_spec(self.case, DISTRIBUTION_STORAGE[0]).region_metadata), {})

    def test_finite_values_not_repaired(self):
        spec = DISTRIBUTION_STORAGE[2]
        for value in (0., -0.5, 2.):
            with self.subTest(value=value):
                original = np.full(spec.raw_shape, value)
                self.write(spec, original)
                dist = read_spec(self.case, spec)
                np.testing.assert_array_equal(dist.values, original)
                self.assertEqual(dist.value_sum, original.sum())
                self.assertEqual(dist.value_semantics, 'normalized_bin_mass')

    def test_wrong_counts_both_shapes(self):
        for spec in (DISTRIBUTION_STORAGE[0], DISTRIBUTION_STORAGE[2]):
            for delta in (-1, 1):
                with self.subTest(storage=spec.storage_name, delta=delta):
                    (self.root / (spec.storage_name+'.016')).write_text('0 '*(np.prod(spec.raw_shape)+delta))
                    with self.assertRaisesRegex(DistributionError, 'scalars'):
                        read_spec(self.case, spec)

    def test_bad_numeric_tokens(self):
        path = self.root / 'vdeparperp.016'
        for token in ('oops', 'NaN', 'Inf', '-inf', '1e999', '1_000', '1D'):
            with self.subTest(token=token):
                path.write_text(token+' '+'0 '*80600)
                with self.assertRaises(DistributionError):
                    read_spec(self.case, DISTRIBUTION_STORAGE[0])

    def test_fortran_exponents_data_and_metadata(self):
        (self.root / 'vdeparperp.016').write_text('1D-2 '+'2d-2 '*80600)
        self.log.write_text(LOG.replace('normalized to: 10', 'normalized to: 1D1'))
        dist = read_spec(self.case, DISTRIBUTION_STORAGE[0])
        self.assertEqual(dist.values[0, 0], .01)
        self.assertEqual(dist.values[1, 0], .02)
        self.assertEqual(dist.axis_values[0][-1], 10)

    def test_bad_metadata(self):
        replacements = [('vmax_e/c: 0.5', 'vmax_e/c: '+v) for v in ('0', '-1', 'nan', '1e999')]
        replacements += [('vmax_i/c: 0.1', 'vmax_i/c: -1'),
                         ('normalized to: 10', 'normalized to: 0'),
                         ('(xmax-xmin)*lx: 4', '(xmax-xmin)*lx: 0'),
                         ('(ymax-ymin)*ly: 2', '(ymax-ymin)*ly: -1'),
                         ('xmin*lx: 3', 'xmin*lx: Inf'),
                         ('0.25 <= z/lz <= 0.75', '0.8 <= z/lz <= 0.7'),
                         ('0.2 <= x/lx <= 0.6', '0.2 <= x/lx <= 0.2'),
                         ('0.1 <= y/ly <= 0.3', 'NaN <= y/ly <= 0.3'),
                         ('0.25 <= z/lz <= 0.75', 'broken z/lz bounds')]
        for before, after in replacements:
            with self.subTest(after=after):
                self.log.write_text(LOG.replace(before, after))
                with self.assertRaises(DistributionMetadataError):
                    _read_metadata(self.log)

    def test_unsupported_region_modes(self):
        self.write()
        for mode in ('1', '2', '3', '-1', '0.5', 'bad', 'NaN'):
            with self.subTest(mode=mode):
                self.log.write_text(LOG.replace('if 2: 0', 'if 2: '+mode))
                with self.assertRaises(DistributionMetadataError):
                    read_spec(self.case, DISTRIBUTION_STORAGE[0])

    def test_all_required_labels_and_duplicates(self):
        for line in LOG.splitlines():
            if ':' not in line:
                continue
            with self.subTest(line=line):
                self.log.write_text(LOG.replace(line, ''))
                with self.assertRaisesRegex(DistributionMetadataError, 'Missing'):
                    _read_metadata(self.log)
                self.log.write_text(LOG+'\n'+line)
                with self.assertRaisesRegex(DistributionMetadataError, 'Duplicate'):
                    _read_metadata(self.log)
        self.log.write_text(LOG+'\n vmax_e/c: 2')
        with self.assertRaisesRegex(DistributionMetadataError, 'Duplicate'):
            _read_metadata(self.log)

    def test_duplicate_bounds(self):
        self.log.write_text(LOG+'\n0 <= z/lz <= 1')
        with self.assertRaisesRegex(DistributionMetadataError, 'Duplicate'):
            _read_metadata(self.log)

    def test_bin_count_contract(self):
        for axis, old, new in (('v', 200, 201), ('x', 100, 99), ('y', 100, 100.5)):
            self.log.write_text(LOG.replace(f'in {axis}: {old}', f'in {axis}: {new}'))
            with self.assertRaisesRegex(DistributionMetadataError, 'Unsupported'):
                _read_metadata(self.log)

    def test_derived_coordinate_overflow_or_collapse(self):
        self.write(DISTRIBUTION_STORAGE[2])
        for old, new in [('xmin*lx: 3', 'xmin*lx: 1e300'),
                         ('normalized to: 10', 'normalized to: 5e-324'),
                         ('vmax_e/c: 0.5', 'vmax_e/c: 5e-324')]:
            self.log.write_text(LOG.replace(old, new))
            with self.assertRaises(DistributionMetadataError):
                read_spec(self.case, DISTRIBUTION_STORAGE[2])

    def test_reordered_metadata(self):
        self.log.write_text('\n'.join(reversed(LOG.splitlines())))
        self.write()
        self.assertEqual(read_spec(self.case, DISTRIBUTION_STORAGE[0]).axis_values[0][-1], 10)

    def test_xarray_contract_and_detachment(self):
        for spec in (DISTRIBUTION_STORAGE[0], DISTRIBUTION_STORAGE[3]):
            original = self.write(spec)
            dist = read_spec(self.case, spec)
            da = dist.to_xarray()
            self.assertEqual(da.dims, tuple(a+'_bin' for a in dist.axis_names))
            self.assertEqual(da.name, dist.quantity)
            for name, index, values in zip(dist.axis_names, dist.axis_indices, dist.axis_values):
                np.testing.assert_array_equal(da[name+'_bin'], index)
                np.testing.assert_array_equal(da[name], values)
                self.assertNotIn('units', da[name].attrs)
            for key in ('storage_name', 'species', 'category', 'checkpoint_suffix', 'region_kind', 'legacy_roi_mode', 'value_semantics'):
                self.assertEqual(da.attrs[key], getattr(dist, key))
            self.assertEqual(da.attrs['region_metadata']['ZMIN'], .25)
            self.assertEqual(da.attrs['serialization_order'], 'F')
            self.assertNotIn('time', da.coords)
            self.assertNotIn('units', da.attrs)
            self.assertNotIn('simulation_dimensionality', da.attrs)
            self.assertIn('not guaranteed a marginal', da.attrs['known_selection_limitation'])
            da.values[:] = -999
            da[dist.axis_names[0]].values[:] = 777
            da.attrs['region_metadata']['ZMIN'] = -100
            np.testing.assert_array_equal(dist.values, original)
            self.assertNotEqual(dist.axis_values[0][0], 777)
            self.assertEqual(dist.region_metadata['ZMIN'], .25)

    def test_authoritative_immutability(self):
        self.write()
        dist = read_spec(self.case, DISTRIBUTION_STORAGE[0])
        with self.assertRaises(FrozenInstanceError):
            dist.quantity = 'other'
        for array in (dist.values, *dist.axis_indices, *dist.axis_values):
            with self.assertRaises(ValueError):
                array.flat[0] = 5
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        with self.assertRaises(TypeError):
            dist.region_metadata['ZMIN'] = 4

    def test_semantic_rename_does_not_change_storage_or_region(self):
        self.write()
        before = read_spec(self.case, DISTRIBUTION_STORAGE[0])
        with patch.dict(DISTRIBUTION_QUANTITIES, parperp='provisional_new_name'):
            after = read_spec(self.case, DISTRIBUTION_STORAGE[0])
        self.assertEqual(after.quantity, 'provisional_new_name')
        self.assertEqual(after.storage_name, before.storage_name)
        self.assertEqual(after.region_metadata, before.region_metadata)
        np.testing.assert_array_equal(after.values, before.values)

    def test_invalid_api_arguments(self):
        for species in ('proton', 'Electron', '', None, 1):
            with self.assertRaises(ValueError):
                self.case.parallel_perpendicular_velocity_distribution(species, checkpoint='016')
        for axis in ('z', 'X', '', None):
            with self.assertRaises(ValueError):
                self.case.position_parallel_velocity_distribution('ion', position_axis=axis, checkpoint='016')
        for suffix in (16, '', '../016', '016.bak', '-1', ' 016', '１６'):
            with self.assertRaises(ValueError):
                self.case.parallel_perpendicular_velocity_distribution('electron', checkpoint=suffix)

    def test_missing_product_and_log(self):
        with self.assertRaises(FileNotFoundError):
            read_spec(self.case, DISTRIBUTION_STORAGE[0])
        self.write()
        self.log.unlink()
        with self.assertRaisesRegex(DistributionMetadataError, 'Missing reducer log'):
            read_spec(self.case, DISTRIBUTION_STORAGE[0])

    def test_exact_discovery_and_preserved_suffixes(self):
        for spec in DISTRIBUTION_STORAGE:
            for suffix in ('016', '16', '099'):
                name = spec.storage_name+'.'+suffix
                (self.root/name).touch()
                self.assertEqual(parse_reduced_filename(name), (spec.storage_name, suffix))
            for tail in ('016.bak', '016.txt', '-1', 'abc'):
                name = spec.storage_name+'.'+tail
                (self.root/name).touch()
                self.assertIsNone(parse_reduced_filename(name))
        for name in ('vdeparperpt.016', 'prefix.xpepar.016', 'p3d-001.016', 'movie.bx.016'):
            (self.root/name).touch()
            self.assertIsNone(parse_reduced_filename(name))
        (self.root/'xpepar.777').mkdir()
        self.assertNotIn(('xpepar', '777'), scan_reduced_products(self.root))
        self.log.rename(self.root/'vd2dgyro.16')
        (self.root/'vdeparperp.16').write_text('0 '*80601)
        dist = self.case.parallel_perpendicular_velocity_distribution('electron', checkpoint='16')
        self.assertEqual(dist.checkpoint_suffix, '16')
        self.assertEqual(self.case.suffixes, ())
        fresh = KGlobalCase(self.root)
        self.assertEqual(fresh.suffixes, ('016',))  # Only the movie filename.

    def test_only_selected_text_files_opened(self):
        xr.DataArray([0.])  # Initialize optional adapter imports before guarding data I/O.
        self.write()
        for name in ('p3d-001.016', 'movie.bx.016', 'movie.log.016', 'p3d.stdout.016', 'xpipar.016'):
            (self.root/name).touch()
        allowed = {self.log, self.root/'vdeparperp.016'}
        seen = []
        original = Path.open
        def guard(path, *args, **kwargs):
            self.assertIn(path, allowed)
            self.assertEqual(args[0] if args else kwargs.get('mode', 'r'), 'r')
            seen.append(path)
            return original(path, *args, **kwargs)
        # Both package readers use Path.open (read_text delegates to it).
        # Python 3.10 caches io.open in pathlib's accessor, bypassing io spies.
        with patch.object(Path, 'open', new=guard), \
             patch('numpy.fromfile', side_effect=AssertionError('binary read')), \
             patch('numpy.memmap', side_effect=AssertionError('binary read')), \
             patch.object(KGlobalCase, 'read_movie_frame', side_effect=AssertionError('movie read')):
            dist = read_spec(self.case, DISTRIBUTION_STORAGE[0])
            dist.to_xarray()
        self.assertCountEqual(seen, allowed)


if __name__ == '__main__':
    unittest.main()
