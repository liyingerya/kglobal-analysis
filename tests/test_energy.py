"""Reduced text storage, metadata, semantic mapping and faithful coordinates."""
from dataclasses import FrozenInstanceError
from pathlib import Path
from tempfile import TemporaryDirectory
import math
import unittest
from unittest.mock import patch

import numpy as np

from kglobal_analysis import KGlobalCase, EnergyMetadataError, EnergySpectrumError
from kglobal_analysis.reduced import parse_reduced_filename, scan_reduced_products

LOG = 'unrelated reducer stdout\n Emin: 1.0D-4\n Emax: 1.0D2\n Imin: 2.0d-4\n Imax: 2.0d2\n'


class EnergyTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.write_values('xenergylog', [0.] + [float(i) for i in range(1, 201)])
        self.write_values('xenergylogi', [0.] + [2.] * 200)
        (self.root/'vd2dgyro.016').write_text(LOG)
        self.case = KGlobalCase(self.root)

    def write_values(self, name, values, suffix='016'):
        (self.root/f'{name}.{suffix}').write_text('\n'.join(map(str, values))+'\n')

    def read(self, species='electron', suffix='016'):
        return self.case.energy_spectrum(species, checkpoint=suffix)

    def test_electron_identity_and_storage(self):
        e = self.read()
        self.assertEqual((e.quantity, e.species, e.storage_name), ('energy_spectrum', 'electron', 'xenergylog'))
        self.assertEqual(e.source_path, self.root/'xenergylog.016')
        self.assertEqual(e.log_path, self.root/'vd2dgyro.016')
        self.assertEqual(e.checkpoint_suffix, '016')
        self.assertEqual(e.storage_values, (0.,) + tuple(float(i) for i in range(1,201)))
        self.assertEqual((len(e.storage_values), len(e.values), e.bin_count), (201,200,200))
        self.assertEqual((e.Emin, e.Emax), (1e-4, 100.))

    def test_ion_mapping(self):
        e = self.read('ion')
        self.assertEqual(e.storage_name, 'xenergylogi')
        self.assertEqual(e.species, 'ion')
        self.assertEqual(e.values, (2.,)*200)
        self.assertEqual((e.Emin, e.Emax), (2e-4, 200.))

    def test_exact_filename_discovery(self):
        for name in ['xenergylog.0009','xenergylogi.5','vd2dgyro.099']:
            self.assertIsNotNone(parse_reduced_filename(name))
        for name in ['xenergylogt.016','prefixxenergylog.016','xenergylog.016.bak',
                     'xenergylog.016\n','xenergylog.-1','xenergylog.foo','vd2dgyro.016.extra',
                     'movie.xenergylog.016','p3d-001.016']:
            with self.subTest(name=name):
                self.assertIsNone(parse_reduced_filename(name))
        (self.root/'xenergylog.099').mkdir()
        self.assertNotIn(('xenergylog','099'), scan_reduced_products(self.root))
        self.assertIn(('vd2dgyro','016'), scan_reduced_products(self.root))

    def test_noncontinuous_suffixes_preserve_spelling(self):
        for suffix in ['0002', '2', '901']:
            self.write_values('xenergylog', [0.]*201, suffix)
        self.assertEqual(self.case.energy_spectrum_suffixes('electron'), ('0002','2','016','901'))
        self.assertEqual(self.case.energy_spectrum_suffixes('ion'), ('016',))
        self.assertEqual(self.case.suffixes, ())
        self.assertEqual(self.case.variables, ())

    def test_caller_suffix_is_not_coerced(self):
        self.write_values('xenergylog', [0.]*201, '00016')
        (self.root/'vd2dgyro.00016').write_text(LOG)
        self.assertEqual(self.read(suffix='00016').checkpoint_suffix, '00016')
        with self.assertRaises(FileNotFoundError):
            self.read(suffix='16')

    def test_invalid_suffixes(self):
        for suffix in [16, None, '', '../016', '016.bak', '-1', '016\n']:
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                self.read(suffix=suffix)

    def test_species_rejected(self):
        for species in ['proton','Electron','xenergylog','',None,[]]:
            with self.subTest(species=species), self.assertRaises(ValueError):
                self.read(species)
        with self.assertRaises(ValueError):
            self.case.energy_spectrum_suffixes('proton')

    def test_edges_use_200_not_201(self):
        e = self.read()
        expected = e.Emin * (e.Emax/e.Emin)**(np.arange(201)/200)
        np.testing.assert_allclose(e.edges, expected, rtol=3e-14)
        self.assertEqual(len(e.edges), 201)
        self.assertEqual(e.edges[0], e.Emin)
        self.assertEqual(e.edges[-1], e.Emax)
        self.assertAlmostEqual(e.edges[100], math.sqrt(e.Emin*e.Emax))
        self.assertNotAlmostEqual(e.edges[-1], e.Emin*(e.Emax/e.Emin)**(200/201))

    def test_centers_widths_and_integral(self):
        e = self.read()
        for low, high, center, width in zip(e.lower_edges,e.upper_edges,e.centers,e.widths):
            self.assertLess(low, center)
            self.assertLess(center, high)
            self.assertGreater(width, 0)
            self.assertAlmostEqual(center, math.sqrt(low*high))
            self.assertEqual(width, high-low)
        self.assertAlmostEqual(e.width_integral, float(np.dot(e.values, e.widths)))

    def test_nonzero_sentinel_no_renormalization(self):
        self.write_values('xenergylog', [7.] + [2.]*200)
        e = self.read()
        self.assertEqual(e.sentinel_value, 7.)
        self.assertEqual(e.values, (2.,)*200)
        self.assertAlmostEqual(e.width_integral, 2*(e.Emax-e.Emin))
        self.assertNotAlmostEqual(e.width_integral, 1.)
        self.assertEqual(e.to_xarray().attrs['sentinel_value'], 7.)

    def test_finite_negative_values_preserved(self):
        self.write_values('xenergylog', [-9.]+[-1.]*200)
        e = self.read()
        self.assertEqual(e.sentinel_value, -9.)
        self.assertEqual(e.values, (-1.,)*200)
        self.assertLess(e.width_integral, 0)

    def test_immutable_and_xarray_detached(self):
        e = self.read()
        with self.assertRaises(FrozenInstanceError):
            e.Emin = 10
        with self.assertRaises(TypeError):
            e.storage_values[0] = 9
        da = e.to_xarray()
        da.values[0] = 99
        da.coords['energy_lower'].values[0] = 99
        self.assertEqual(e.values[0], 1.)
        self.assertEqual(e.lower_edges[0], 1e-4)

    def test_xarray_metadata_no_time_or_invented_units(self):
        e = self.read()
        da = e.to_xarray()
        self.assertEqual(da.dims, ('energy_bin',))
        self.assertEqual(da.shape, (200,))
        self.assertEqual(da.name, e.quantity)
        np.testing.assert_array_equal(da.energy_bin, np.arange(1,201))
        np.testing.assert_array_equal(da.values, e.values)
        for coord, values in [('energy_lower',e.lower_edges),('energy_upper',e.upper_edges),
                              ('energy_center',e.centers),('energy_width',e.widths)]:
            self.assertEqual(da[coord].dims, ('energy_bin',))
            np.testing.assert_array_equal(da[coord], values)
        self.assertNotIn('time', da.coords)
        self.assertNotIn('units', da.attrs)
        self.assertEqual(da.attrs['storage_name'], 'xenergylog')
        self.assertEqual(da.attrs['checkpoint_suffix'], '016')
        self.assertEqual(da.attrs['storage_entries'], 201)
        self.assertEqual(da.attrs['active_bins'], 200)
        self.assertIn('not raw counts', da.attrs['estimator'])
        self.assertEqual(da.attrs['source_path'], str(e.source_path))
        self.assertEqual(da.attrs['log_path'], str(e.log_path))

    def test_semantic_rename_does_not_change_storage_parser(self):
        before = scan_reduced_products(self.root)
        with patch('kglobal_analysis.energy.ENERGY_QUANTITY', 'energy_distribution'):
            e = self.read()
        self.assertEqual(e.quantity, 'energy_distribution')
        self.assertEqual(e.to_xarray().name, 'energy_distribution')
        self.assertEqual(e.storage_name, 'xenergylog')
        self.assertEqual(e.values, self.read().values)
        self.assertEqual(before, scan_reduced_products(self.root))

    def test_wrong_lengths(self):
        for count in [0,200,202]:
            with self.subTest(count=count):
                self.write_values('xenergylog', [0.]*count)
                with self.assertRaisesRegex(EnergySpectrumError, '201'):
                    self.read()

    def test_malformed_or_nonfinite_storage(self):
        for token in ['bad','NaN','Inf','-Infinity','1e999','1_0']:
            with self.subTest(token=token):
                self.write_values('xenergylog', [token]+[0.]*200)
                with self.assertRaises(EnergySpectrumError):
                    self.read()

    def test_fortran_storage_exponents(self):
        self.write_values('xenergylog', ['0D0']+['2.5d-1']*200)
        self.assertEqual(self.read().values, (.25,)*200)

    def test_missing_spectrum(self):
        (self.root/'xenergylog.016').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'xenergylog.016'):
            self.read()

    def test_missing_log(self):
        (self.root/'vd2dgyro.016').unlink()
        with self.assertRaisesRegex(EnergyMetadataError, 'vd2dgyro.016'):
            self.read()

    def test_missing_malformed_duplicate_limits(self):
        for text in ['Emin: 1e-4\n', 'Emax: 100\n', 'Emin: bad\nEmax: 100\n',
                     'Emin: 1e-4\nEmax: 100 extra\n', LOG+'Emin: 1e-4\n',
                     'prefix Emin: 1e-4\nEmax: 100\n']:
            with self.subTest(text=text):
                (self.root/'vd2dgyro.016').write_text(text)
                with self.assertRaises(EnergyMetadataError):
                    self.read()

    def test_invalid_limit_ranges(self):
        for low,high in [('nan','10'),('1','inf'),('1','1e999'),('0','1'),('-1','10'),('1','1'),('2','1')]:
            with self.subTest(low=low, high=high):
                (self.root/'vd2dgyro.016').write_text(f'Emin: {low}\nEmax: {high}\n')
                with self.assertRaises(EnergyMetadataError):
                    self.read()

    def test_selected_species_metadata_only(self):
        (self.root/'vd2dgyro.016').write_text('Emin: 1e-4\nEmax: 100\nImin: broken\n')
        self.assertEqual(self.read().Emax,100.)
        with self.assertRaises(EnergyMetadataError):
            self.read('ion')

    def test_extreme_limits_avoid_ratio_overflow(self):
        (self.root/'vd2dgyro.016').write_text('Emin: 1e-200\nEmax: 1e200\n')
        e = self.read()
        self.assertTrue(all(math.isfinite(x) and x>0 for x in e.edges+e.centers+e.widths))
        self.assertEqual(e.edges[-1], 1e200)

    def test_no_checkpoint_or_movie_reads(self):
        (self.root/'p3d-001.016').write_bytes(b'not a checkpoint')
        (self.root/'movie.bx.016').write_bytes(b'not a movie')
        original = Path.open
        opened = []
        def guard(path, *args, **kwargs):
            self.assertIn(path.name, ['xenergylog.016','xenergylogi.016','vd2dgyro.016'])
            opened.append(path.name)
            return original(path,*args,**kwargs)
        with patch.object(Path,'open',new=guard), \
                patch('numpy.fromfile',side_effect=AssertionError('binary sample read')):
            case = KGlobalCase(self.root)
            self.assertEqual(case.energy_spectrum_suffixes('electron'), ('016',))
            e = case.energy_spectrum('electron', checkpoint='016')
            e.to_xarray()
        self.assertEqual(opened, ['xenergylog.016','vd2dgyro.016'])
