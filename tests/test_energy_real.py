"""Read-only historical reduced products; not a golden reducer validation."""
from pathlib import Path
import math
import os
import unittest
from unittest.mock import patch

from kglobal_analysis import KGlobalCase

FIXTURE = Path(os.environ.get('KGLOBAL_ENERGY_VALIDATION_CASE',
    Path(__file__).resolve().parents[2]/'validation-data/hcs_large_energy016'))


@unittest.skipUnless(FIXTURE.is_dir(), 'historical reduced energy fixture unavailable')
class RealEnergyTests(unittest.TestCase):
    def check_species(self, species, storage, high, integral):
        original = Path.open
        def guard(path,*args,**kwargs):
            self.assertIn(path.name, [f'{storage}.016','vd2dgyro.016'])
            self.assertNotIn('w', args[0] if args else kwargs.get('mode','r'))
            return original(path,*args,**kwargs)
        with patch.object(Path,'open',new=guard), \
                patch('numpy.fromfile',side_effect=AssertionError('binary read')):
            e = KGlobalCase(FIXTURE).energy_spectrum(species, checkpoint='016')
            self.assertEqual(e.storage_name, storage)
            self.assertEqual(len(e.storage_values),201)
            self.assertEqual(len(e.values),200)
            self.assertEqual(e.sentinel_value,0.)
            self.assertTrue(all(math.isfinite(x) for x in e.storage_values))
            self.assertAlmostEqual(e.Emin,9.99999975e-5,places=14)
            self.assertAlmostEqual(e.Emax,high,places=7)
            self.assertEqual(e.edges[-1],e.Emax)
            self.assertAlmostEqual(e.width_integral,integral,delta=2e-7)
            self.assertEqual(e.to_xarray().shape,(200,))
            self.assertNotIn('time',e.to_xarray().coords)
            self.assertNotIn('Problem:', (FIXTURE/'vd2dgyro.016').read_text())

    def test_electron(self):
        self.check_species('electron','xenergylog',225.897324,.9999991)

    def test_ion(self):
        self.check_species('ion','xenergylogi',5647.43359,1.00000036)
