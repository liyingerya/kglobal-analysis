"""Historical compatibility, not particle-level numerical certification."""
import builtins
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr

from kglobal_analysis import KGlobalCase
from kglobal_analysis.reduced import DISTRIBUTION_STORAGE
from test_distribution import read_spec

FIXTURE = Path(__file__).resolve().parents[2] / 'validation-data/hcs_large_distribution016'
AVAILABLE = all((FIXTURE / (name+'.016')).is_file()
                for name in ('vd2dgyro', *(s.storage_name for s in DISTRIBUTION_STORAGE)))


@unittest.skipUnless(AVAILABLE, 'External historical distribution fixture unavailable')
class HistoricalDistributionTests(unittest.TestCase):
    def test_all_six_files_guarded_values_axes_and_provenance(self):
        xr.DataArray([0.])  # Initialize adapter imports outside the data-I/O guard.
        case = KGlobalCase(FIXTURE)
        expected_bounds = dict(XMIN=.549, XMAX=.610, YMIN=.732, YMAX=.7498, ZMIN=0., ZMAX=1.)
        for spec in DISTRIBUTION_STORAGE:
            with self.subTest(storage=spec.storage_name):
                allowed = {FIXTURE / (spec.storage_name+'.016'), FIXTURE/'vd2dgyro.016'}
                seen = []
                original = io.open
                def guarded(file, *args, **kwargs):
                    self.assertIn(Path(file), allowed)
                    seen.append(Path(file))
                    return original(file, *args, **kwargs)
                with patch('io.open', guarded), patch('builtins.open', guarded), \
                     patch('numpy.fromfile', side_effect=AssertionError('binary read')), \
                     patch('numpy.memmap', side_effect=AssertionError('binary read')):
                    dist = read_spec(case, spec)
                    da = dist.to_xarray()
                self.assertCountEqual(seen, allowed)
                self.assertEqual(dist.storage_name, spec.storage_name)
                self.assertEqual(dist.species, spec.species)
                self.assertEqual(dist.position_axis, spec.position_axis)
                self.assertEqual(dist.shape, spec.raw_shape)
                self.assertEqual(dist.values.size, 80601 if spec.position_axis is None else 40501)
                self.assertTrue(np.isfinite(dist.values).all())
                self.assertEqual(dist.values.min(), 0)
                self.assertAlmostEqual(dist.value_sum, 1, places=12)
                self.assertEqual(dist.legacy_roi_mode, 0)
                self.assertEqual(dist.region_kind, 'box')
                for key, value in expected_bounds.items():
                    self.assertAlmostEqual(dist.region_metadata[key], value)
                vmax = 30.679616928100586 if spec.species == 'electron' else 3.0679616928100586
                parallel = dist.axis_names.index('v_parallel')
                self.assertEqual(dist.axis_indices[parallel][200], 0)
                self.assertEqual(dist.axis_values[parallel][200], 0)
                self.assertAlmostEqual(dist.axis_values[parallel][-1], vmax)
                if spec.position_axis is None:
                    self.assertEqual(dist.axis_indices[1][0], 0)
                    self.assertEqual(dist.axis_values[1][0], 0)
                    self.assertAlmostEqual(dist.axis_values[1][-1], vmax)
                else:
                    origin, extent = ((3.4494688296318059, .38327431440353360)
                                      if spec.position_axis == 'x' else
                                      (2.2996457118988038, .055920346546173214))
                    self.assertEqual(dist.axis_indices[0][0], 0)
                    self.assertAlmostEqual(dist.axis_values[0][0], origin)
                    self.assertAlmostEqual(dist.axis_values[0][-1], origin+extent)
                self.assertEqual(da.shape, dist.shape)
                self.assertNotIn('time', da.coords)
                self.assertNotIn('simulation_dimensionality', da.attrs)


if __name__ == '__main__':
    unittest.main()
