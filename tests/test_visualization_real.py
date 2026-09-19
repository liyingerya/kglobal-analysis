"""Real reader/rendering compatibility, not publication reproduction."""
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from kglobal_analysis import KGlobalCase
from kglobal_analysis.analysis import spacetime
from kglobal_analysis.segment import MovieSegment
from kglobal_analysis.plotting import plot_scalar_map, plot_energy_spectrum, plot_distribution
from test_visualization import HAS_PLOT

ROOT = Path(__file__).resolve().parents[2] / 'validation-data'


@unittest.skipUnless(HAS_PLOT, 'Install kglobal-analysis[plot]')
class RealVisualizationTests(unittest.TestCase):
    def setUp(self):
        from matplotlib import pyplot as plt
        self.addCleanup(plt.close, 'all')

    @unittest.skipUnless((ROOT/'hcs_large_005/movie.bx.005').is_file(), 'External Bx fixture absent')
    def test_bx_index_map_and_two_event_spacetime(self):
        series = KGlobalCase(ROOT/'hcs_large_005').movie('bx', byteorder='little')
        frame = series.read_frame_xarray(0)
        self.assertEqual(frame.shape, (8192, 4096))
        # Explicit display subsampling keeps rendering validation modest. Label
        # retained original indices; this is not a region-efficient file read.
        sample = frame.isel(x=slice(None, None, 64), y=slice(None, None, 64)).copy(deep=True)
        sample = sample.assign_coords(x=np.arange(0, 8192, 64), y=np.arange(0, 4096, 64))
        del frame
        result = plot_scalar_map(sample, horizontal='x', vertical='y',
                                 xlabel='x index', ylabel='y index')
        result.figure.canvas.draw()
        self.assertEqual(result.axes.get_xlabel(), 'x index')
        self.assertEqual(result.metadata['storage_name'], 'bx')
        original = MovieSegment.read_frame
        seen = []
        def counted(segment, index):
            seen.append(index)
            return original(segment, index)
        with patch.object(MovieSegment, 'read_frame', counted):
            st = spacetime(series, line_axis='x', fixed_indices={'y': 0},
                           times=[series.times[0], series.times[-1]])
        self.assertEqual(seen, [0, 19])
        self.assertEqual(st.shape, (2, 8192))
        np.testing.assert_array_equal(st.time, [series.times[0], series.times[-1]])

    @unittest.skipUnless((ROOT/'hcs_large_energy016/xenergylog.016').is_file(), 'External energy fixture absent')
    def test_real_energy_estimator(self):
        case = KGlobalCase(ROOT/'hcs_large_energy016')
        for species in ('electron', 'ion'):
            spectrum = case.energy_spectrum(species, checkpoint='016')
            before = spectrum.storage_values
            result = plot_energy_spectrum(spectrum, xscale='log', yscale='log')
            result.figure.canvas.draw()
            self.assertEqual(result.axes.get_ylabel(), 'legacy energy estimator')
            self.assertEqual(spectrum.storage_values, before)

    @unittest.skipUnless((ROOT/'hcs_large_distribution016/vdeparperp.016').is_file(), 'External distribution fixture absent')
    def test_all_six_real_distributions(self):
        from matplotlib import pyplot as plt
        case = KGlobalCase(ROOT/'hcs_large_distribution016')
        for species in ('electron', 'ion'):
            for axis in (None, 'x', 'y'):
                dist = (case.parallel_perpendicular_velocity_distribution(species, checkpoint='016')
                        if axis is None else case.position_parallel_velocity_distribution(
                            species, position_axis=axis, checkpoint='016'))
                result = plot_distribution(dist, coordinate_mode='centers', normalization='log')
                result.figure.canvas.draw()
                self.assertEqual(result.colorbar.ax.get_ylabel(), 'normalized bin mass')
                self.assertEqual(result.metadata['region_metadata']['ZMAX'], 1)
                self.assertEqual(dist.value_sum, 1)
                plt.close(result.figure)
