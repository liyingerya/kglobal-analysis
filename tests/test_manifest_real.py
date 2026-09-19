"""Real metadata validation; binary sample access is forbidden throughout."""
from pathlib import Path
from tempfile import TemporaryDirectory
import os
import unittest

from test_manifest import metadata_report

DATA = Path(__file__).resolve().parents[2]/'validation-data'
FIRST = Path(os.environ.get('KGLOBAL_VALIDATION_CASE', DATA/'hcs_large_005'))
MULTI = Path(os.environ.get('KGLOBAL_MULTI_VALIDATION_CASE', DATA/'hcs_large_multi'))


@unittest.skipUnless(FIRST.is_dir() and MULTI.is_dir(), 'real fixtures unavailable')
class RealManifestTests(unittest.TestCase):
    def test_partial_real_case_without_sample_reads(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            (root/'param_hcs_large').symlink_to((FIRST/'param_hcs_large').resolve(strict=True))
            for suffix in ('004', '005', '006'):
                source = FIRST if suffix == '005' else MULTI
                for stem in ('movie.bx', 'movie.log', 'p3d.stdout'):
                    name = f'{stem}.{suffix}'
                    (root/name).symlink_to((source/name).resolve(strict=True))
            (root/'movie.jihpar.004').symlink_to((MULTI/'movie.jihpar.004').resolve(strict=True))
            report = metadata_report(self, root)
            self.assertTrue(report.ok, report.errors)
            p = report.parameters
            self.assertEqual((p.Nx, p.Ny, p.Nz), (8192, 4096, 1))
            self.assertEqual(p.movie_header, 'movie_kglobal3.0.h')
            self.assertEqual(p.encoding, 'double_byte')
            self.assertAlmostEqual(p.nominal_cadence, .05)
            variables = {v.storage_name: v for v in report.variables}
            self.assertEqual(variables['bx'].suffixes, ('004', '005', '006'))
            self.assertEqual(variables['jihpar'].suffixes, ('004',))
            for name, frames, last in [('bx', 60, 7.), ('jihpar', 20, 5.)]:
                self.assertEqual(variables[name].total_frame_count, frames)
                self.assertAlmostEqual(variables[name].first_time, 4.05, places=6)
                self.assertAlmostEqual(variables[name].last_time, last, places=6)
            for segment in report.segments:
                self.assertEqual(segment.frame_count, 20)
                self.assertEqual(segment.stdout_event_count, 20)
                self.assertEqual(segment.log_entry_count, 360)
                self.assertEqual(segment.log_frame_count, 20)
                for file in segment.files:
                    self.assertEqual(file.size_bytes, 1342177280)
            self.assertIn('unequal_coverage', [w.code for w in report.warnings])
