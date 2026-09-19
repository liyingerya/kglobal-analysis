"""Metadata-only validation, including partial but internally valid coverage."""
from dataclasses import FrozenInstanceError
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

from kglobal_analysis import KGlobalCase, STANDARD_MOVIE_FORMAT
from kglobal_analysis.segment import MovieSegment

PARAM = '''#define nx 3
#define ny 2
#define nz 1
#define pex 1
#define pey 1
#define pez 1
#define movie_header "movie_kglobal3.0.h"
#define double_byte
#define dt .025
#define n_movieout 2
'''


def metadata_report(test, root):
    original = Path.open
    def guard(path, *args, **kwargs):
        test.assertTrue(path.name.startswith(('param', 'movie.log.', 'p3d.stdout.')), path)
        test.assertNotIn('b', args[0] if args else kwargs.get('mode', 'r'))
        return original(path, *args, **kwargs)
    with patch.object(Path, 'open', new=guard), \
            patch('kglobal_analysis.movie.np.fromfile', side_effect=AssertionError('sample read')), \
            patch.object(KGlobalCase, 'read_movie_frame', side_effect=AssertionError('frame read')), \
            patch.object(MovieSegment, 'read_frame', side_effect=AssertionError('frame read')):
        return KGlobalCase(root).validate_movie_case()


class ManifestTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root/'param').write_text(PARAM)
        self.segment('090', [1., 1.05], ('bx', 'jihpar'))
        self.segment('002', [1.10, 1.15])

    def segment(self, suffix, times, variables=('bx',), samples=6):
        for name in variables:
            (self.root/f'movie.{name}.{suffix}').write_bytes(b'\0'*(samples*2*len(times)))
        (self.root/f'movie.log.{suffix}').write_text('0 1\n'*len(STANDARD_MOVIE_FORMAT.variables)*len(times))
        (self.root/f'p3d.stdout.{suffix}').write_text('two-byte movie output\n'+''.join(
            f'movie output, t= {time}\n' for time in times))

    def report(self):
        return metadata_report(self, self.root)

    def assert_issue(self, code):
        report = self.report()
        self.assertFalse(report.ok)
        self.assertIn(code, [issue.code for issue in report.errors])
        json.dumps(report.to_dict(), allow_nan=False)
        return report

    def test_valid_narrow_coverage_physical_order_and_no_samples(self):
        report = self.report()
        self.assertTrue(report.ok, report.errors)
        variables = {v.storage_name: v for v in report.variables}
        self.assertEqual(variables['bx'].suffixes, ('090', '002'))
        self.assertEqual(variables['bx'].frame_counts, (('090', 2), ('002', 2)))
        self.assertEqual(variables['bx'].total_frame_count, 4)
        self.assertEqual((variables['bx'].first_time, variables['bx'].last_time), (1., 1.15))
        self.assertEqual(variables['jihpar'].suffixes, ('090',))
        self.assertEqual(variables['jihpar'].total_frame_count, 2)
        self.assertEqual({w.code for w in report.warnings}, {'absent_variables', 'unequal_coverage'})
        self.assertEqual(report.parameters.frame_bytes, 12)
        for segment in report.segments:
            self.assertEqual(segment.stdout_event_count, 2)
            self.assertEqual(segment.log_entry_count, 36)
            self.assertEqual(segment.log_frame_count, 2)
            self.assertTrue(segment.stdout_valid and segment.log_valid)

    def test_serialization_is_detached_and_report_immutable(self):
        report = self.report()
        result = json.loads(json.dumps(report.to_dict(), allow_nan=False))
        self.assertTrue(result['ok'])
        result['parameters']['Nx'] = 99
        self.assertEqual(report.parameters.Nx, 3)
        with self.assertRaises(FrozenInstanceError):
            report.directory = 'changed'
        self.assertIn('Movie case OK', str(report))

    def test_partial_frame(self):
        (self.root/'movie.bx.002').write_bytes(b'\0'*23)
        report = self.assert_issue('invalid_binary_size')
        self.assertIsNone(next(v for v in report.variables if v.storage_name == 'bx').total_frame_count)

    def test_binary_stdout_mismatch(self):
        (self.root/'movie.bx.002').write_bytes(b'\0'*12)
        self.assert_issue('binary_stdout_count')

    def test_segment_binary_counts_disagree(self):
        (self.root/'movie.jihpar.090').write_bytes(b'\0'*12)
        self.assert_issue('segment_frame_counts')

    def test_invalid_stdout(self):
        for tokens in [('bad', '1.15'), ('nan', '1.15'), ('inf', '1.15'),
                       ('1.15', '1.1'), ('1.1', '1.1'), ('1.1', '1.2')]:
            with self.subTest(tokens=tokens):
                (self.root/'p3d.stdout.002').write_text(''.join(f'movie output, t= {t}\n' for t in tokens))
                self.assert_issue('invalid_stdout')

    def test_no_actual_stdout_events(self):
        (self.root/'p3d.stdout.002').write_text('two-byte movie output\n')
        self.assert_issue('invalid_stdout')

    def test_log_count(self):
        for count, code in [(35, 'invalid_log'), (54, 'log_frame_count'), (0, 'invalid_log')]:
            with self.subTest(count=count):
                (self.root/'movie.log.002').write_text('0 1\n'*count)
                self.assert_issue(code)

    def test_invalid_log_ranges(self):
        for line in ['2 1', 'nan 1', '0 inf', 'bad', '0 1 2']:
            with self.subTest(line=line):
                (self.root/'movie.log.002').write_text(line+'\n'+'0 1\n'*35)
                self.assert_issue('invalid_log')

    def test_fortran_log_exponents(self):
        (self.root/'movie.log.002').write_text('-1.0D+00 1.0D+00\n'*36)
        self.assertTrue(self.report().ok)

    def test_boundary_errors(self):
        for times in [[1.05, 1.10], [1.025, 1.075], [1.2, 1.25], [1.075, 1.125]]:
            with self.subTest(times=times):
                self.segment('002', times)
                self.assert_issue('variable_timeline')

    def test_missing_companions_collects_errors(self):
        (self.root/'p3d.stdout.002').unlink()
        (self.root/'movie.log.090').unlink()
        report = self.assert_issue('missing_stdout')
        self.assertIn('missing_log', [error.code for error in report.errors])

    def test_companion_only_segments(self):
        self.segment('700', [8., 8.05], ())
        report = self.report()
        self.assertTrue(report.ok, report.errors)
        segment = next(s for s in report.segments if s.suffix == '700')
        self.assertIsNone(segment.frame_count)
        self.assertEqual(segment.log_frame_count, 2)
        self.assertIn('companion_only_segment', [w.code for w in report.warnings])

    def test_unsupported_format_encoding(self):
        for parameters in [PARAM.replace('movie_kglobal3.0.h', 'unknown.h'),
                           PARAM.replace('#define double_byte', '#define four_byte'),
                           PARAM+'#define four_byte\n', PARAM.replace('#define double_byte', '')]:
            with self.subTest(parameters=parameters):
                (self.root/'param').write_text(parameters)
                report = self.assert_issue('unsupported_format')
                self.assertFalse(report.parameters.format_supported)

    def test_unknown_variable(self):
        (self.root/'movie.unknown.002').write_bytes(b'\0'*24)
        self.assert_issue('unsupported_variable')

    def test_missing_dimensions(self):
        (self.root/'param').write_text(PARAM.replace('#define pex 1', ''))
        self.assert_issue('invalid_dimensions')

    def test_volume_layout_metadata(self):
        (self.root/'param').write_text(PARAM.replace('#define nz 1', '#define nz 2'))
        self.segment('090', [1., 1.05], ('bx', 'jihpar'), samples=12)
        self.segment('002', [1.10, 1.15], samples=12)
        report = self.report()
        self.assertTrue(report.ok, report.errors)
        self.assertEqual(report.parameters.frame_bytes, 24)
        self.assertEqual(report.parameters.Nz, 2)
