"""Schema, volume layout and generic timelines, without real simulation data."""
from pathlib import Path
from tempfile import TemporaryDirectory
import struct
import unittest
from unittest.mock import patch

import numpy as np

from kglobal_analysis import (KGlobalCase, MovieSegment, MovieSeries, VolumeLayout,
                              STANDARD_MOVIE_FORMAT, TimeMetadataError)

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
# Independent expected schema, not derived from the implementation.
NAMES = ('ni', 'jix', 'jiy', 'jiz', 'bx', 'by', 'bz', 'pi', 'neh',
         'pehpar', 'pehperp', 'epar', 'pc', 'jhpar', 'nih', 'pihpar', 'pihperp', 'jihpar')


class MovieTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'param').write_text(PARAM)
        for j, name in enumerate(NAMES):
            (self.root / f'movie.{name}.090').write_bytes(struct.pack('<12h', *range(j*100, j*100+12)))
        self.write_log('090')
        self.write_times('090', [1., 1.05])

    def write_log(self, suffix):
        (self.root / f'movie.log.{suffix}').write_text(''.join(
            f'{f*100+j*2} {f*100+j*2+1}\n' for f in range(2) for j in range(18)))

    def write_times(self, suffix, times):
        (self.root / f'p3d.stdout.{suffix}').write_text('two-byte movie output\n'+''.join(
            f'movie output, t= {t}\n' for t in times))

    def test_selected_format_controls_log_stride_and_variable_index(self):
        from kglobal_analysis import MovieFormat, VariableSpec
        # Test-only selector substitution, not an additional supported format.
        schema = MovieFormat("test-only", (VariableSpec("jihpar", 0), VariableSpec("bx", 1)))
        (self.root/'movie.log.090').write_text("10 10\n20 20\n30 30\n40 40\n")
        case = KGlobalCase(self.root)
        with patch('kglobal_analysis.movie.movie_format', return_value=schema) as select:
            for variable, expected in (("jihpar", 30.), ("bx", 40.)):
                select.reset_mock()
                values = case.read_movie_frame(variable, '090', 1, byteorder='little')
                select.assert_called_once_with(case.parameters)
                np.testing.assert_array_equal(values, np.full((3, 2), expected))
            # Count validation also uses the selected schema, not 18 entries.
            with (self.root/'movie.log.090').open('a') as stream:
                stream.write("50 50\n")
            with self.assertRaisesRegex(ValueError, "more than 4 entries"):
                case.read_movie_frame('bx', '090', 1, byteorder='little')

    def test_standalone_log_api_retains_standard_format(self):
        from kglobal_analysis.movie import read_movie_minmax
        self.assertEqual(read_movie_minmax(self.root/'movie.log.090', 'jihpar', 1, 2),
                         (134., 135.))

    def test_all_schema_mappings(self):
        self.assertEqual(len(STANDARD_MOVIE_FORMAT.variables), 18)
        for j, name in enumerate(NAMES):
            spec = STANDARD_MOVIE_FORMAT.variable(name)
            self.assertEqual((spec.storage_name, spec.log_index), (name, j))

    def test_every_variable_selects_its_file_and_log_entry(self):
        case = KGlobalCase(self.root)
        original = np.fromfile
        for j, name in enumerate(NAMES):
            for f in range(2):
                with self.subTest(variable=name, frame=f):
                    def bounded(stream, *, dtype, count):
                        self.assertEqual(Path(stream.name).name, f'movie.{name}.090')
                        self.assertEqual(stream.tell(), f*12)
                        self.assertEqual(count, 6)
                        return original(stream, dtype=dtype, count=count)
                    with patch('kglobal_analysis.movie.np.fromfile', side_effect=bounded) as read:
                        values = case.read_movie_frame(name, '090', f, byteorder='little')
                    self.assertEqual(read.call_count, 1)
                    self.assertEqual(values.shape, (3, 2))
                    self.assertTrue(values.flags.owndata)
                    for x, y in ((0, 0), (1, 0), (0, 1), (2, 1)):
                        q = j*100+f*6+x+3*y
                        expected = f*100+j*2+(q+32768)/65535
                        self.assertEqual(values[x, y], expected)

    def test_volume_layout_and_validation(self):
        layout = VolumeLayout(3, 2, 2)
        self.assertEqual(layout.shape, (3, 2, 2))
        self.assertEqual(layout.samples_per_frame, 12)
        self.assertEqual(layout.frame_bytes, 24)
        self.assertEqual(layout.frame_count(48), 2)
        for size in (0, -1, 23, 25):
            with self.assertRaises(ValueError):
                layout.frame_count(size)
        for shape in ((0, 2, 2), (3, -1, 2), (3, 2, 0)):
            with self.assertRaises(ValueError):
                VolumeLayout(*shape)
        for shape in ((True, 2, 2), (3, 2., 2), (3, 2, '2')):
            with self.assertRaises(TypeError):
                VolumeLayout(*shape)

    def test_multiframe_3d_fixed_voxels_and_offsets(self):
        (self.root/'param').write_text(PARAM.replace('#define nz 1', '#define nz 2'))
        codes = list(range(-12, 12))
        (self.root/'movie.by.090').write_bytes(struct.pack('<24h', *codes))
        # identity decode: q -> q, for every variable in both frames
        (self.root/'movie.log.090').write_text('-32768 32767\n'*36)
        case = KGlobalCase(self.root)
        self.assertEqual(case.movie_frame_count('by', '090'), 2)
        original = np.fromfile
        for f in (0, 1):
            def bounded(stream, *, dtype, count):
                self.assertEqual(stream.tell(), f*24)
                self.assertEqual(count, 12)
                result = original(stream, dtype=dtype, count=count)
                self.assertEqual(stream.tell(), (f+1)*24)
                return result
            with patch('kglobal_analysis.movie.np.fromfile', side_effect=bounded) as read:
                values = case.read_movie_frame('by', '090', f, byteorder='little')
            self.assertEqual(read.call_count, 1)
            self.assertEqual(values.shape, (3, 2, 2))
            self.assertEqual(values.dtype, np.dtype('float64'))
            self.assertTrue(values.flags.f_contiguous)
            self.assertTrue(values.flags.owndata)
            for x, y, z in ((0,0,0), (1,0,0), (0,1,0), (0,0,1), (2,1,1)):
                self.assertEqual(values[x,y,z], codes[f*12+x+3*(y+2*z)])
        series = case.movie('by', byteorder='little')
        np.testing.assert_array_equal(series.read_time(1.05), values)

    def test_unknown_storage_name_rejected_even_when_file_exists(self):
        (self.root/'movie.other.090').write_bytes(b'\0'*24)
        case = KGlobalCase(self.root)
        for name in ('other', 'Bx', '../bx', 'log'):
            for call in (lambda: case.read_movie_frame(name, '090', 0, byteorder='little'),
                         lambda: case.movie_frame_count(name, '090'),
                         lambda: case.movie_segment(name, '090', byteorder='little'),
                         lambda: case.movie(name, byteorder='little')):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Unsupported movie variable'):
                    call()

    def test_unsupported_formats_rejected_before_reading(self):
        for text in (PARAM.replace('movie_kglobal3.0.h','other.h'),
                     PARAM.replace('#define double_byte','#define four_byte'),
                     PARAM+'#define four_byte\n', PARAM+'#define mult_species\n',
                     PARAM+'#define heatfluxmovies\n', PARAM+'#define movie_header2 "x.h"\n'):
            (self.root/'param').write_text(text)
            case = KGlobalCase(self.root)
            with patch('kglobal_analysis.movie.np.fromfile', side_effect=AssertionError('read')):
                for call in (lambda: case.read_movie_frame('jihpar','090',0,byteorder='little'),
                             lambda: case.movie('jihpar',byteorder='little')):
                    with self.assertRaises(ValueError):
                        call()

    def test_bx_wrapper_delegates_to_generic_decoder(self):
        case = KGlobalCase(self.root)
        result = object()
        with patch.object(case, 'read_movie_frame', return_value=result) as reader:
            self.assertIs(case.read_bx_frame('090', 1, byteorder='little'), result)
            reader.assert_called_once_with('bx', '090', 1, byteorder='little')

    def test_generic_byteorder_and_invalid_index(self):
        case = KGlobalCase(self.root)
        expected = case.read_movie_frame('ni', '090', 1, byteorder='little')
        (self.root/'movie.ni.090').write_bytes(struct.pack('>12h', *range(12)))
        np.testing.assert_array_equal(case.read_movie_frame('ni', '090', 1, byteorder='big'), expected)
        with patch('kglobal_analysis.movie.np.fromfile', side_effect=AssertionError('read')):
            for i in (-1, 2):
                with self.assertRaises(IndexError):
                    case.read_movie_frame('ni','090',i,byteorder='big')
            for i in (True, .5, '0'):
                with self.assertRaises(TypeError):
                    case.read_movie_frame('ni','090',i,byteorder='big')
            with self.assertRaises(ValueError):
                case.read_movie_frame('ni','090',0,byteorder='native')

    def add_second_segment(self):
        (self.root/'movie.jihpar.002').write_bytes(struct.pack('<12h', *range(12)))
        self.write_log('002')
        self.write_times('002', [1.1, 1.15])

    def test_generic_timeline_order_selection_and_lazy_construction(self):
        self.add_second_segment()
        (self.root/'movie.log.999').touch()
        case = KGlobalCase(self.root)
        original = Path.open
        def guard(path, *args, **kwargs):
            self.assertTrue(path.name.startswith('p3d.stdout.'))
            return original(path, *args, **kwargs)
        with patch.object(Path, 'open', new=guard), patch('kglobal_analysis.movie.np.fromfile', side_effect=AssertionError('read')):
            series = case.movie('jihpar', byteorder='little')
            self.assertIsInstance(series, MovieSeries)
            self.assertEqual(series.variable.storage_name, 'jihpar')
            self.assertEqual(series.suffixes, ('090', '002'))
            self.assertEqual(series.frame_count, 4)
            self.assertEqual(series.times, (1., 1.05, 1.1, 1.15))
            self.assertEqual(series.locate_time(1.05), ('090', 1))
            self.assertEqual(series.locate_time(1.1), ('002', 0))
            self.assertEqual(case.movie('ni',byteorder='little').suffixes, ('090',))
        with patch.object(case,'read_movie_frame',return_value=object()) as reader:
            self.assertIs(series.read_time(1.1), reader.return_value)
            reader.assert_called_once_with('jihpar','002',0,byteorder='little')
            reader.reset_mock()
            with self.assertRaises(KeyError):
                series.read_time(1.075)
            reader.assert_not_called()

    def test_generic_boundaries_and_invalid_times(self):
        self.add_second_segment()
        for times, message in (([1.05,1.1], 'Overlapping'), ([1.025,1.075], 'Overlapping'),
                               ([1.2,1.25], 'Gap'), ([1.15,1.1], 'strictly increasing'),
                               ([1.1], 'jihpar has 2 frames')):
            self.write_times('002', times)
            with self.subTest(times=times), self.assertRaisesRegex(TimeMetadataError, message):
                KGlobalCase(self.root).movie('jihpar',byteorder='little')

    def test_generic_missing_stdout_and_index_access(self):
        (self.root/'p3d.stdout.090').unlink()
        case = KGlobalCase(self.root)
        segment = case.movie_segment('jihpar','090',byteorder='little')
        self.assertIsInstance(segment, MovieSegment)
        with self.assertRaisesRegex(TimeMetadataError,'No p3d.stdout'):
            _ = segment.times
        self.assertEqual(segment.read_frame(0).shape, (3,2))
        with self.assertRaisesRegex(TimeMetadataError,'jihpar segment 090'):
            case.movie('jihpar',byteorder='little')

    def test_generic_missing_variable_companions_and_partial_volume(self):
        case = KGlobalCase(self.root)
        (self.root/'movie.jihpar.090').write_bytes(b'\0'*23)
        with self.assertRaisesRegex(ValueError,'positive multiple'):
            case.movie_frame_count('jihpar','090')
        (self.root/'movie.jihpar.090').unlink()
        with self.assertRaises(FileNotFoundError):
            KGlobalCase(self.root).read_movie_frame('jihpar','090',0,byteorder='little')
        with self.assertRaisesRegex(ValueError,'No jihpar segments'):
            KGlobalCase(self.root).movie('jihpar',byteorder='little')
