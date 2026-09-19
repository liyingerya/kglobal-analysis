"""Real Bx/jihpar through the same decoder; external fixtures stay read-only."""
from pathlib import Path
from tempfile import TemporaryDirectory
import os
import re
import struct
import unittest
from unittest.mock import patch

import numpy as np
from kglobal_analysis import KGlobalCase, STANDARD_MOVIE_FORMAT, VolumeLayout

DATA = Path(__file__).resolve().parents[2] / 'validation-data'
FIRST = Path(os.environ.get('KGLOBAL_VALIDATION_CASE', DATA / 'hcs_large_005'))
MULTI = Path(os.environ.get('KGLOBAL_MULTI_VALIDATION_CASE', DATA / 'hcs_large_multi'))


@unittest.skipUnless(FIRST.is_dir() and MULTI.is_dir(), 'real fixture directories unavailable')
class RealMovieTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'param_hcs_large').symlink_to((FIRST/'param_hcs_large').resolve(strict=True))
        for name in ('movie.bx.004', 'movie.jihpar.004', 'movie.log.004', 'p3d.stdout.004'):
            (self.root/name).symlink_to((MULTI/name).resolve(strict=True))
        self.case = KGlobalCase(self.root)

    def test_observed_file_log_and_stdout_counts_agree(self):
        p = self.case.parameters
        layout = VolumeLayout(p.Nx, p.Ny, p.Nz)
        size = (self.root/'movie.jihpar.004').stat().st_size
        # Verified from the provided file before adding this assertion.
        self.assertEqual(size, 1342177280)
        count, remainder = divmod(size, layout.frame_bytes)
        self.assertEqual((count, remainder), (20, 0))
        self.assertEqual(layout.frame_bytes, 67108864)
        self.assertEqual(len((self.root/'movie.log.004').read_text().splitlines()), count*18)
        events = re.findall(r'^\s*movie output,\s*t=\s*(\S+)\s*$',
                            (self.root/'p3d.stdout.004').read_text(), re.M)
        self.assertEqual(len(events), count)
        self.assertEqual(self.case.movie_frame_count('jihpar','004'), count)
        series = self.case.movie('jihpar',byteorder='little')
        self.assertEqual(series.frame_count,count)
        self.assertEqual(series.times,tuple(map(float,events)))
        self.assertAlmostEqual(series.times[0],4.05,places=6)
        self.assertAlmostEqual(series.times[-1],5.,places=6)
        self.assertEqual(STANDARD_MOVIE_FORMAT.variable('jihpar').log_index,17)

    def test_jihpar_first_last_time_reads_are_bounded_and_equal(self):
        series = self.case.movie('jihpar',byteorder='little')
        original = np.fromfile
        lines = (self.root/'movie.log.004').read_text().splitlines()
        movie = self.root/'movie.jihpar.004'
        for index, time in ((0,4.05),(19,5.)):
            calls=[]
            def bounded(stream, *, dtype, count):
                self.assertEqual(Path(stream.name).name, 'movie.jihpar.004')
                self.assertEqual(count,8192*4096)
                self.assertEqual(dtype,np.dtype('<i2'))
                self.assertEqual(stream.tell(),index*67108864)
                result=original(stream,dtype=dtype,count=count)
                self.assertEqual(stream.tell(),(index+1)*67108864)
                self.assertEqual(result.nbytes,67108864)
                calls.append(result.nbytes)
                return result
            with patch('kglobal_analysis.movie.np.fromfile',side_effect=bounded), \
                    patch.object(Path,'read_bytes',side_effect=AssertionError('unbounded read')):
                by_time=series.read_time(time)
                self.assertEqual(calls,[67108864])
                direct=self.case.read_movie_frame('jihpar','004',index,byteorder='little')
                self.assertEqual(calls,[67108864,67108864])
            np.testing.assert_array_equal(by_time,direct)
            self.assertEqual(direct.shape,(8192,4096))
            self.assertEqual(direct.dtype,np.dtype('float64'))
            low,high=map(float,lines[index*18+17].split())
            self.assertAlmostEqual(float(direct.min()),low,places=12)
            self.assertAlmostEqual(float(direct.max()),high,places=12)
            with movie.open('rb') as stream:
                for x,y in ((0,0),(1,0),(0,1),(4096,2048),(8191,4095)):
                    stream.seek(index*67108864+2*(x+8192*y))
                    q=struct.unpack('<h',stream.read(2))[0]
                    self.assertAlmostEqual(direct[x,y],low+(q+32768)*(high-low)/65535,places=12)
            del by_time,direct

    def test_real_bx_generic_matches_compatibility_and_uses_same_decoder(self):
        from kglobal_analysis.movie import read_movie_frame
        for index in (0,19):
            with patch('kglobal_analysis.movie.read_movie_frame',wraps=read_movie_frame) as reader:
                generic=self.case.read_movie_frame('bx','004',index,byteorder='little')
                compatible=self.case.read_bx_frame('004',index,byteorder='little')
            self.assertEqual(reader.call_count,2)
            for call in reader.call_args_list:
                self.assertEqual(call.args,(self.case,'bx','004',index))
            np.testing.assert_array_equal(generic,compatible)
            del generic,compatible
