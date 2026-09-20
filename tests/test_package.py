"""Public release surface and version/import behavior, without real case data."""
from importlib import metadata, util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import tarfile
import io

import kglobal_analysis as kga

ROOT = Path(__file__).resolve().parents[1]
spec = util.spec_from_file_location('check_release', ROOT/'scripts/check_release.py')
release = util.module_from_spec(spec)
spec.loader.exec_module(release)


class PackageTests(unittest.TestCase):
    def test_root_public_exports(self):
        self.assertEqual(len(kga.__all__), len(set(kga.__all__)))
        for name in kga.__all__:
            self.assertTrue(hasattr(kga, name), name)
        for name in ('KGlobalCase', 'particle_quantity_catalog', 'particle_quantity_series',
                     'particle_map_series', 'firehose_series', 'firehose_map_series', 'magnetic_flux_series'):
            self.assertIn(name, kga.__all__)
            self.assertTrue(callable(getattr(kga, name)))
        self.assertIn('__version__', kga.__all__)
        self.assertTrue(kga.__version__)

    def test_installed_version_when_available(self):
        try:
            version = metadata.version('kglobal-analysis')
        except metadata.PackageNotFoundError:
            self.assertEqual(kga.__version__, '0+unknown')
        else:
            self.assertEqual(kga.__version__, version)

    def test_installed_metadata_when_available(self):
        try:
            dist = metadata.distribution('kglobal-analysis')
        except metadata.PackageNotFoundError:
            self.skipTest('Distribution metadata unavailable in uninstalled checkout')
        self.assertEqual(dist.metadata['Name'], 'kglobal-analysis')
        self.assertEqual(dist.metadata['Requires-Python'], '>=3.10')
        for dependency in ('numpy>=1.23', 'xarray>=2024.7.0', 'dask[array]>=2024.7.0'):
            self.assertIn(dependency, [r.replace(' ', '') for r in dist.requires])
        plotting = [r for r in dist.requires if r.startswith('matplotlib')]
        self.assertEqual(len(plotting), 1)
        self.assertIn('extra == "plot"', plotting[0])

    def test_version_metadata_and_source_fallback(self):
        # Fresh subprocesses simulate missing metadata without uninstalling anything.
        for expected in ('7.8.9', '0+unknown'):
            code = f'''from importlib import metadata
from unittest.mock import patch
original = metadata.version
def version(name):
    if name != 'kglobal-analysis':
        return original(name)
    if {expected!r} == '0+unknown':
        raise metadata.PackageNotFoundError(name)
    return {expected!r}
with patch('importlib.metadata.version', side_effect=version):
    import kglobal_analysis as kga
    assert kga.__version__ == {expected!r}, kga.__version__
'''
            subprocess.run([sys.executable, '-c', code], check=True)

    def test_import_without_plotting_case_access_network_or_compute(self):
        code = '''import sys, builtins, socket
from pathlib import Path
from unittest.mock import patch
import dask.base
import numpy as np
original = builtins.open
def guarded(file, *args, **kwargs):
    name = str(file)
    if any(part in name for part in ('validation-data', 'movie.', 'p3d-', '/param')):
        raise AssertionError('case/product access during import: ' + name)
    return original(file, *args, **kwargs)
with patch('builtins.open', guarded), patch('socket.socket', side_effect=AssertionError('network')), \\
     patch('dask.base.compute', side_effect=AssertionError('compute')), \\
     patch('numpy.fromfile', side_effect=AssertionError('sample read')):
    import kglobal_analysis as kga
assert 'matplotlib' not in sys.modules
'''
        subprocess.run([sys.executable, '-c', code], check=True)

    def test_wheel_privacy_guard(self):
        base = [f'kglobal_analysis/{n}.py' for n in ('__init__', 'case', 'derived', 'geometry', 'flux', 'workflows', 'movie')]
        base += ['kglobal_analysis-0.1.0.dist-info/'+n for n in ('METADATA', 'WHEEL', 'RECORD')]
        bad = ['notes/a.md', 'upstream/a.py', 'legacy/a.py', 'validation-data/a', 'worktrees/a',
               'kglobal_analysis/movie.bx.005', 'kglobal_analysis/p3d-001',
               'kglobal_analysis/private.data', 'kglobal_analysis/__pycache__/a.pyc',
               'kglobal_analysis/.DS_Store', '/absolute.py', '../escape.py']
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'test.whl'
            for extra in [None] + bad:
                with zipfile.ZipFile(path, 'w') as z:
                    for name in base + ([extra] if extra else []):
                        z.writestr(name, '')
                if extra:
                    with self.subTest(extra=extra), self.assertRaises(ValueError):
                        release.check_wheel(path)
                else:
                    release.check_wheel(path)
            with zipfile.ZipFile(path, 'w') as z:
                for name in base:
                    z.writestr(name, '/Users/someone/private' if name.endswith('case.py') else '')
            with self.assertRaises(ValueError):
                release.check_wheel(path)

    def test_sdist_privacy_guard(self):
        base = ['pyproject.toml', 'README.md', 'src/kglobal_analysis/__init__.py',
                'src/kglobal_analysis/movie.py']
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'source.tar.gz'
            for extra in (None, 'notes/private.md', 'src/kglobal_analysis/movie.bx.005'):
                with tarfile.open(path, 'w:gz') as archive:
                    for name in base + ([extra] if extra else []):
                        info = tarfile.TarInfo('kglobal_analysis-0.1.0/'+name)
                        archive.addfile(info, io.BytesIO(b''))
                if extra:
                    with self.assertRaises(AssertionError):
                        release.check_sdist(path)
                else:
                    release.check_sdist(path)
