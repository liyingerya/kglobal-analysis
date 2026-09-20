"""Release checks: wheel privacy/content and installed metadata/import smoke.

No build/install is performed here. Run --wheel FILE, --installed-root DIR,
or --editable-root DIR. Use an external cwd and the intended installed target.
"""
import argparse
from importlib import metadata
from pathlib import Path, PurePosixPath
import sys
import tarfile
import zipfile


def check_wheel(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        forbidden = {'notes', 'upstream', 'legacy', 'validation-data', 'worktrees',
                     '__pycache__', '.git', '.venv'}
        for name in names:
            parts = PurePosixPath(name).parts
            leaf = parts[-1] if parts else ''
            if (name.startswith('/') or '\\' in name or '..' in parts or forbidden.intersection(parts)
                    or leaf == '.DS_Store' or (leaf.startswith('movie.') and leaf != 'movie.py') or leaf.startswith('p3d-')
                    or leaf.endswith(('.data', '.pyc', '.pyo'))):
                raise ValueError(f'Unexpected wheel member: {name}')
            if not (parts[0] == 'kglobal_analysis' or parts[0].endswith('.dist-info')):
                raise ValueError(f'Unexpected wheel top-level member: {name}')
            if leaf.endswith(('.py', '.md')) or leaf in ('METADATA', 'RECORD'):
                content = archive.read(name)
                if any(marker in content for marker in (b'/Users/', b'/home/', b'file:///')):
                    raise ValueError(f'Local absolute path in wheel: {name}')
        for module in ('__init__', 'case', 'derived', 'geometry', 'flux', 'workflows'):
            if f'kglobal_analysis/{module}.py' not in names:
                raise ValueError(f'Missing package module: {module}')
        for leaf in ('METADATA', 'WHEEL', 'RECORD'):
            if sum(n.endswith('.dist-info/'+leaf) for n in names) != 1:
                raise ValueError(f'Missing/ambiguous wheel metadata: {leaf}')
    print(f'Wheel content/privacy passed: {len(names)} members')


def check_sdist(path):
    """Inspect archive names without extracting untrusted paths."""
    with tarfile.open(path) as archive:
        names = archive.getnames()
        roots = {PurePosixPath(n).parts[0] for n in names}
        assert len(roots) == 1, roots
        relative = set()
        for member in archive.getmembers():
            parts = PurePosixPath(member.name).parts
            assert not member.issym() and not member.islnk(), member.name
            assert not member.name.startswith('/') and '..' not in parts, member.name
            relative.add('/'.join(parts[1:]))
            assert not {'notes', 'upstream', 'legacy', 'validation-data', 'worktrees', '__pycache__', '.venv'}.intersection(parts), member.name
            assert not (parts[-1].startswith('movie.') and parts[-1] != 'movie.py'), member.name
            assert not parts[-1].startswith('p3d-'), member.name
            assert not parts[-1].endswith(('.data', '.pyc', '.pyo')), member.name
        for required in ('pyproject.toml', 'README.md', 'src/kglobal_analysis/__init__.py'):
            assert required in relative, required
    print(f'Sdist content/privacy passed: {len(names)} members')


def check_installed(root):
    import kglobal_analysis as kga
    assert 'matplotlib' not in sys.modules
    package = Path(kga.__file__).resolve()
    assert package.is_relative_to(Path(root).resolve()), (package, root)
    dist = metadata.distribution('kglobal-analysis')
    assert dist.metadata['Name'] == 'kglobal-analysis'
    assert kga.__version__ == dist.version and kga.__version__ != '0+unknown'
    assert dist.metadata['Requires-Python'] == '>=3.10'
    requirements = dist.requires or []
    for dep in ('numpy>=1.23', 'xarray>=2024.7.0', 'dask[array]>=2024.7.0'):
        assert dep in [r.replace(' ', '') for r in requirements], requirements
    plotting = [r for r in requirements if r.lower().startswith('matplotlib')]
    assert len(plotting) == 1 and 'extra == "plot"' in plotting[0]
    assert 'plot' in dist.metadata.get_all('Provides-Extra', [])
    for name in ('KGlobalCase', 'particle_quantity_catalog', 'particle_quantity_series',
                 'particle_map_series', 'firehose_series', 'firehose_map_series', 'magnetic_flux_series'):
        assert callable(getattr(kga, name)) and name in kga.__all__
    print('version:', kga.__version__)
    print('package:', package)
    print('case:', kga.KGlobalCase)
    print('map workflow:', kga.particle_map_series)
    print('Installed metadata/API/import smoke passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--wheel', type=Path)
    group.add_argument('--sdist', type=Path)
    group.add_argument('--installed-root', type=Path)
    group.add_argument('--editable-root', type=Path)
    args = parser.parse_args()
    if args.wheel:
        check_wheel(args.wheel)
    elif args.sdist:
        check_sdist(args.sdist)
    else:
        check_installed(args.installed_root or args.editable_root)
