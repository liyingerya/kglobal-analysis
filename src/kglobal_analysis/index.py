"""Filename parsing and nonrecursive discovery; no file contents are read."""

from dataclasses import dataclass, field
from pathlib import Path
import re


@dataclass(frozen=True)
class FileRecord:
    kind: str
    suffix: str
    variable: str | None = None


def parse_filename(name: str) -> FileRecord | None:
    """Parse an exact basename, preserving the numeric suffix as written."""
    match = re.fullmatch(r"movie\.([A-Za-z_][A-Za-z0-9_]*)\.([0-9]+)", name)
    if match:
        variable, suffix = match.groups()
        if variable == "log":
            return FileRecord("log", suffix)
        return FileRecord("movie", suffix, variable)
    match = re.fullmatch(r"p3d\.stdout\.([0-9]+)", name)
    if match:
        return FileRecord("stdout", match.group(1))
    return None


@dataclass
class Segment:
    suffix: str
    movies: dict[str, Path] = field(default_factory=dict)
    movie_log: Path | None = None
    stdout: Path | None = None

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(sorted(self.movies))


@dataclass
class CaseIndex:
    segments: dict[str, Segment]

    @property
    def suffixes(self) -> tuple[str, ...]:
        return tuple(sorted(self.segments, key=lambda value: (int(value), value)))

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(sorted({name for segment in self.segments.values()
                             for name in segment.movies}))


def scan_case(directory: str | Path) -> CaseIndex:
    """Index immediate files, including segments with only log/stdout files."""
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    segments: dict[str, Segment] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        record = parse_filename(path.name)
        if record is None:
            continue
        segment = segments.setdefault(record.suffix, Segment(record.suffix))
        if record.kind == "movie":
            assert record.variable is not None
            segment.movies[record.variable] = path
        elif record.kind == "log":
            segment.movie_log = path
        else:
            segment.stdout = path
    return CaseIndex(segments)
