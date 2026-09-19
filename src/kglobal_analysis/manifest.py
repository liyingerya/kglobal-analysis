"""Read-only movie metadata validation: no sample reads, hashes, or repairs."""

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING
import math

from .format import VolumeLayout, movie_format
from .movie_log import iter_movie_ranges
from .times import read_movie_times, validate_movie_times

if TYPE_CHECKING:
    from .case import KGlobalCase


@dataclass(frozen=True)
class MovieValidationIssue:
    code: str
    message: str
    suffix: str | None = None
    storage_name: str | None = None
    path: str | None = None


@dataclass(frozen=True)
class MovieParameterSummary:
    parameter_file: str | None
    Nx: int | None
    Ny: int | None
    Nz: int | None
    dt: float | None
    n_movieout: int | None
    nominal_cadence: float | None
    movie_header: str | None
    encoding: str | None
    double_byte: bool
    four_byte: bool
    format_supported: bool
    schema_variables: tuple[str, ...]
    frame_bytes: int | None
    definitions: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class MovieFileSummary:
    storage_name: str
    path: str
    size_bytes: int | None
    frame_count: int | None


@dataclass(frozen=True)
class MovieSegmentSummary:
    suffix: str
    files: tuple[MovieFileSummary, ...]
    frame_count: int | None
    stdout_path: str | None
    stdout_event_count: int | None
    stdout_valid: bool | None
    times: tuple[float, ...]
    first_time: float | None
    last_time: float | None
    log_path: str | None
    log_entry_count: int | None
    log_frame_count: int | None
    log_valid: bool | None


@dataclass(frozen=True)
class MovieVariableCoverage:
    storage_name: str
    suffixes: tuple[str, ...]
    frame_counts: tuple[tuple[str, int | None], ...]
    total_frame_count: int | None
    timeline_valid: bool | None
    first_time: float | None
    last_time: float | None


@dataclass(frozen=True)
class MovieCaseReport:
    directory: str
    parameters: MovieParameterSummary
    segments: tuple[MovieSegmentSummary, ...]
    variables: tuple[MovieVariableCoverage, ...]
    absent_schema_variables: tuple[str, ...]
    errors: tuple[MovieValidationIssue, ...]
    warnings: tuple[MovieValidationIssue, ...]

    @property
    def ok(self) -> bool:
        """No detected metadata errors; not proof of valid sample values."""
        return not self.errors

    def to_dict(self) -> dict:
        """Return detached JSON-compatible metadata; never movie sample arrays."""
        def convert(value):
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [convert(item) for item in value]
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value
        result = convert(asdict(self))
        result['ok'] = self.ok
        return result

    def summary(self) -> str:
        status = 'OK' if self.ok else 'INVALID'
        return (f"Movie case {status}: {len(self.segments)} segments, "
                f"{len(self.variables)} variables, {len(self.errors)} errors, "
                f"{len(self.warnings)} warnings")

    def __str__(self) -> str:
        return self.summary()


def validate_movie_case(case: "KGlobalCase") -> MovieCaseReport:
    """Validate the case snapshot using text metadata and binary file sizes only.

    Absent/narrower variable coverage is not corruption. Missing companions for
    present binaries are errors because their required metadata cannot be checked.
    Construct a new KGlobalCase to rediscover added/removed files or parameters.
    """
    errors = []
    warnings = []

    def issue(code, message, suffix=None, storage_name=None, path=None):
        errors.append(MovieValidationIssue(code, message, suffix, storage_name,
                                           str(path) if path is not None else None))

    p = case.parameters
    schema = None
    layout = None
    try:
        schema = movie_format(p)
    except ValueError as error:
        issue('unsupported_format', str(error))
    try:
        if p.Nx is None or p.Ny is None or p.Nz is None:
            raise ValueError('Resolved nx, ny, nz, pex, pey, pez are required')
        if schema is not None:
            layout = VolumeLayout(p.Nx, p.Ny, p.Nz)
    except (ValueError, TypeError) as error:
        issue('invalid_dimensions', str(error))
    if p.movie_dt is None:
        warnings.append(MovieValidationIssue('unknown_cadence', 'Nominal cadence is unresolved'))
    elif not math.isfinite(p.movie_dt) or p.movie_dt <= 0:
        issue('invalid_cadence', 'Nominal movie cadence must be finite and positive for increasing times')

    encoding = ('conflicting' if p.double_byte and p.four_byte else
                'double_byte' if p.double_byte else 'four_byte' if p.four_byte else None)
    parameters = MovieParameterSummary(
        str(case.parameter_file) if case.parameter_file else None,
        p.Nx, p.Ny, p.Nz, p.dt, p.n_movieout, p.movie_dt, p.movie_header,
        encoding, p.double_byte, p.four_byte, schema is not None,
        tuple(spec.storage_name for spec in schema.variables) if schema else (),
        layout.frame_bytes if layout else None, tuple(sorted(p.definitions.items())),
    )
    supported = set(parameters.schema_variables)
    segments = []
    for suffix in case.suffixes:
        entry = case.index.segments[suffix]
        files = []
        for name in entry.variables:
            path = entry.movies[name]
            size = count = None
            if schema is not None and name not in supported:
                issue('unsupported_variable', f'Unsupported movie variable {name!r}', suffix, name, path)
            try:
                size = path.stat().st_size
                if layout is not None and name in supported:
                    count = layout.frame_count(size)
            except (OSError, ValueError) as error:
                issue('invalid_binary_size', str(error), suffix, name, path)
            files.append(MovieFileSummary(name, str(path), size, count))
        known_counts = {file.frame_count for file in files if file.frame_count is not None}
        count = next(iter(known_counts)) if files and len(known_counts) == 1 and all(
            file.frame_count is not None for file in files) else None
        if len(known_counts) > 1:
            issue('segment_frame_counts', f'Binary frame counts disagree: {sorted(known_counts)}', suffix)
        if not files:
            warnings.append(MovieValidationIssue('companion_only_segment', 'No movie binaries in segment', suffix))

        times = ()
        event_count = None
        stdout_valid = None
        if entry.stdout is not None:
            try:
                times = read_movie_times(entry.stdout)
                event_count = len(times)
                if not times:
                    raise ValueError('No actual movie timestamps in stdout')
                validate_movie_times(times, len(times), p.movie_dt)
                stdout_valid = True
            except (OSError, ValueError) as error:
                stdout_valid = False
                issue('invalid_stdout', str(error), suffix, path=entry.stdout)
        elif files:
            stdout_valid = False
            issue('missing_stdout', 'Absolute movie times unavailable: missing stdout', suffix)
        if stdout_valid:
            for file in files:
                if file.frame_count is not None:
                    try:
                        validate_movie_times(times, file.frame_count, p.movie_dt, variable=file.storage_name)
                    except ValueError as error:
                        issue('binary_stdout_count', str(error), suffix, file.storage_name, entry.stdout)

        log_count = log_frames = None
        log_valid = None
        if entry.movie_log is not None:
            try:
                log_count = sum(1 for _ in iter_movie_ranges(entry.movie_log))
                if schema is not None:
                    width = len(schema.variables)
                    if log_count == 0 or log_count % width:
                        raise ValueError(f'Movie log has {log_count} entries; expected a positive multiple of {width}')
                    log_frames = log_count // width
                    log_valid = True
                    # Compare against every known binary count; stdout provides
                    # a check even in a segment without readable movie binaries.
                    expected_counts = set(known_counts)
                    if stdout_valid:
                        expected_counts.add(event_count)
                    for expected in sorted(expected_counts):
                        if log_frames != expected:
                            log_valid = False
                            issue('log_frame_count', f'Movie log has {log_count} entries; expected '
                                  f'{expected * width} for {expected} frames', suffix, path=entry.movie_log)
            except (OSError, ValueError) as error:
                log_valid = False
                issue('invalid_log', str(error), suffix, path=entry.movie_log)
        elif files:
            log_valid = False
            issue('missing_log', 'Missing movie.log for present movie binaries', suffix)
        segments.append(MovieSegmentSummary(
            suffix, tuple(files), count, str(entry.stdout) if entry.stdout else None,
            event_count, stdout_valid, times if stdout_valid else (),
            times[0] if stdout_valid else None, times[-1] if stdout_valid else None,
            str(entry.movie_log) if entry.movie_log else None, log_count, log_frames, log_valid,
        ))

    variables = []
    valid_coverages = []
    for name in case.variables:
        present = [segment for segment in segments if any(file.storage_name == name for file in segment.files)]
        counts = {segment.suffix: next(file.frame_count for file in segment.files if file.storage_name == name)
                  for segment in present}
        timeline_valid = None
        first = last = None
        suffixes = tuple(segment.suffix for segment in present)
        if layout is not None and name in supported:
            timeline_valid = False
            if all(segment.stdout_valid and counts[segment.suffix] == segment.stdout_event_count for segment in present):
                try:
                    # MovieSeries validates metadata only. Its required byteorder
                    # argument is unused here; no byte order is inferred or reported.
                    series = case.movie(name, byteorder='little')
                    timeline_valid = True
                    suffixes = series.suffixes
                    first, last = series.times[0], series.times[-1]
                    valid_coverages.append((name, series.times, series._locations))
                except (OSError, ValueError) as error:
                    issue('variable_timeline', str(error), storage_name=name)
        variables.append(MovieVariableCoverage(
            name, suffixes, tuple((suffix, counts[suffix]) for suffix in suffixes),
            sum(counts.values()) if all(count is not None for count in counts.values()) else None,
            timeline_valid, first, last,
        ))
    absent = tuple(name for name in parameters.schema_variables if name not in case.variables)
    if absent:
        warnings.append(MovieValidationIssue('absent_variables', 'Absent schema variables: ' + ', '.join(absent)))
    if not variables:
        warnings.append(MovieValidationIssue('no_movie_binaries', 'No movie binaries discovered'))
    if valid_coverages and any(coverage[1:] != valid_coverages[0][1:] for coverage in valid_coverages[1:]):
        warnings.append(MovieValidationIssue('unequal_coverage', 'Internally valid variable timelines have unequal coverage: '
                                             + ', '.join(coverage[0] for coverage in valid_coverages)))
    return MovieCaseReport(str(case.directory), parameters, tuple(segments), tuple(variables),
                           absent, tuple(errors), tuple(warnings))
