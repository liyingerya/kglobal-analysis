"""Absolute movie-event times from stdout, independent of binary reading."""

from pathlib import Path
import math
import re


class TimeMetadataError(ValueError):
    """Absolute frame times are missing, malformed, or inconsistent."""


_EVENT = re.compile(r"^\s*movie output,\s*t\s*=\s*(.*?)\s*$")
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?")


def read_movie_times(path: str | Path) -> tuple[float, ...]:
    """Read only actual movie-event lines, preserving file order and values.

    Descriptive lines such as 'two-byte movie output' are ignored. Malformed
    event values are errors, not silently skipped timestamps.
    """
    path = Path(path)
    times = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for lineno, line in enumerate(stream, 1):
                match = _EVENT.fullmatch(line)
                if match is None:
                    continue
                value = match[1]
                if not _NUMBER.fullmatch(value):
                    raise TimeMetadataError(f"{path}:{lineno}: invalid movie timestamp {value!r}")
                time = float(value.replace("D", "E").replace("d", "e"))
                if not math.isfinite(time):
                    raise TimeMetadataError(f"{path}:{lineno}: movie timestamp must be finite")
                times.append(time)
    except OSError as error:
        raise TimeMetadataError(f"Cannot read movie timestamps from {path}: {error}") from error
    return tuple(times)


def validate_movie_times(times: tuple[float, ...], frame_count: int,
                         cadence: float | None, *, variable: str = "bx") -> None:
    """Check count, finite/increasing times, and optional nominal cadence.

    Cadence comparison uses rtol=1e-6 and atol=1e-10 in simulation time units.
    Values from stdout are never replaced with a synthetic cadence grid.
    """
    label = "Bx" if variable == "bx" else variable
    if len(times) != frame_count:
        raise TimeMetadataError(f"Stdout has {len(times)} movie timestamps; {label} has {frame_count} frames")
    if not all(math.isfinite(time) for time in times):
        raise TimeMetadataError("Movie timestamps must be finite")
    if cadence is not None and (not math.isfinite(cadence) or cadence <= 0):
        raise TimeMetadataError("Nominal movie cadence must be finite and positive for increasing times")
    for index, (left, right) in enumerate(zip(times, times[1:]), 1):
        spacing = right - left
        if right <= left or not math.isfinite(spacing):
            raise TimeMetadataError(f"Movie timestamps must be strictly increasing at frame {index}")
        if cadence is not None and not math.isclose(spacing, cadence, rel_tol=1e-6, abs_tol=1e-10):
            raise TimeMetadataError(
                f"Movie cadence mismatch at frame {index}: observed {spacing}, expected {cadence}")
