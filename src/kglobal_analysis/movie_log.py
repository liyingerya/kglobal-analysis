"""Streaming movie-log range validation, shared by decoding and manifests."""

from collections.abc import Iterator
from pathlib import Path
import math


def iter_movie_ranges(path: str | Path, *, max_entries: int | None = None) -> Iterator[tuple[float, float]]:
    """Yield finite ordered ranges; accept Fortran exponents and E14.6 fields."""
    with Path(path).open('r', encoding='ascii') as stream:
        for count, line in enumerate(stream, 1):
            if max_entries is not None and count > max_entries:
                raise ValueError(f"Movie log has more than {max_entries} entries")
            line = line.rstrip('\r\n')
            fields = line.split()
            # Fortran (E14.6,E14.6) need not separate the fields with whitespace.
            if len(fields) != 2 and len(line) == 28:
                fields = [line[:14], line[14:]]
            try:
                if len(fields) != 2:
                    raise ValueError('expected two numbers')
                low, high = (float(value.replace('D', 'E').replace('d', 'e')) for value in fields)
                if not math.isfinite(low) or not math.isfinite(high) or low > high:
                    raise ValueError('invalid range')
            except ValueError as error:
                raise ValueError(f"Invalid movie log entry on line {count}: {line!r}") from error
            yield low, high
