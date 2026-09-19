"""One generic bounded frame decoder for the validated double-byte movie format."""

from pathlib import Path
from typing import TYPE_CHECKING
import math
import operator
import numpy as np

from .format import STANDARD_MOVIE_FORMAT, MovieFormat, VolumeLayout, movie_format

if TYPE_CHECKING:
    from .case import KGlobalCase


def _integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


def _frame_index(frame_index: int, frame_count: int) -> int:
    frame_index = _integer(frame_index, "frame_index")
    if not 0 <= frame_index < frame_count:
        raise IndexError(f"frame_index {frame_index} outside [0, {frame_count})")
    return frame_index


def display_name(variable: str) -> str:
    """Keep historical Bx error labels; other labels are storage names."""
    return "Bx" if variable == "bx" else variable


def _layout(case: "KGlobalCase", variable: str, segment: str) -> tuple[Path, VolumeLayout, MovieFormat]:
    schema = movie_format(case.parameters)
    spec = schema.variable(variable)
    p = case.parameters
    if p.Nx is None or p.Ny is None or p.Nz is None:
        raise ValueError("Resolved nx, ny, nz, pex, pey, pez are required")
    layout = VolumeLayout(p.Nx, p.Ny, p.Nz)
    entry = case.index.segments[segment]
    if spec.storage_name not in entry.movies:
        raise FileNotFoundError(f"No movie.{spec.storage_name}.{segment} in {case.directory}")
    return entry.movies[spec.storage_name], layout, schema


def movie_frame_count(case: "KGlobalCase", variable: str, segment: str) -> int:
    path, layout, _schema = _layout(case, variable, segment)
    return layout.frame_count(path.stat().st_size)


def read_movie_minmax(log_path: str | Path, variable: str, frame_index: int, frame_count: int) -> tuple[float, float]:
    """Compatibility entry point for standalone standard-format log reading."""
    return _read_movie_minmax(log_path, variable, frame_index, frame_count,
                              schema=STANDARD_MOVIE_FORMAT)


def _read_movie_minmax(log_path: str | Path, variable: str, frame_index: int,
                       frame_count: int, *, schema: MovieFormat) -> tuple[float, float]:
    """Validate and select log ranges using the caller's selected format."""
    spec = schema.variable(variable)
    frame_count = _integer(frame_count, "frame_count")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    frame_index = _frame_index(frame_index, frame_count)
    expected = frame_count * len(schema.variables)
    selected = None
    count = 0
    with Path(log_path).open("r", encoding="ascii") as stream:
        for count, line in enumerate(stream, 1):
            if count > expected:
                raise ValueError(f"Movie log has more than {expected} entries")
            line = line.rstrip("\r\n")
            fields = line.split()
            # Fortran (E14.6,E14.6) need not have whitespace between fields.
            if len(fields) != 2 and len(line) == 28:
                fields = [line[:14], line[14:]]
            try:
                if len(fields) != 2:
                    raise ValueError("expected two numbers")
                low, high = (float(value.replace("D", "E").replace("d", "e"))
                             for value in fields)
                if not math.isfinite(low) or not math.isfinite(high) or low > high:
                    raise ValueError("invalid range")
            except ValueError as error:
                raise ValueError(f"Invalid movie log entry on line {count}: {line!r}") from error
            if count - 1 == frame_index * len(schema.variables) + spec.log_index:
                selected = (low, high)
    if count != expected:
        raise ValueError(f"Movie log has {count} entries; expected {expected}")
    assert selected is not None
    return selected


def _inverse_quantization(encoded: np.ndarray, low: float, high: float) -> np.ndarray:
    # Convert before adding the signed offset to avoid int16 overflow. Float64
    # minimizes additional arithmetic error; IDL's formula uses float32.
    values = np.array(encoded, dtype=np.float64, order="F", copy=True)
    values += 32768.0
    values *= high - low
    values /= 65535.0
    values += low
    return values


def read_movie_frame(case: "KGlobalCase", variable: str, segment: str, frame_index: int,
                     *, byteorder: str) -> np.ndarray:
    """Read one owning float64 Fortran array: (Nx, Ny), or (Nx, Ny, Nz)."""
    if byteorder not in ("little", "big"):
        raise ValueError("byteorder must be explicitly 'little' or 'big'")
    path, layout, schema = _layout(case, variable, segment)
    entry = case.index.segments[segment]
    if entry.movie_log is None:
        raise FileNotFoundError(f"No movie.log.{segment} in {case.directory}")
    dtype = np.dtype("<i2" if byteorder == "little" else ">i2")
    samples = layout.samples_per_frame
    with path.open("rb") as stream:
        # Obtain size from the same open handle used for the bounded read.
        stream.seek(0, 2)
        count = layout.frame_count(stream.tell())
        frame_index = _frame_index(frame_index, count)
        low, high = _read_movie_minmax(entry.movie_log, variable, frame_index, count,
                                     schema=schema)
        stream.seek(frame_index * layout.frame_bytes)
        encoded = np.fromfile(stream, dtype=dtype, count=samples)
    if encoded.size != samples:
        raise ValueError(f"Short {display_name(variable)} frame read; file may have changed during reading")
    return _inverse_quantization(encoded.reshape(layout.public_shape, order="F"), low, high)
