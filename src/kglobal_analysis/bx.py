"""One-frame 2D Bx reader for the standard 18-entry KGlobal movie log.

Disk order is x fastest, then y, then frame, with native signed int16 samples.
The producer's byte order must be supplied explicitly; files do not label it.
"""

from pathlib import Path
from typing import TYPE_CHECKING
import math
import operator

import numpy as np

if TYPE_CHECKING:
    from .case import KGlobalCase


def _integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


def frame_count_from_size(size_bytes: int, Nx: int, Ny: int) -> int:
    """Count frames of Nx*Ny signed int16 samples; reject partial/empty files."""
    size_bytes = _integer(size_bytes, "size_bytes")
    Nx, Ny = _integer(Nx, "Nx"), _integer(Ny, "Ny")
    if Nx <= 0 or Ny <= 0:
        raise ValueError("Nx and Ny must be positive")
    frame_bytes = Nx * Ny * 2
    if size_bytes <= 0 or size_bytes % frame_bytes:
        raise ValueError(f"Bx file size {size_bytes} is not a positive multiple of {frame_bytes}")
    return size_bytes // frame_bytes


def _frame_index(frame_index: int, frame_count: int) -> int:
    frame_index = _integer(frame_index, "frame_index")
    if not 0 <= frame_index < frame_count:
        raise IndexError(f"frame_index {frame_index} outside [0, {frame_count})")
    return frame_index


def _layout(case: "KGlobalCase", segment: str) -> tuple[Path, int, int]:
    p = case.parameters
    if p.Nx is None or p.Ny is None or p.Nz is None:
        raise ValueError("Resolved nx, ny, nz, pex, pey, pez are required")
    if p.Nz != 1:
        raise ValueError("Only 2D data (Nz == 1) is supported")
    if not p.double_byte or p.four_byte:
        raise ValueError("Only double_byte without four_byte is supported")
    if p.movie_header != "movie_kglobal3.0.h":
        raise ValueError("Only movie_kglobal3.0.h with the standard 18-entry log is supported")
    for name in ("heatfluxmovies", "mult_species", "movie_header2"):
        if name in p.definitions:
            raise ValueError(f"Unsupported movie layout option: {name}")
    entry = case.index.segments[segment]
    if "bx" not in entry.movies:
        raise FileNotFoundError(f"No movie.bx.{segment} in {case.directory}")
    return entry.movies["bx"], p.Nx, p.Ny


def bx_frame_count(case: "KGlobalCase", segment: str) -> int:
    path, Nx, Ny = _layout(case, segment)
    return frame_count_from_size(path.stat().st_size, Nx, Ny)


def read_bx_minmax(log_path: str | Path, frame_index: int, frame_count: int) -> tuple[float, float]:
    """Read Bx entry 4 of frame_index and validate the 18-pairs/frame text log.

    The log must match the complete-frame count from the Bx file. Malformed,
    nonfinite, reversed, missing, or extra entries are rejected.
    """
    frame_count = _integer(frame_count, "frame_count")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    frame_index = _frame_index(frame_index, frame_count)
    expected = frame_count * 18
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
            if count - 1 == frame_index * 18 + 4:
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


def read_bx_frame(case: "KGlobalCase", segment: str, frame_index: int,
                  *, byteorder: str) -> np.ndarray:
    """Seek and read exactly one frame, returning an owning float64 (Nx, Ny) array."""
    if byteorder not in ("little", "big"):
        raise ValueError("byteorder must be explicitly 'little' or 'big'")
    path, Nx, Ny = _layout(case, segment)
    entry = case.index.segments[segment]
    if entry.movie_log is None:
        raise FileNotFoundError(f"No movie.log.{segment} in {case.directory}")
    dtype = np.dtype("<i2" if byteorder == "little" else ">i2")
    samples = Nx * Ny
    with path.open("rb") as stream:
        # Obtain size from the same open handle used for the bounded read.
        stream.seek(0, 2)
        count = frame_count_from_size(stream.tell(), Nx, Ny)
        frame_index = _frame_index(frame_index, count)
        low, high = read_bx_minmax(entry.movie_log, frame_index, count)
        stream.seek(frame_index * samples * 2)
        encoded = np.fromfile(stream, dtype=dtype, count=samples)
    if encoded.size != samples:
        raise ValueError("Short Bx frame read; file may have changed during reading")
    return _inverse_quantization(encoded.reshape((Nx, Ny), order="F"), low, high)
