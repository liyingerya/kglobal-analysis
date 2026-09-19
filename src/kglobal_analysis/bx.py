"""Compatibility wrappers for the original 2D Bx API; decoding lives in movie."""

# Retain the existing module's NumPy instrumentation hook.
import numpy as np
from .format import VolumeLayout
from .movie import _integer, _inverse_quantization, read_movie_minmax


def frame_count_from_size(size_bytes: int, Nx: int, Ny: int) -> int:
    return VolumeLayout(_integer(Nx, "Nx"), _integer(Ny, "Ny"), 1).frame_count(
        _integer(size_bytes, "size_bytes"))


def _require_2d(case):
    if case.parameters.Nz is not None and case.parameters.Nz != 1:
        raise ValueError("Only 2D data (Nz == 1) is supported by the compatibility Bx API; use the generic movie API")


def bx_frame_count(case, segment: str) -> int:
    _require_2d(case)
    return case.movie_frame_count("bx", segment)


def read_bx_minmax(log_path, frame_index: int, frame_count: int) -> tuple[float, float]:
    return read_movie_minmax(log_path, "bx", frame_index, frame_count)


def read_bx_frame(case, segment: str, frame_index: int, *, byteorder: str):
    _require_2d(case)
    return case.read_movie_frame("bx", segment, frame_index, byteorder=byteorder)
