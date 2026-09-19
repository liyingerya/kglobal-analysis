"""Time-aware access to one movie segment; every read still returns one frame."""

from typing import TYPE_CHECKING
import math

from .format import movie_format
from .times import TimeMetadataError, read_movie_times, validate_movie_times

if TYPE_CHECKING:
    import numpy as np
    import xarray as xr
    from .case import KGlobalCase


class MovieSegment:
    """Snapshot movie frame count and lazily validate absolute times from stdout.

    Missing/invalid stdout blocks time-based access, not index-based reads.
    Time equality uses absolute tolerance min(1e-6, minimum frame spacing/1000),
    with no relative tolerance, interpolation, or nearest-time fallback.
    """

    def __init__(self, case: "KGlobalCase", variable: str, suffix: str, *, byteorder: str):
        if byteorder not in ("little", "big"):
            raise ValueError("byteorder must be explicitly 'little' or 'big'")
        self.variable = movie_format(case.parameters).variable(variable)
        self._case = case
        self._suffix = suffix
        self._byteorder = byteorder
        self._frame_count = self._count_frames()
        self._cadence = case.parameters.movie_dt
        self._stdout = case.index.segments[suffix].stdout
        self._times: tuple[float, ...] | None = None

    def _count_frames(self) -> int:
        return self._case.movie_frame_count(self.variable.storage_name, self._suffix)

    @property
    def suffix(self) -> str:
        return self._suffix

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def cadence(self) -> float | None:
        """Nominal dt*n_movieout, never an absolute start time."""
        return self._cadence

    @property
    def times(self) -> tuple[float, ...]:
        """Validated stdout times in frame order; immutable, cached on success."""
        if self._times is None:
            if self._stdout is None:
                raise TimeMetadataError(f"No p3d.stdout.{self.suffix}; absolute movie times unavailable")
            times = read_movie_times(self._stdout)
            validate_movie_times(times, self.frame_count, self.cadence,
                                 variable=self.variable.storage_name)
            self._times = times
        return self._times

    def read_frame(self, frame_index: int) -> "np.ndarray":
        """Delegate to the single generic decoder."""
        return self._case.read_movie_frame(self.variable.storage_name, self.suffix, frame_index,
                                          byteorder=self._byteorder)

    def frame_index_at_time(self, time: float) -> int:
        """Find a stored time within floating-point tolerance or raise KeyError."""
        time = float(time)
        if not math.isfinite(time):
            raise ValueError("Requested movie time must be finite")
        times = self.times
        tolerance = min(1e-6, min((b - a for a, b in zip(times, times[1:])), default=math.inf) / 1000)
        matches = [index for index, stored in enumerate(times) if abs(stored - time) <= tolerance]
        if not matches:
            raise KeyError(f"No movie frame at time {time} in segment {self.suffix} (atol={tolerance})")
        if len(matches) != 1:
            raise ValueError(f"Ambiguous movie time {time} in segment {self.suffix}")
        return matches[0]

    def read_time(self, time: float) -> "np.ndarray":
        """Read the frame matching a stored physical time; no interpolation."""
        return self.read_frame(self.frame_index_at_time(time))

    def read_frame_xarray(self, frame_index: int) -> "xr.DataArray":
        """Materialize one local frame with scalar stdout time and axis labels.

        x/y/z have no coordinate values or implied units/staggering. Use isel
        for spatial selection. Valid time metadata is required even for an
        index-based xarray read; the NumPy read_frame API remains independent.
        """
        import xarray as xr
        from .movie import _frame_index

        frame_index = _frame_index(frame_index, self.frame_count)
        time = self.times[frame_index]
        schema = movie_format(self._case.parameters)
        values = self.read_frame(frame_index)
        return xr.DataArray(
            values,
            dims=("x", "y", "z")[:values.ndim],
            coords={"time": time},
            name=self.variable.storage_name,
            attrs={
                "storage_name": self.variable.storage_name,
                "movie_header": schema.header,
                "encoding": "double_byte",
                "source_segment": self.suffix,
                "local_frame_index": frame_index,
                "storage_order": "Fortran: x fastest, then y, then z, then frame",
            },
        )

    def read_time_xarray(self, time: float) -> "xr.DataArray":
        """Wrap exactly one matching frame; time coordinate is the stored time."""
        return self.read_frame_xarray(self.frame_index_at_time(time))


class BxSegment(MovieSegment):
    """Original 2D Bx interface, sharing all generic time validation."""

    def __init__(self, case: "KGlobalCase", suffix: str, *, byteorder: str):
        super().__init__(case, "bx", suffix, byteorder=byteorder)

    def _count_frames(self) -> int:
        return self._case.bx_frame_count(self._suffix)

    def read_frame(self, frame_index: int) -> "np.ndarray":
        return self._case.read_bx_frame(self.suffix, frame_index, byteorder=self._byteorder)
