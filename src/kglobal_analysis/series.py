"""A validated global movie timeline with lazy, one-frame reads."""

from typing import TYPE_CHECKING
import math

from .format import movie_format
from .times import TimeMetadataError, validate_movie_times

if TYPE_CHECKING:
    import numpy as np
    import xarray as xr
    from .case import KGlobalCase


class MovieSeries:
    """Combine every segment of one supported storage variable in a case, ordered by actual stdout times.

    Construction reads stdout metadata and binary file sizes, never samples.
    Multiple segments require nominal cadence to validate their boundaries.
    No overlaps, duplicate times, gaps, or cadence discontinuities are repaired.
    """

    def __init__(self, case: "KGlobalCase", variable: str, *, byteorder: str):
        if byteorder not in ("little", "big"):
            raise ValueError("byteorder must be explicitly 'little' or 'big'")
        self.variable = movie_format(case.parameters).variable(variable)
        label = "Bx" if variable == "bx" else variable
        segments = []
        for suffix, entry in case.index.segments.items():
            if self.variable.storage_name not in entry.movies:
                continue
            segment = self._make_segment(case, suffix, byteorder=byteorder)
            try:
                _ = segment.times
            except TimeMetadataError as error:
                raise TimeMetadataError(f"{label} segment {suffix}: {error}") from error
            segments.append(segment)
        if not segments:
            raise ValueError(f"No {label} segments available in this case")
        segments.sort(key=lambda segment: segment.times[0])
        cadence = case.parameters.movie_dt
        if len(segments) > 1 and cadence is None:
            raise TimeMetadataError(f"Resolved dt*n_movieout is required to validate {label} segment boundaries")
        for previous, following in zip(segments, segments[1:]):
            left, right = previous.times[-1], following.times[0]
            boundary = f"{previous.suffix} -> {following.suffix} ({left} -> {right})"
            if right <= left:
                raise TimeMetadataError(f"Overlapping segments or duplicate physical times at {boundary}")
            spacing = right - left
            if not math.isclose(spacing, cadence, rel_tol=1e-6, abs_tol=1e-10):
                reason = "Gap" if spacing > cadence else "Cadence discontinuity"
                raise TimeMetadataError(f"{reason} at {boundary}: spacing {spacing}, expected {cadence}")
        self._segments = {segment.suffix: segment for segment in segments}
        self._times = tuple(time for segment in segments for time in segment.times)
        self._locations = tuple((segment.suffix, index) for segment in segments
                                for index in range(segment.frame_count))
        validate_movie_times(self._times, sum(segment.frame_count for segment in segments), cadence,
                             variable=self.variable.storage_name)
        self._tolerance = min(1e-6, min((b - a for a, b in zip(self._times, self._times[1:])),
                                     default=math.inf) / 1000)

    def _make_segment(self, case, suffix: str, *, byteorder: str):
        return case.movie_segment(self.variable.storage_name, suffix, byteorder=byteorder)

    @property
    def suffixes(self) -> tuple[str, ...]:
        """Participating variable suffixes in physical-time order."""
        return tuple(self._segments)

    @property
    def frame_count(self) -> int:
        return len(self._times)

    @property
    def times(self) -> tuple[float, ...]:
        """Actual stdout times, unchanged and globally strictly increasing."""
        return self._times

    def locate_time(self, time: float) -> tuple[str, int]:
        """Return (suffix, local frame index) for a stored physical time.

        Uses the Milestone 3 equality rule across the global times: absolute
        tolerance min(1e-6, minimum spacing/1000), with zero relative tolerance.
        """
        time = float(time)
        if not math.isfinite(time):
            raise ValueError("Requested movie time must be finite")
        matches = [index for index, stored in enumerate(self.times)
                   if abs(stored - time) <= self._tolerance]
        if not matches:
            raise KeyError(f"No {self.variable.storage_name} frame at time {time} (atol={self._tolerance})")
        if len(matches) != 1:
            raise ValueError(f"Ambiguous {self.variable.storage_name} time {time}")
        return self._locations[matches[0]]

    def read_time(self, time: float) -> "np.ndarray":
        """Read only the located segment's one local frame via MovieSegment."""
        suffix, index = self.locate_time(time)
        return self._segments[suffix].read_frame(index)

    def read_frame_xarray(self, global_frame_index: int) -> "xr.DataArray":
        """Materialize one frame indexed in validated physical-time order.

        The result has a scalar time coordinate, not a time dimension. No
        other frames are read and no physical spatial coordinates are inferred.
        """
        from .movie import _frame_index

        global_frame_index = _frame_index(global_frame_index, self.frame_count)
        suffix, local_index = self._locations[global_frame_index]
        frame = self._segments[suffix].read_frame_xarray(local_index)
        frame.attrs["global_frame_index"] = global_frame_index
        return frame

    def read_time_xarray(self, time: float) -> "xr.DataArray":
        """Materialize one exact-time match using the existing lookup tolerance."""
        location = self.locate_time(time)
        return self.read_frame_xarray(self._locations.index(location))


class BxSeries(MovieSeries):
    """Original Bx entry point, sharing the generic timeline implementation."""

    def __init__(self, case: "KGlobalCase", *, byteorder: str):
        super().__init__(case, "bx", byteorder=byteorder)

    def _make_segment(self, case, suffix: str, *, byteorder: str):
        return case.bx_segment(suffix, byteorder=byteorder)
