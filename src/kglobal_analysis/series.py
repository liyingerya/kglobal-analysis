"""A validated global Bx timeline with lazy, one-frame reads."""

from typing import TYPE_CHECKING
import math

from .times import TimeMetadataError, validate_movie_times

if TYPE_CHECKING:
    import numpy as np
    from .case import KGlobalCase


class BxSeries:
    """Combine every Bx segment in a case, ordered by actual stdout times.

    Construction reads stdout metadata and binary file sizes, never samples.
    Multiple segments require nominal cadence to validate their boundaries.
    No overlaps, duplicate times, gaps, or cadence discontinuities are repaired.
    """

    def __init__(self, case: "KGlobalCase", *, byteorder: str):
        if byteorder not in ("little", "big"):
            raise ValueError("byteorder must be explicitly 'little' or 'big'")
        segments = []
        for suffix, entry in case.index.segments.items():
            if "bx" not in entry.movies:
                continue
            segment = case.bx_segment(suffix, byteorder=byteorder)
            try:
                _ = segment.times
            except TimeMetadataError as error:
                raise TimeMetadataError(f"Bx segment {suffix}: {error}") from error
            segments.append(segment)
        if not segments:
            raise ValueError("No Bx segments available in this case")
        segments.sort(key=lambda segment: segment.times[0])
        cadence = case.parameters.movie_dt
        if len(segments) > 1 and cadence is None:
            raise TimeMetadataError("Resolved dt*n_movieout is required to validate Bx segment boundaries")
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
        validate_movie_times(self._times, sum(segment.frame_count for segment in segments), cadence)
        self._tolerance = min(1e-6, min((b - a for a, b in zip(self._times, self._times[1:])),
                                     default=math.inf) / 1000)

    @property
    def suffixes(self) -> tuple[str, ...]:
        """Participating Bx suffixes in physical-time order."""
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
            raise KeyError(f"No Bx frame at time {time} (atol={self._tolerance})")
        if len(matches) != 1:
            raise ValueError(f"Ambiguous Bx time {time}")
        return self._locations[matches[0]]

    def read_time(self, time: float) -> "np.ndarray":
        """Read only the located segment's one local frame via BxSegment."""
        suffix, index = self.locate_time(time)
        return self._segments[suffix].read_frame(index)
