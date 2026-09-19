"""Bounded event iteration and headless PNG export, without video codecs."""

from dataclasses import dataclass
import gc
from pathlib import Path

import numpy as np
import xarray as xr

from .analysis import _event_indices, _movie_input


@dataclass(frozen=True)
class MovieFrame:
    ordinal: int
    source_index: int
    time: float
    data: xr.DataArray
    contours: xr.DataArray | None = None


@dataclass(frozen=True)
class RenderedFrame:
    path: Path
    time: float
    source_index: int
    vmin: float
    vmax: float


def iter_movie_frames(data, *, times=None, stride=1, variable=None, contour_variable=None):
    """Yield one computed event, retaining no earlier frames.

    A Dataset must already be aligned (prefer case.movie_dataset). This helper
    performs no alignment or interpolation. Consumers must release frames too.
    Time-chunk sizes greater than one are rejected to bound decoder execution.
    """
    source, actual = _movie_input(data, variable, contour_variable)
    for ordinal, index in enumerate(_event_indices(actual, times, stride)):
        event = source.isel(time=index).compute(scheduler='synchronous')
        if isinstance(event, xr.Dataset):
            base = event[variable]
            overlay = event[contour_variable] if contour_variable is not None else None
        else:
            base, overlay = event, None
        yield MovieFrame(ordinal, index, float(actual[index]), base, overlay)
        del event, base, overlay


def render_frame_sequence(data, output_dir, *, horizontal, vertical,
                          times=None, stride=1, variable=None, contour_variable=None,
                          color_policy='fixed', normalization='linear', vmin=None, vmax=None,
                          overwrite=False, dpi=100, figsize=(6.4, 4.8), **map_options):
    """Render deterministic PNGs with fixed/global/per_frame color limits.

    Global uses two bounded passes. Fixed requires both limits; per_frame is
    explicit. Figures use Agg directly, without switching notebook backends.
    Returns compact path/time/limit records, never figures or decoded arrays.
    Existing files are refused unless overwrite=True. A later runtime failure
    may leave earlier completed PNGs; no unrelated destination files are removed.
    """
    from .plotting import _matplotlib, _normalizer, plot_scalar_map
    _matplotlib()
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if color_policy not in ('fixed', 'global', 'per_frame'):
        raise ValueError('color_policy must be fixed, global, or per_frame')
    if color_policy == 'fixed' and (vmin is None or vmax is None):
        raise ValueError('Fixed color policy requires explicit vmin and vmax')
    if color_policy != 'fixed' and (vmin is not None or vmax is not None):
        raise ValueError('Supply limits only for fixed color policy')
    if set(map_options) & {'ax', 'title', 'contours', 'horizontal', 'vertical'}:
        raise ValueError('Exporter owns axes, event title, and synchronized contours')
    source, actual = _movie_input(data, variable, contour_variable)
    events = _event_indices(actual, times, stride)
    base = source[variable] if isinstance(source, xr.Dataset) else source
    spatial = set(base.dims) - {'time'}
    if horizontal == vertical or spatial != {horizontal, vertical}:
        raise ValueError('Explicitly slice to two spatial display axes before exporting')
    # Validate policy before any output directory or movie read.
    if normalization not in ('linear', 'log', 'signed'):
        raise ValueError('normalization must be linear, log, or signed')
    if color_policy == 'fixed':
        _normalizer(np.ma.array([vmin, vmax]), normalization, vmin, vmax)
    output_dir = Path(output_dir)
    paths = tuple(output_dir / f'frame_{i:06d}.png' for i in range(len(events)))
    for path in paths:
        if path.exists() and (not overwrite or not path.is_file()):
            raise FileExistsError(f'Output exists: {path}; set overwrite=True to replace files')
    selected = tuple(actual[i] for i in events)
    options = dict(times=selected, variable=variable, contour_variable=contour_variable)
    if color_policy == 'global':
        low, high = np.inf, -np.inf
        # Do not compute overlays during the extrema pass.
        for frame in iter_movie_frames(source, times=selected, variable=variable):
            values = np.asarray(frame.data.values)
            valid = np.isfinite(values)
            if normalization == 'log':
                valid &= values > 0
            if valid.any():
                low = min(low, float(values[valid].min()))
                high = max(high, float(values[valid].max()))
            del values, valid, frame
        if not np.isfinite(low):
            raise ValueError('No positive finite values in selection' if normalization == 'log'
                             else 'No finite values in selection')
        # Expand constant/signed limits once so every frame has identical limits.
        norm = _normalizer(np.ma.array([low, high]), normalization, None, None)
        vmin, vmax = norm.vmin, norm.vmax
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for frame in iter_movie_frames(source, **options):
        figure = Figure(figsize=figsize)
        FigureCanvasAgg(figure)
        result = None
        try:
            result = plot_scalar_map(
                frame.data, horizontal=horizontal, vertical=vertical,
                ax=figure.subplots(), contours=frame.contours,
                title=f'{frame.data.name or "stored scalar"}, t={frame.time:.17g}',
                normalization=normalization, vmin=vmin, vmax=vmax, **map_options)
            path = paths[frame.ordinal]
            # Exclusive creation enforces overwrite policy even after preflight.
            with path.open('wb' if overwrite else 'xb') as stream:
                figure.savefig(stream, format='png', dpi=dpi)
            records.append(RenderedFrame(path, frame.time, frame.source_index,
                                         float(result.artist.norm.vmin), float(result.artist.norm.vmax)))
        finally:
            figure.clear()
            del result, figure, frame
            gc.collect()  # Release cyclic Matplotlib references before the next event.
    return tuple(records)
