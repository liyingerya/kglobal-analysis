"""Explicit index selection and bounded time composition; no derived physics."""

from copy import deepcopy
import operator

import numpy as np
import xarray as xr


def _index(value, size, name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f'{name} must be an integer index')
    value = operator.index(value)
    if not 0 <= value < size:
        raise IndexError(f'{name} index {value} outside [0, {size})')
    return value


def index_cut(frame: xr.DataArray, *, line_axis: str, fixed_indices: dict) -> xr.DataArray:
    """Retain one named axis; explicitly index every other axis, without wrapping.

    A time dimension must first be selected; scalar time/provenance is retained.
    Selection stays lazy. No implicit sheet center or interpolation is applied.
    """
    if not isinstance(frame, xr.DataArray) or 'time' in frame.dims:
        raise ValueError('Supply a DataArray frame with no time dimension')
    if line_axis not in frame.dims or set(fixed_indices) != set(frame.dims) - {line_axis}:
        raise ValueError('Specify exactly one fixed index for every axis except line_axis')
    selected = {axis: _index(value, frame.sizes[axis], axis)
                for axis, value in fixed_indices.items()}
    result = frame.isel(selected).copy(deep=False)
    result.attrs = deepcopy(frame.attrs)
    result.attrs.update(line_axis=line_axis, fixed_indices=selected)
    return result


def _movie_input(data, variable=None, contour_variable=None):
    if not isinstance(data, (xr.DataArray, xr.Dataset)):
        from .series import MovieSeries
        if not isinstance(data, MovieSeries):
            raise TypeError('Expected MovieSeries, time DataArray, or aligned Dataset')
        data = data.to_xarray()
    if isinstance(data, xr.Dataset):
        if variable is None or variable not in data.data_vars:
            raise ValueError('Select a Dataset variable explicitly')
        names = [variable]
        if contour_variable is not None:
            if contour_variable not in data.data_vars:
                raise ValueError('Unknown contour_variable')
            names.append(contour_variable)
        data = data[list(dict.fromkeys(names))]
        base = data[variable]
        if contour_variable is not None:
            other = data[contour_variable]
            if other.dims != base.dims or other.shape != base.shape:
                raise ValueError('Dataset contour dimensions must match the selected variable')
    else:
        if variable is not None or contour_variable is not None:
            raise ValueError('variable/contour_variable require an aligned Dataset')
        base = data
    if 'time' not in base.dims or 'time' not in base.coords or base.time.dims != ('time',):
        raise ValueError('An explicit one-dimensional actual time coordinate is required')
    times = np.asarray(base.time.values, dtype=float)
    if not len(times) or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError('Actual times must be finite, nonempty and strictly increasing')
    arrays = data.data_vars.values() if isinstance(data, xr.Dataset) else [data]
    for array in arrays:
        if array.chunks is not None:
            if any(size != 1 for size in array.chunksizes['time']):
                raise ValueError('Bounded execution requires one event per time chunk; use the movie reader layout')
    return data, times


def _event_indices(times, selected_times=None, stride=1):
    if isinstance(stride, (bool, np.bool_)) or operator.index(stride) <= 0:
        raise ValueError('stride must be a positive integer')
    if selected_times is None:
        return tuple(range(0, len(times), stride))
    tolerance = min(1e-6, float(np.min(np.diff(times))) / 1000 if len(times) > 1 else 1e-6)
    indices = []
    for requested in selected_times:
        requested = float(requested)
        matches = np.flatnonzero(np.abs(times - requested) <= tolerance)
        if not np.isfinite(requested) or len(matches) != 1:
            raise KeyError(f'No unique stored event at {requested}; no nearest-time fallback')
        indices.append(int(matches[0]))
    if not indices or len(indices) != len(set(indices)):
        raise ValueError('Select a nonempty set of distinct events')
    return tuple(sorted(indices)[::stride])


def spacetime(data, *, line_axis: str, fixed_indices: dict,
              times=None, stride=1, variable=None) -> xr.DataArray:
    """Eager small (time, line_axis) matrix, built one event at a time.

    Current movie tasks still decode whole frames. Only detached lines survive
    each iteration; no full-series compute or concurrent frame prefetch occurs.
    """
    source, actual = _movie_input(data, variable)
    if isinstance(source, xr.Dataset):
        source = source[variable]
    events = _event_indices(actual, times, stride)
    # Validate selection before computing a sample.
    template = index_cut(source.isel(time=events[0]), line_axis=line_axis,
                         fixed_indices=fixed_indices)
    lines = []
    for event in events:
        line = index_cut(source.isel(time=event), line_axis=line_axis,
                         fixed_indices=fixed_indices).compute(scheduler='synchronous')
        lines.append(line.copy(deep=True))  # Never retain a view into a whole frame.
        del line
    result = xr.concat(lines, dim='time', join='exact', coords='different', compat='equals').transpose('time', line_axis)
    result.attrs = deepcopy(template.attrs)
    result.attrs['execution'] = 'one event at a time; source movie tasks decode whole frames'
    return result
