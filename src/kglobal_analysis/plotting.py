"""Optional rendering of supplied arrays and audited reduced products only."""

from copy import deepcopy
from dataclasses import dataclass
import importlib

import numpy as np
import xarray as xr


@dataclass
class PlotResult:
    figure: object
    axes: object
    artist: object
    colorbar: object = None
    contours: object = None
    metadata: dict | None = None


def _matplotlib():
    try:
        return importlib.import_module('matplotlib')
    except ImportError as error:
        raise ImportError('Visualization requires the plotting extra: pip install "kglobal-analysis[plot]"') from error


def _axes(ax):
    _matplotlib()
    if ax is not None:
        return ax.figure, ax
    from matplotlib import pyplot as plt
    return plt.subplots()


def _resolved(data):
    if not isinstance(data, xr.DataArray):
        raise TypeError('Supply an xarray.DataArray with explicit dimension names')
    if data.ndim != 2:
        raise ValueError('Map input must have exactly two dimensions; explicitly slice volumes first')
    if data.chunks is not None:
        raise ValueError('Resolve one selected frame before rendering; renderer does not compute lazy data')


def _coordinate(data, axis, coordinates):
    if axis in coordinates:
        values = coordinates[axis]
        label = axis
    elif axis in data.coords:
        if data[axis].dims != (axis,):
            raise ValueError(f'{axis} coordinate must be one-dimensional on its own axis')
        values = data[axis].values
        label = axis
        if data[axis].attrs.get('units'):
            label += f" [{data[axis].attrs['units']}]"
    else:
        return np.arange(data.sizes[axis]), axis + ' index'
    values = np.array(values, dtype=float, copy=True)
    delta = np.diff(values)
    if (values.ndim != 1 or len(values) != data.sizes[axis] or not np.isfinite(values).all()
            or not (np.all(delta > 0) or np.all(delta < 0))):
        raise ValueError(f'{axis} coordinates must be finite, monotonic, 1D and match axis length')
    return values, label


def _aligned(base, other):
    _resolved(other)
    if base.dims != other.dims or base.shape != other.shape:
        raise ValueError('Scalar contours require identical dimension order and shape')
    spatial_coords = lambda array: {
        name for name, coord in array.coords.items() if set(coord.dims) & set(base.dims)
    }
    if spatial_coords(base) != spatial_coords(other):
        raise ValueError('Scalar contour coordinate presence must match')
    for name in spatial_coords(base):
        if (base[name].dims != other[name].dims or
                not np.array_equal(base[name].values, other[name].values) or
                base[name].attrs.get('units') != other[name].attrs.get('units')):
            raise ValueError('Scalar contour coordinates must match exactly')
    if 'time' in base.coords and 'time' in other.coords:
        if base.time.ndim or other.time.ndim or base.time.item() != other.time.item():
            raise ValueError('Scalar contour times must match exactly')


def _display(values, normalization):
    if normalization not in ('linear', 'log', 'signed'):
        raise ValueError('normalization must be linear, log, or signed')
    display = np.ma.masked_invalid(np.array(values, dtype=float, copy=True))
    if normalization == 'log':
        display = np.ma.masked_less_equal(display, 0, copy=False)
    if not display.count():
        raise ValueError('No positive finite values for log display' if normalization == 'log'
                         else 'No finite values to display')
    return display


def _normalizer(display, normalization, vmin, vmax):
    _matplotlib()
    from matplotlib.colors import Normalize, LogNorm, TwoSlopeNorm
    for value in (vmin, vmax):
        if value is not None and not np.isfinite(value):
            raise ValueError('Color limits must be finite')
    low = float(display.min()) if vmin is None else float(vmin)
    high = float(display.max()) if vmax is None else float(vmax)
    if low > high:
        raise ValueError('vmin must not exceed vmax')
    if normalization == 'log' and low <= 0:
        raise ValueError('Log color limits must be positive')
    if normalization == 'signed':
        span = max(abs(low), abs(high)) or 1.
        low = -span if vmin is None else low
        high = span if vmax is None else high
        if not low < 0 < high:
            raise ValueError('Signed limits must straddle zero')
        return TwoSlopeNorm(vcenter=0, vmin=low, vmax=high)
    if low == high:
        # Constant data needs a nondegenerate display interval (not data repair).
        if normalization == 'log':
            low, high = low / 2, high * 2
        else:
            pad = max(abs(low) * .01, .5)
            low, high = low - pad, high + pad
    return (LogNorm if normalization == 'log' else Normalize)(vmin=low, vmax=high)


def plot_scalar_map(data, *, horizontal, vertical, coordinates=None, normalization='linear',
                    vmin=None, vmax=None, cmap=None, ax=None, colorbar=True,
                    title=None, xlabel=None, ylabel=None, colorbar_label=None,
                    contours=None, contour_levels=7, aspect='equal') -> PlotResult:
    """Render resolved 2D data. Presentation rows=vertical, columns=horizontal.

    Coordinates are nominal display centers; generated edges are presentation
    geometry, not scientific integration edges. No data/coordinate mutation.
    Supplied contours are generic scalars, never automatically magnetic flux.
    """
    _resolved(data)
    if horizontal == vertical or {horizontal, vertical} != set(data.dims):
        raise ValueError('Choose both distinct display dimensions explicitly')
    coordinates = {} if coordinates is None else coordinates
    if set(coordinates) - set(data.dims):
        raise ValueError('Unknown coordinate axis')
    x, x_label = _coordinate(data, horizontal, coordinates)
    y, y_label = _coordinate(data, vertical, coordinates)
    if contours is not None:
        _aligned(data, contours)
    values = _display(data.transpose(vertical, horizontal).values, normalization)
    norm = _normalizer(values, normalization, vmin, vmax)
    figure, ax = _axes(ax)
    artist = ax.pcolormesh(x, y, values, shading='nearest', norm=norm,
                          cmap=cmap or ('RdBu_r' if normalization == 'signed' else 'viridis'),
                          rasterized=True)
    overlay = None
    if contours is not None:
        overlay = ax.contour(x, y, np.ma.masked_invalid(contours.transpose(vertical, horizontal).values),
                             levels=contour_levels, colors='black')
    ax.set(xlabel=x_label if xlabel is None else xlabel,
           ylabel=y_label if ylabel is None else ylabel, aspect=aspect)
    if title is not None:
        ax.set_title(title)
    bar = figure.colorbar(artist, ax=ax) if colorbar else None
    if bar is not None:
        bar.set_label(colorbar_label if colorbar_label is not None else
                      str(data.attrs.get('storage_name', data.name or 'stored scalar')))
    metadata = deepcopy(data.attrs)
    metadata.update(display_horizontal=horizontal, display_vertical=vertical,
                    normalization=normalization, display_geometry='centers; presentation edges only',
                    overlay='supplied scalar contours' if contours is not None else None)
    if 'time' in data.coords and data.time.ndim == 0:
        metadata['actual_time'] = data.time.item()
    return PlotResult(figure, ax, artist, bar, overlay, metadata)


def plot_spacetime(data, *, line_axis, **kwargs):
    """Plot line position horizontally and explicit actual time vertically."""
    if set(data.dims) != {'time', line_axis} or 'time' not in data.coords:
        raise ValueError('Expected (time, line_axis) with actual time coordinates')
    kwargs.setdefault('aspect', 'auto')
    return plot_scalar_map(data, horizontal=line_axis, vertical='time', **kwargs)


def plot_energy_spectrum(spectrum, *, ax=None, xscale='linear', yscale='linear',
                         label=None, xlabel='energy (reconstructed center)',
                         ylabel='legacy energy estimator', **line_kwargs):
    """Display active M11 bins, unchanged; no published-spectrum conversion."""
    from .energy import EnergySpectrum
    if not isinstance(spectrum, EnergySpectrum):
        raise TypeError('Expected EnergySpectrum')
    if xscale not in ('linear', 'log') or yscale not in ('linear', 'log'):
        raise ValueError('Use linear or log line scales')
    x, y = np.array(spectrum.centers), np.array(spectrum.values)
    valid = np.isfinite(x) & np.isfinite(y)
    if xscale == 'log':
        valid &= x > 0
    if yscale == 'log':
        valid &= y > 0
    if not valid.any():
        raise ValueError('No valid points for requested line scales')
    figure, ax = _axes(ax)
    artist, = ax.plot(np.ma.array(x, mask=~valid), np.ma.array(y, mask=~valid),
                      label=label, **line_kwargs)
    ax.set(xscale=xscale, yscale=yscale, xlabel=xlabel, ylabel=ylabel)
    return PlotResult(figure, ax, artist, metadata={
        'storage_name': spectrum.storage_name, 'species': spectrum.species,
        'checkpoint_suffix': spectrum.checkpoint_suffix, 'source_path': str(spectrum.source_path),
        'value_semantics': 'legacy energy estimator; unchanged',
    })


def plot_energy_overlay(spectra, *, ax=None, **kwargs):
    """Overlay a user-labelled mapping of spectra; no case-label inference."""
    if not spectra:
        raise ValueError('Supply at least one labelled spectrum')
    _, ax = _axes(ax)
    results = [plot_energy_spectrum(spectrum, ax=ax, label=label, **kwargs)
               for label, spectrum in spectra.items()]
    ax.legend()
    return results


def plot_distribution(distribution, *, coordinate_mode='index', **kwargs):
    """Render all M12 bins as normalized bin mass, with no density conversion."""
    from .distribution import ReducedDistribution
    if not isinstance(distribution, ReducedDistribution):
        raise TypeError('Expected ReducedDistribution')
    if distribution.value_semantics != 'normalized_bin_mass':
        raise ValueError('Only normalized_bin_mass display is supported')
    if coordinate_mode not in ('index', 'centers'):
        raise ValueError('coordinate_mode must be index or centers')
    da = distribution.to_xarray()
    horizontal, vertical = da.dims
    coordinates = {}
    if coordinate_mode == 'centers':
        coordinates = dict(zip(da.dims, distribution.axis_values))
    names = list(distribution.axis_names)
    if distribution.position_axis is not None:
        names[0] = distribution.position_axis
    for key, name in zip(('xlabel', 'ylabel'), names):
        kwargs.setdefault(key, name + (' index' if coordinate_mode == 'index' else ' (nominal center)'))
    kwargs.setdefault('colorbar_label', 'normalized bin mass')
    kwargs.setdefault('aspect', 'auto')
    result = plot_scalar_map(da, horizontal=horizontal, vertical=vertical,
                             coordinates=coordinates, **kwargs)
    result.metadata['coordinate_mode'] = coordinate_mode
    return result
