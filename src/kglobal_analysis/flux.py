"""Periodic 2D centered-difference flux projection; no rendering or particle semantics."""
from copy import deepcopy

import numpy as np
import xarray as xr

from .geometry import GeometryError, MovieGeometry, attach_movie_geometry, _equal


class FluxError(ValueError):
    """Unsupported or misaligned magnetic-flux input."""


def _inputs(data, geometry):
    if not isinstance(data, xr.Dataset) or not {'bx', 'by'} <= set(data.data_vars):
        raise FluxError('Supply a strictly aligned Dataset containing bx and by')
    # Inspect only participating fields, not Bz or other sample arrays.
    ds = attach_movie_geometry(data[['bx', 'by']], geometry)
    bx, by = ds.bx, ds.by
    if bx.dims != by.dims or set(bx.dims) not in ({'x', 'y'}, {'time', 'x', 'y'}):
        raise FluxError('Bx/By must have identical ordered x/y or time/x/y dimensions')
    if set(bx.coords) != set(by.coords):
        raise FluxError('Bx/By coordinates differ')
    for name in bx.coords:
        if bx[name].chunks is not None or by[name].chunks is not None:
            raise FluxError('Coordinates must be eager metadata')
        if not bx[name].identical(by[name]):
            raise FluxError(f'Bx/By coordinate mismatch: {name}')
    for array, role in ((bx, 'magnetic_x'), (by, 'magnetic_y')):
        if array.dtype.kind not in 'iuf':
            raise FluxError('Bx/By must be real numeric arrays')
        if array.attrs.get('storage_name', array.name) != array.name:
            raise FluxError(f'Conflicting storage_name for {array.name}')
        if array.attrs.get('scientific_quantity', role) != role:
            raise FluxError(f'Conflicting scientific_quantity for {array.name}')
        if array.chunks is not None and 'time' in array.dims and any(c != 1 for c in array.chunksizes['time']):
            raise FluxError('Require one-event time chunks; a source task must not load other events')
    for key in set(bx.coords) | set(bx.dims) | {'region', 'selection', 'fixed_indices', 'time', 'source_segment', 'local_frame_index'}:
        stamps = [a.attrs[key] for a in (bx, by) if key in a.attrs]
        if stamps:
            if len(stamps) != 2 or not _equal(*stamps):
                raise FluxError(f'Conflicting per-variable {key} provenance')
            if key in bx.coords and not _equal(stamps[0], bx[key].values):
                raise FluxError(f'{key} provenance conflicts with shared coordinates')
            if key in ds.attrs and not _equal(stamps[0], ds.attrs[key]):
                raise FluxError(f'{key} provenance conflicts with Dataset attrs')
        if key in ds.attrs and key in bx.coords and not _equal(ds.attrs[key], bx[key].values):
            raise FluxError(f'Dataset {key} provenance conflicts with shared coordinates')
    if 'time' in bx.dims and 'time' not in bx.coords:
        raise FluxError('Time dimension requires actual timestamps')
    if 'time' in bx.coords:
        t = bx.time
        if t.dims not in ((), ('time',)) or t.dtype.kind not in 'iuf':
            raise FluxError('Time must be scalar or one-dimensional numeric metadata')
        values = t.values
        if not values.size or not np.isfinite(values).all() or (values.ndim and np.any(np.diff(values) <= 0)):
            raise FluxError('Times must be finite and strictly increasing')
    for key in ('source_segment', 'local_frame_index'):
        if key in bx.coords:
            c = bx[key]
            if c.dims != (('time',) if 'time' in bx.dims else ()):
                raise FluxError(f'{key} must follow event dimensions')
            if key == 'local_frame_index':
                if c.dtype.kind not in 'iu' or np.any(c.values < 0):
                    raise FluxError('Invalid local frame index')
            elif any(not isinstance(v, str) or not v for v in c.values.ravel().tolist()):
                raise FluxError('Invalid segment provenance')
    for attrs in (ds.attrs, bx.attrs, by.attrs):
        if attrs.get('units_status', 'code_normalized') != 'code_normalized':
            raise FluxError('Bx/By must be code-normalized fields')
    return ds, bx.astype(np.float64), by.astype(np.float64)


def _symbol(n, spacing):
    symbol = np.sin(2*np.pi*np.fft.fftfreq(n)) / spacing
    # Exact discrete indices, not sin(pi)'s floating remainder or a threshold.
    symbol[0] = 0.
    if n % 2 == 0:
        symbol[n//2] = 0.
    return symbol


def _plane(bx, by, *, dx, dy, diagnostics):
    """One evaluated plane only, float64 / complex128 FFT algebra."""
    if not np.isfinite(bx).all() or not np.isfinite(by).all():
        invalid = np.full(bx.shape, np.nan, dtype=np.float64)
        return (invalid, invalid.copy()) if diagnostics else invalid
    fx = np.fft.fft2(bx - bx.mean())
    fy = np.fft.fft2(by - by.mean())
    a = _symbol(bx.shape[0], dx)[:, None]
    b = _symbol(bx.shape[1], dy)[None, :]
    denominator = a*a + b*b
    nonnull = denominator != 0
    # A=(Dy,-Dx). Normal equation A* A psi = A* (Bx,By):
    # psi_hat=(conj(Dy)*Bx_hat-conj(Dx)*By_hat)/(a²+b²)
    #        = i*(a*By_hat-b*Bx_hat)/(a²+b²).
    spectrum = np.zeros(bx.shape, dtype=np.complex128)
    np.divide(1j*(a*fy-b*fx), denominator, out=spectrum, where=nonnull)
    periodic = np.fft.ifft2(spectrum).real
    periodic -= periodic.mean()
    if not diagnostics:
        return periodic
    null_x = np.fft.ifft2(np.where(nonnull, 0., fx)).real
    null_y = np.fft.ifft2(np.where(nonnull, 0., fy)).real
    return periodic, np.hypot(null_x, null_y)


def _derivative(array, axis, spacing):
    return (array.roll({axis: -1}, roll_coords=False) -
            array.roll({axis: 1}, roll_coords=False)) / (2*spacing)


def _solution(data, geometry, *, diagnostics=False):
    ds, bx, by = _inputs(data, geometry)
    # Only spatial rechunking. Each time chunk stays one independent event.
    fields = [a.chunk({'x': -1, 'y': -1}) if a.chunks is not None else a for a in (bx, by)]
    count = 2 if diagnostics else 1
    outputs = xr.apply_ufunc(
        _plane, *fields, input_core_dims=[['x', 'y']]*2,
        output_core_dims=[['x', 'y']]*count, vectorize=True,
        dask='parallelized', output_dtypes=[np.float64]*count,
        kwargs=dict(dx=geometry.dx, dy=geometry.dy, diagnostics=diagnostics),
        join='exact')
    periodic, null = outputs if diagnostics else (outputs, None)
    periodic = periodic.transpose(*bx.dims)
    # A single invalid sample invalidates that entire evaluated event.
    valid = np.isfinite(bx).all(('x', 'y')) & np.isfinite(by).all(('x', 'y'))
    mx = bx.mean(('x', 'y'), skipna=False).where(valid)
    my = by.mean(('x', 'y'), skipna=False).where(valid)
    total = periodic + mx*(bx.y-bx.y.mean()) - my*(bx.x-bx.x.mean())
    attrs = deepcopy(ds.attrs)
    attrs.update(deepcopy(bx.attrs))
    for key in ('storage_name', 'units', 'scientific_quantity', 'species', 'population'):
        attrs.pop(key, None)
    attrs.update(scientific_quantity='magnetic_flux_function',
                 convention='Bx=dpsi/dy; By=-dpsi/dx',
                 reconstruction_method='periodic spectral least squares',
                 derivative_operator='centered periodic difference: sin(k*spacing)/spacing',
                 boundary_condition='periodic',
                 gauge='mean(psi_periodic)=0; x_ref=mean(x), y_ref=mean(y)',
                 mean_field_treatment='linear Bx_mean*(y-y_ref)-By_mean*(x-x_ref)',
                 source_variables=('bx', 'by'), units_status='code_normalized_flux',
                 normalization='B0*L0; dimensional scales unspecified',
                 null_mode_policy='zero psi coefficients at joint zero/Nyquist derivative modes',
                 invalid_policy='any nonfinite Bx/By sample invalidates the entire event to NaN')
    total = total.transpose(*bx.dims).rename('psi')
    total.attrs = attrs
    return total, periodic, mx, my, bx, by, valid, null


def magnetic_flux_2d(data: xr.Dataset, *, geometry: MovieGeometry) -> xr.DataArray:
    """Reconstruct 2D psi lazily from exactly aligned bx/by, ignoring bz.

    Full x/y planes are necessary. Lazy time chunks must each contain one event;
    spatial chunks are combined lazily. No computation/persistence is requested.
    Mean in-plane B gives a nonperiodic linear term in the total flux function.
    """
    return _solution(data, geometry)[0]


def magnetic_flux_diagnostics(data: xr.Dataset, *, geometry: MovieGeometry) -> xr.Dataset:
    """Lazy psi, projected B, input-minus-projection residuals and divergences.

    Derivatives act on psi_periodic; mean B is added analytically, avoiding a
    periodic wrap seam in the linear mean-field term. null_magnitude is the
    pointwise magnitude of joint-null B content AFTER mean removal; it is not
    the full residual (which also includes non-solenoidal content).
    """
    total, periodic, mx, my, bx, by, valid, null = _solution(data, geometry, diagnostics=True)
    rx = (_derivative(periodic, 'y', geometry.dy) + mx).transpose(*bx.dims)
    ry = (-_derivative(periodic, 'x', geometry.dx) + my).transpose(*bx.dims)
    ex, ey = (bx-rx).where(valid), (by-ry).where(valid)
    result = xr.Dataset(dict(psi=total, reconstructed_bx=rx, reconstructed_by=ry,
                             residual_bx=ex, residual_by=ey, residual_magnitude=np.hypot(ex, ey),
                             input_divergence=(_derivative(bx, 'x', geometry.dx) +
                                               _derivative(by, 'y', geometry.dy)).where(valid),
                             reconstructed_divergence=_derivative(rx, 'x', geometry.dx) +
                                                       _derivative(ry, 'y', geometry.dy),
                             null_magnitude=null.transpose(*bx.dims)))
    result.attrs = deepcopy(total.attrs)
    result.attrs['scientific_quantity'] = 'magnetic_flux_reconstruction_diagnostics'
    result.attrs.pop('units_status')
    result.attrs.pop('normalization')
    for name in result.data_vars:
        if name != 'psi':
            result[name].attrs = dict(scientific_quantity=name, source_variables=('bx', 'by'),
                                     units_status='code_normalized',
                                     normalization='B0/L0' if 'divergence' in name else 'B0',
                                     invalid_policy=total.attrs['invalid_policy'])
    return result
