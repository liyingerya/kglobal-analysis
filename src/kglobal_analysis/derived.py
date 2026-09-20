"""Verified particle moments on prealigned Datasets, independent of rendering.

The supported profile describes the current unit-charge electron/ion producer,
not arbitrary historical executables. Relativistic raw moments are supported;
exact parallel thermal centering and tension require a nonrelativistic profile.
"""

from dataclasses import dataclass
from copy import deepcopy
import math
import re
from typing import ClassVar

import numpy as np
import xarray as xr

from .format import movie_format
from .parameters import Parameters, _unparenthesize

__all__ = [
    'ParticleMomentProfile', 'DerivedQuantityError', 'DerivedSemanticsError',
    'DerivedAlignmentError', 'particle_number_density', 'raw_parallel_stress',
    'particle_perpendicular_pressure', 'particle_perpendicular_temperature',
    'particle_parallel_pressure', 'particle_parallel_temperature',
    'particle_total_temperature', 'firehose_parameter', 'validity_mask',
]


class DerivedQuantityError(ValueError):
    """Invalid derived-quantity request."""


class DerivedSemanticsError(DerivedQuantityError):
    """Missing, conflicting, or unsupported producer semantics."""


class DerivedAlignmentError(DerivedQuantityError):
    """Required inputs do not share observable coordinates/provenance."""


# Storage identity is centralized here, separate from result labels/formulas.
# A future profile can extend population/species keys without changing decoding.
_STORAGE = (
    ('number_density', 'ion', 'fluid', 'ni'),
    ('number_density', 'ion', 'particle', 'nih'),
    ('number_density', 'electron', 'particle', 'neh'),
    ('raw_parallel_stress', 'ion', 'particle', 'pihpar'),
    ('raw_parallel_stress', 'electron', 'particle', 'pehpar'),
    ('perpendicular_pressure', 'ion', 'particle', 'pihperp'),
    ('perpendicular_pressure', 'electron', 'particle', 'pehperp'),
    ('parallel_current', 'ion', 'particle', 'jihpar'),
    ('parallel_current', 'electron', 'particle', 'jhpar'),
    ('magnetic_x', 'field', 'field', 'bx'),
    ('magnetic_y', 'field', 'field', 'by'),
    ('magnetic_z', 'field', 'field', 'bz'),
)


@dataclass(frozen=True)
class ParticleMomentProfile:
    """Explicit declaration of the supported producer's scientific contract.

    Direct construction is useful for documented synthetic/external data; all
    fields are validated. For a case use ``case.particle_moment_profile()``.
    CPP flags have #ifdef semantics: a present definition, even '0', is enabled.
    Absence means disabled only within the supplied complete parameter contract.
    Metadata cannot certify an arbitrary executable or unrecorded overrides.
    """

    electron_mass_ratio: float
    relativistic: bool
    smooth_part_mom: bool = False
    doublesmooth: bool = False
    testmovie: bool = False
    producer_id: str = 'kglobal-standard-electron-ion-v1'
    parameter_source: str = 'explicit declaration'
    storage_mapping: ClassVar[tuple] = _STORAGE

    def __post_init__(self):
        if self.producer_id != 'kglobal-standard-electron-ion-v1':
            raise DerivedSemanticsError(f'Unsupported semantic profile: {self.producer_id!r}')
        mass = self.electron_mass_ratio
        if isinstance(mass, (bool, str)) or not isinstance(mass, (int, float, np.number)):
            raise DerivedSemanticsError('electron_mass_ratio must be a positive finite number')
        if not np.isrealobj(mass) or not math.isfinite(mass) or mass <= 0:
            raise DerivedSemanticsError('electron_mass_ratio must be a positive finite number')
        object.__setattr__(self, 'electron_mass_ratio', float(mass))
        for name in ('relativistic', 'smooth_part_mom', 'doublesmooth', 'testmovie'):
            if type(getattr(self, name)) is not bool:
                raise DerivedSemanticsError(f'{name} must be an explicit bool')
        if not isinstance(self.parameter_source, str) or not self.parameter_source:
            raise DerivedSemanticsError('parameter_source must describe the metadata source')

    @classmethod
    def from_parameters(cls, parameters: Parameters, *, parameter_source='parsed parameters'):
        """Resolve literal mass and flags; never evaluate CPP expressions.

        The caller declares these parameters describe the standard producer.
        Unknown headers, layout variants and conflicting typed/raw metadata fail.
        testmovie is recorded: it changes pc, which these operations never use.
        """
        if not isinstance(parameters, Parameters):
            raise DerivedSemanticsError('Expected parsed Parameters')
        try:
            movie_format(parameters)
        except ValueError as error:
            raise DerivedSemanticsError(str(error)) from error
        definitions = parameters.definitions
        if definitions.get('movie_header') != f'"{parameters.movie_header}"':
            raise DerivedSemanticsError('Missing or conflicting movie_header metadata')
        for name in ('double_byte', 'four_byte'):
            if (name in definitions) != getattr(parameters, name):
                raise DerivedSemanticsError(f'Conflicting {name} metadata')
        literal = definitions.get('m_e')
        if not isinstance(literal, str):
            raise DerivedSemanticsError('Missing literal electron mass m_e')
        literal = _unparenthesize(literal.strip())
        if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?', literal):
            raise DerivedSemanticsError('m_e must be a numeric literal, not a CPP expression')
        return cls(float(re.sub('[dD]', 'e', literal)),
                   **{flag: flag in definitions for flag in
                      ('relativistic', 'smooth_part_mom', 'doublesmooth', 'testmovie')},
                   parameter_source=str(parameter_source))

    def storage_for(self, quantity: str, *, species: str, population='particle') -> str:
        """Resolve a scientific role to an immutable historical storage name."""
        for role, sp, pop, storage in self.storage_mapping:
            if (role, sp, pop) == (quantity, species, population):
                return storage
        raise DerivedSemanticsError(f'Unsupported mapping: {species}/{population}/{quantity}')

    def mass_charge(self, species: str) -> tuple[float, int]:
        """Current profile masses/reference mass and signed unit charges."""
        if species == 'electron':
            return self.electron_mass_ratio, -1
        if species == 'ion':
            return 1., 1
        raise DerivedSemanticsError(f'Unsupported species: {species!r}')

    @property
    def smoothing_passes(self) -> int:
        return (2 if self.doublesmooth else 1) if self.smooth_part_mom else 0


def _profile(profile, *, nr=False):
    if not isinstance(profile, ParticleMomentProfile):
        raise DerivedSemanticsError('Supply an explicit ParticleMomentProfile')
    if nr and profile.relativistic:
        raise DerivedSemanticsError('Exact parallel thermal moments/tension require a '
                                    'nonrelativistic profile; no relativistic fallback')


def _metadata_equal(left, right):
    if hasattr(left, '__dask_graph__') or hasattr(right, '__dask_graph__'):
        raise DerivedAlignmentError('Provenance stamps must be eager metadata')
    try:
        return np.array_equal(np.asarray(left), np.asarray(right))
    except (TypeError, ValueError):
        return False


def _inputs(ds, profile, requests):
    """Validate metadata only. Never align, broadcast, or compute sample arrays."""
    _profile(profile)
    if not isinstance(ds, xr.Dataset):
        raise DerivedAlignmentError('Supply an already strictly aligned xarray.Dataset')
    names = tuple(profile.storage_for(role, species=sp, population=pop)
                  for role, sp, pop in requests)
    if any(name not in ds.data_vars for name in names):
        raise DerivedAlignmentError(f'Missing required variables: {set(names) - set(ds.data_vars)}')
    arrays = [ds[name] for name in names]
    reference = arrays[0]
    for name, array in zip(names, arrays):
        if array.dims != reference.dims or array.shape != reference.shape:
            raise DerivedAlignmentError(f'Dimensions do not match for {name}')
        if set(array.coords) != set(reference.coords):
            raise DerivedAlignmentError(f'Coordinates do not match for {name}')
        if array.dtype.kind not in 'iuf':
            raise DerivedQuantityError(f'{name} must contain real numeric moments')
        if array.attrs.get('storage_name', name) != name:
            raise DerivedSemanticsError(f'Conflicting storage_name for {name}')
        for coord in array.coords:
            current, expected = array.coords[coord], reference.coords[coord]
            if current.chunks is not None or expected.chunks is not None:
                raise DerivedAlignmentError('Coordinates must be eager metadata, not sample tasks')
            if not current.identical(expected):
                raise DerivedAlignmentError(f'Coordinate mismatch: {coord}')
    for array, (role, species, population) in zip(arrays, requests):
        identity = dict(scientific_quantity=role, species=species, population=population)
        if population == 'particle':
            identity['mass_ratio'] = profile.mass_charge(species)[0]
        if role == 'raw_parallel_stress':
            identity['centered'] = False
        for key, expected in identity.items():
            if key in array.attrs and not _metadata_equal(array.attrs[key], expected):
                raise DerivedSemanticsError(f'Conflicting {key} on {array.name}')
    # Datasets share coordinate indexes. Retained per-variable coordinate stamps
    # can expose a conflict from external prior alignment; lost history cannot.
    for key in set(reference.coords) | set(reference.dims) | {'time', 'source_segment', 'local_frame_index', 'region', 'selection', 'fixed_indices'}:
        stamps = [array.attrs[key] for array in arrays if key in array.attrs]
        if stamps:
            if len(stamps) != len(arrays) or any(not _metadata_equal(stamps[0], v) for v in stamps[1:]):
                raise DerivedAlignmentError(f'Conflicting per-variable {key} provenance')
            if key in reference.coords and not _metadata_equal(stamps[0], reference[key].values):
                raise DerivedAlignmentError(f'{key} provenance conflicts with Dataset coordinates')
    if 'time' in reference.dims and 'time' not in reference.coords:
        raise DerivedAlignmentError('A time dimension requires actual time coordinates')
    if 'time' in reference.coords:
        time = reference.time
        if time.dims not in ((), ('time',)) or time.dtype.kind not in 'iuf':
            raise DerivedAlignmentError('Actual time must be scalar or one-dimensional numeric metadata')
        values = np.asarray(time.values)
        if not values.size or not np.isfinite(values).all() or (values.ndim and np.any(np.diff(values) <= 0)):
            raise DerivedAlignmentError('Actual times must be finite and strictly increasing')
    for key in ('source_segment', 'local_frame_index'):
        if key in reference.coords:
            coord = reference[key]
            if coord.dims != (('time',) if 'time' in reference.dims else ()):
                raise DerivedAlignmentError(f'{key} must follow actual event dimensions')
            values = np.asarray(coord.values)
            if key == 'local_frame_index':
                if values.dtype.kind not in 'iu' or np.any(values < 0):
                    raise DerivedAlignmentError('Invalid local_frame_index provenance')
            elif any(not isinstance(v, str) or not v for v in values.ravel().tolist()):
                raise DerivedAlignmentError('Invalid source_segment provenance')
    contract = dict(producer_id=profile.producer_id, relativistic=profile.relativistic,
                    electron_mass_ratio=profile.electron_mass_ratio,
                    smooth_part_mom=profile.smooth_part_mom, doublesmooth=profile.doublesmooth,
                    testmovie=profile.testmovie, units_status='code_normalized',
                    movie_header='movie_kglobal3.0.h')
    for attrs in [ds.attrs] + [array.attrs for array in arrays]:
        for key, expected in contract.items():
            if key in attrs and not _metadata_equal(attrs[key], expected):
                raise DerivedSemanticsError(f'Profile conflicts with input {key}')
    return tuple(array.astype(np.float64) for array in arrays), names


def _requests(species, *roles):
    return tuple((role, species, 'particle') for role in roles)


def _finish(value, valid, *, context, profile, species, quantity, names, formula, convention,
            units, **attrs):
    result = value.where(valid & np.isfinite(value)).rename(f'particle_{species}_{quantity}')
    # Copy selection/region provenance without inheriting a misleading raw label.
    result.attrs = deepcopy(context.attrs)
    result.attrs.update(deepcopy(context[names[0]].attrs))
    for key in ('storage_name', 'units', 'centered', 'drift_subtraction', 'mass_ratio'):
        result.attrs.pop(key, None)
    result.attrs.update(scientific_quantity=quantity, species=species, population='particle',
                        source_variables=tuple(names), source_storage_names=tuple(names),
                        formula=formula, convention=convention,
                        moment_model='gamma_weighted' if profile.relativistic else 'nonrelativistic',
                        units_status='code_normalized', normalization=units,
                        producer_id=profile.producer_id, parameter_source=profile.parameter_source,
                        relativistic=profile.relativistic, smooth_part_mom=profile.smooth_part_mom,
                        doublesmooth=profile.doublesmooth, smoothing_passes=profile.smoothing_passes,
                        testmovie=profile.testmovie, approximation='none',
                        invalid_policy='NaN for invalid inputs/arithmetic; no clipping or positive floors',
                        **attrs)
    if species in ('ion', 'electron'):
        mass, charge = profile.mass_charge(species)
        result.attrs.update(mass_ratio=mass, charge_convention=f'J_parallel = {charge:+d} * n * mean(v_parallel)')
    return result


def validity_mask(result: xr.DataArray) -> xr.DataArray:
    """Lazy validity mask for a derived output; finite negative tension is valid."""
    if not isinstance(result, xr.DataArray) or 'invalid_policy' not in result.attrs:
        raise DerivedQuantityError('Expected a derived DataArray with invalid_policy')
    mask = np.isfinite(result).rename(f'{result.name}_valid')
    mask.attrs = {'invalid_policy': result.attrs['invalid_policy'], 'meaning': 'True = valid derived sample'}
    return mask


def _nonnegative(ds, profile, species, role, convention, units):
    (value,), names = _inputs(ds, profile, _requests(species, role))
    return _finish(value, value >= 0, context=ds, profile=profile, species=species, quantity=role,
                   names=names, formula='stored moment', convention=convention, units=units,
                   **({'centered': False} if role == 'raw_parallel_stress' else {}))


def particle_number_density(ds, *, species, profile):
    """Particle population only; zero density is valid. Never includes fluid ni."""
    return _nonnegative(ds, profile, species, 'number_density',
                        'weighted particle number density; no fraction or charge rescaling', 'n/n0')


def raw_parallel_stress(ds, *, species, profile):
    """Uncentered particle p_parallel*v_parallel moment, including gamma weighting."""
    return _nonnegative(ds, profile, species, 'raw_parallel_stress',
                        'uncentered particle p_parallel*v_parallel stress', 'P/(n0*m_reference*C_A0^2)')


def particle_perpendicular_pressure(ds, *, species, profile):
    """Gyrotropic scalar p_perp^2/(2*m*gamma), for either perpendicular direction."""
    return _nonnegative(ds, profile, species, 'perpendicular_pressure',
                        'guiding-center gyrotropic scalar perpendicular pressure', 'P/(n0*m_reference*C_A0^2)')


def particle_perpendicular_temperature(ds, *, species, profile):
    """P_perp/n moment temperature; not a fitted core or Lorentz proper-frame T."""
    (q, n), names = _inputs(ds, profile, _requests(species, 'perpendicular_pressure', 'number_density'))
    value = q / n.where(n > 0)
    return _finish(value, (q >= 0) & (n > 0) & np.isfinite(q) & np.isfinite(n),
                   context=ds, profile=profile, species=species, quantity='perpendicular_temperature', names=names,
                   formula='Q/n', convention='guiding-center pressure-per-number moment temperature',
                   units='temperature energy/(m_reference*C_A0^2)')


def _center_parallel(moment, current, density, mass, charge):
    """NR algebra only; the public caller must enforce the producer contract."""
    return moment - (mass / charge**2) * current**2 / density.where(density > 0)


def particle_parallel_pressure(ds, *, species, profile):
    """Exact centered particle pressure only for a verified nonrelativistic profile."""
    _profile(profile, nr=True)
    (m, j, n), names = _inputs(ds, profile, _requests(
        species, 'raw_parallel_stress', 'parallel_current', 'number_density'))
    mass, charge = profile.mass_charge(species)
    value = _center_parallel(m, j, n, mass, charge)
    valid = (n > 0) & (m >= 0) & (value >= 0) & np.isfinite(n) & np.isfinite(m) & np.isfinite(j)
    return _finish(value, valid, context=ds, profile=profile, species=species, quantity='parallel_pressure',
                   names=names, formula='M - (m/q^2)*J^2/n', convention='nonrelativistic centered particle pressure',
                   units='P/(n0*m_reference*C_A0^2)', centered=True, drift_subtraction='current_based')


def particle_parallel_temperature(ds, *, species, profile):
    """Centered NR parallel pressure divided by particle number density."""
    pressure = particle_parallel_pressure(ds, species=species, profile=profile)
    (n,), _ = _inputs(ds, profile, _requests(species, 'number_density'))
    return _finish(pressure / n.where(n > 0), validity_mask(pressure) & np.isfinite(n) & (n > 0),
                   context=ds, profile=profile, species=species, quantity='parallel_temperature',
                   names=pressure.attrs['source_storage_names'], formula='(M - (m/q^2)*J^2/n)/n',
                   convention='nonrelativistic centered particle moment temperature',
                   units='temperature energy/(m_reference*C_A0^2)', centered=True, drift_subtraction='current_based')


def particle_total_temperature(ds, *, species, profile):
    """NR (T_parallel + 2*T_perp)/3; never an additional division by n."""
    _profile(profile, nr=True)
    _, names = _inputs(ds, profile, _requests(species, 'raw_parallel_stress', 'parallel_current',
                                             'number_density', 'perpendicular_pressure'))
    parallel = particle_parallel_temperature(ds, species=species, profile=profile)
    perpendicular = particle_perpendicular_temperature(ds, species=species, profile=profile)
    return _finish((parallel + 2*perpendicular)/3, validity_mask(parallel) & validity_mask(perpendicular),
                   context=ds, profile=profile, species=species, quantity='total_temperature', names=names,
                   formula='(T_parallel + 2*T_perp)/3', convention='nonrelativistic centered particle moment temperature',
                   units='temperature energy/(m_reference*C_A0^2)', centered=True, drift_subtraction='current_based')


def firehose_parameter(ds, *, profile):
    """Unsquared NR tension from particle electron/ion anisotropy; negative is valid."""
    _profile(profile, nr=True)
    roles = ('raw_parallel_stress', 'parallel_current', 'number_density', 'perpendicular_pressure')
    requests = _requests('electron', *roles) + _requests('ion', *roles)
    requests += tuple((f'magnetic_{axis}', 'field', 'field') for axis in ('x', 'y', 'z'))
    arrays, names = _inputs(ds, profile, requests)
    bx, by, bz = arrays[-3:]
    b2 = bx**2 + by**2 + bz**2
    pressures = [particle_parallel_pressure(ds, species=s, profile=profile) for s in ('electron', 'ion')]
    perps = [particle_perpendicular_pressure(ds, species=s, profile=profile) for s in ('electron', 'ion')]
    anisotropy = pressures[0] - perps[0] + pressures[1] - perps[1]
    valid = np.isfinite(bx) & np.isfinite(by) & np.isfinite(bz) & np.isfinite(b2) & (b2 > 0)
    for field in pressures + perps:
        valid = valid & validity_mask(field)
    return _finish(1 - anisotropy / b2.where(b2 > 0), valid, context=ds, profile=profile, species='electron+ion',
                   quantity='firehose_parameter', names=names,
                   formula='1 - (P_parallel_e - Q_e + P_parallel_i - Q_i)/(Bx^2 + By^2 + Bz^2)',
                   convention='unsquared nonrelativistic tension; particle electron and particle ion anisotropy only',
                   units='dimensionless; P0 = B0^2/(4*pi)', electron_mass_ratio=profile.electron_mass_ratio,
                   charge_convention='electron -1; ion +1', centered=True, drift_subtraction='current_based')
