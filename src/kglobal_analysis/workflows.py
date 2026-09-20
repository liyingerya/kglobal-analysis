"""Semantic orchestration of existing science; no decoding or numerical formulas."""
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType

import xarray as xr

from . import derived, flux, geometry

__all__ = ['particle_quantity_catalog', 'particle_quantity_series', 'firehose_series',
           'magnetic_flux_series', 'particle_map_series', 'firehose_map_series',
           'ParticleQuantityAvailability', 'WorkflowError', 'UnknownQuantityError']


class WorkflowError(ValueError):
    """Invalid workflow request."""


class UnknownQuantityError(WorkflowError):
    """Unknown semantic particle quantity; storage names are not quantity keys."""


@dataclass(frozen=True)
class _QuantitySpec:
    roles: tuple[str, ...]
    operation: str
    requires_nr: bool = False


_PARALLEL = ('raw_parallel_stress', 'parallel_current', 'number_density')
_THERMAL = _PARALLEL + ('perpendicular_pressure',)
_QUANTITIES = MappingProxyType({
    'number_density': _QuantitySpec(('number_density',), 'particle_number_density'),
    'raw_parallel_stress': _QuantitySpec(('raw_parallel_stress',), 'raw_parallel_stress'),
    'perpendicular_pressure': _QuantitySpec(('perpendicular_pressure',), 'particle_perpendicular_pressure'),
    'perpendicular_temperature': _QuantitySpec(('perpendicular_pressure', 'number_density'), 'particle_perpendicular_temperature'),
    'parallel_pressure': _QuantitySpec(_PARALLEL, 'particle_parallel_pressure', True),
    'parallel_temperature': _QuantitySpec(_PARALLEL, 'particle_parallel_temperature', True),
    'total_temperature': _QuantitySpec(_THERMAL, 'particle_total_temperature', True),
})
_NR_REASON = 'Exact parallel thermal quantities and firehose require nonrelativistic semantics; no approximation is supported'


@dataclass(frozen=True)
class ParticleQuantityAvailability:
    """Semantic availability only, not a claim that case files exist."""
    key: str
    population: str
    supported: bool
    requires_nr: bool
    reason: str | None


def _profile(profile, *, nr=False):
    if not isinstance(profile, derived.ParticleMomentProfile):
        raise derived.DerivedSemanticsError('Expected ParticleMomentProfile')
    if nr and profile.relativistic:
        raise derived.DerivedSemanticsError(_NR_REASON)
    return profile


def particle_quantity_catalog(profile):
    """List all seven particle keys, including unavailable keys with reasons.

    Firehose is a separate two-species workflow and requires NR semantics.
    This catalog reads no case metadata or movie samples.
    """
    _profile(profile)
    return tuple(ParticleQuantityAvailability(key, 'particle',
                 not (spec.requires_nr and profile.relativistic), spec.requires_nr,
                 _NR_REASON if spec.requires_nr and profile.relativistic else None)
                 for key, spec in _QUANTITIES.items())


def _particle_request(case, quantity, species):
    if not isinstance(quantity, str) or quantity not in _QUANTITIES:
        raise UnknownQuantityError(f'Unknown particle quantity: {quantity!r}')
    spec = _QUANTITIES[quantity]
    profile = _profile(case.particle_moment_profile(), nr=spec.requires_nr)
    names = tuple(profile.storage_for(role, species=species, population='particle') for role in spec.roles)
    return profile, spec, names


def _field_names(profile, axes=('x', 'y', 'z')):
    return tuple(profile.storage_for(f'magnetic_{axis}', species='field', population='field')
                 for axis in axes)


def _firehose_request(case):
    profile = _profile(case.particle_moment_profile(), nr=True)
    names = tuple(profile.storage_for(role, species=species, population='particle')
                  for species in ('electron', 'ion') for role in _THERMAL)
    return profile, names + _field_names(profile)


def particle_quantity_series(case, quantity, *, species, byteorder='little'):
    """Return the existing M14 quantity lazily; no geometry requirement."""
    profile, spec, names = _particle_request(case, quantity, species)
    ds = case.movie_dataset(names, byteorder=byteorder)
    return getattr(derived, spec.operation)(ds, species=species, profile=profile)


def firehose_series(case, *, byteorder='little'):
    """Return M14's NR two-species tension using one strict source Dataset."""
    profile, names = _firehose_request(case)
    ds = case.movie_dataset(names, byteorder=byteorder)
    return derived.firehose_parameter(ds, profile=profile)


def magnetic_flux_series(case, *, byteorder='little'):
    """Compose the M15 Bx/By input contract without particle-profile requirements."""
    geom = case.movie_geometry()
    ds = geometry.attach_movie_geometry(case.movie_dataset(('bx', 'by'), byteorder=byteorder), geom)
    return flux.magnetic_flux_2d(ds, geometry=geom)


def _map_source(case, names, profile, include_flux, byteorder):
    if type(include_flux) is not bool:
        raise WorkflowError('include_flux must be a bool')
    geom = case.movie_geometry()
    # M15's current flux interface consumes bx/by. Particle/field scientific
    # dependencies above are resolved by the semantic profile, never duplicated.
    required = tuple(dict.fromkeys(names + (_field_names(profile, ('x', 'y')) if include_flux else ())))
    ds = case.movie_dataset(required, byteorder=byteorder)
    return geometry.attach_movie_geometry(ds, geom), geom


def _map_result(ds, field, geom, profile, quantity, species, include_flux):
    arrays = {'field': field}
    if include_flux:
        arrays['psi'] = flux.magnetic_flux_2d(ds, geometry=geom)
    # Both outputs originate from the same strict source. Enforce equality before
    # Dataset construction: never let xarray repair/join different coverage.
    for array in arrays.values():
        if array.dims != field.dims or set(array.coords) != set(field.coords):
            raise derived.DerivedAlignmentError('Workflow outputs have different dimensions/coordinates')
        for key in field.coords:
            if not array[key].identical(field[key]):
                raise derived.DerivedAlignmentError(f'Workflow coordinate conflict: {key}')
    result = xr.Dataset({name: array.variable for name, array in arrays.items()},
                        coords=field.coords)
    result.attrs = deepcopy(ds.attrs)
    result.attrs.update(workflow_kind='scientific_particle_map', requested_quantity=quantity,
                        species=species, include_flux=include_flux, producer_id=profile.producer_id)
    return result


def particle_map_series(case, quantity, *, species, include_flux=True, byteorder='little'):
    """Lazy field and optional psi from ONE aligned source on M15's 2D geometry."""
    profile, spec, names = _particle_request(case, quantity, species)
    ds, geom = _map_source(case, names, profile, include_flux, byteorder)
    field = getattr(derived, spec.operation)(ds, species=species, profile=profile)
    return _map_result(ds, field, geom, profile, quantity, species, include_flux)


def firehose_map_series(case, *, include_flux=True, byteorder='little'):
    """NR firehose and optional psi share Bx/By tasks; no persistence/cache."""
    profile, names = _firehose_request(case)
    ds, geom = _map_source(case, names, profile, include_flux, byteorder)
    field = derived.firehose_parameter(ds, profile=profile)
    return _map_result(ds, field, geom, profile, 'firehose_parameter', 'electron+ion', include_flux)
