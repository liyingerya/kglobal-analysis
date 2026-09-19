"""Audited legacy reduced distributions; storage, region, and naming stay separate."""

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping
import math
import re

import numpy as np

from .reduced import distribution_storage, scan_reduced_products

# Provisional names: no storage or metadata parser depends on these strings.
DISTRIBUTION_QUANTITIES = {
    'parperp': 'parallel_perpendicular_velocity_distribution',
    'positionpar': 'position_parallel_velocity_distribution',
}
NORMALIZATION = 'Unit increments for regular weight==1 particles; each producer array divided by its own sum if nonzero; reader does not renormalize'
SELECTION = 'Box selection; par/perp requires accepted parallel and perpendicular indices; position/par requires position and parallel indices only, so is not guaranteed a marginal of par/perp'
RECONSTRUCTION = 'Nominal nearest-index centers from reducer log; velocity k*V_s/200; position origin+r*extent/100; V_i=V_e*(R_i/R_e). Zero and endpoint bins active; endpoint support and floating-point ties are not represented by edges'
_NUMBER = re.compile(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[EeDd][+-]?[0-9]+)?')
_LABELS = {
    'vmax_e/c': 'R_e', 'vmax_i/c': 'R_i',
    '# of bins in v': 'vbin', '# of bins in x': 'xbin', '# of bins in y': 'ybin',
    '(xmax-xmin)*lx': 'x_extent', '(ymax-ymin)*ly': 'y_extent',
    'xmin*lx': 'x_origin', 'ymin*ly': 'y_origin',
    'Max velocity is normalized to': 'V_e',
    'The ROI is between 2 ellipsoids if the following is 1, or in separatrix if 2': 'legacy_roi_mode',
}


class DistributionError(ValueError):
    """Malformed reduced distribution storage."""


class DistributionMetadataError(DistributionError):
    """Missing, ambiguous, or unsupported distribution metadata/region."""


def _number(token):
    if _NUMBER.fullmatch(token) is None:
        raise ValueError(f'Invalid numeric token {token!r}')
    value = float(token.replace('D', 'E').replace('d', 'e'))
    if not math.isfinite(value):
        raise ValueError(f'Nonfinite number {token!r}')
    return value


def _immutable_array(values):
    array = np.asarray(values)
    # Immutable bytes backing prevents callers re-enabling WRITEABLE on a view.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _read_metadata(path):
    found, bounds = {}, {}

    def record(target, key, value):
        if key in target:
            raise ValueError(f'Duplicate metadata label {key}')
        target[key] = value

    try:
        with path.open(encoding='ascii') as stream:
            for line in stream:
                label, separator, token = line.strip().partition(':')
                if separator and label.strip() in _LABELS:
                    record(found, _LABELS[label.strip()], _number(token.strip()))
                # Recognize bounds by their axis labels, not line position.
                for axis in 'xyz':
                    marker = f'{axis}/l{axis}'
                    if marker in line:
                        match = re.fullmatch(r'\s*(.*?)\s*<=\s*' + marker + r'\s*<=\s*(.*?)\s*', line)
                        if match is None:
                            raise ValueError(f'Malformed {axis} bounds')
                        low, high = map(_number, match.groups())
                        if high < low or (axis in 'xy' and high == low):
                            raise ValueError(f'Invalid {axis} bounds')
                        record(bounds, axis.upper() + 'MIN', low)
                        record(bounds, axis.upper() + 'MAX', high)
        missing = set(_LABELS.values()) - found.keys()
        if missing:
            raise ValueError('Missing metadata: ' + ', '.join(sorted(missing)))
        if found['legacy_roi_mode'] != 0:
            raise ValueError(f"Unsupported legacy ROI mode {found['legacy_roi_mode']}; only validated box mode 0 is supported")
        for key, expected in (('vbin', 200), ('xbin', 100), ('ybin', 100)):
            if found[key] != expected:
                raise ValueError(f'Unsupported {key}: expected {expected}')
        for key in ('R_e', 'R_i', 'V_e', 'x_extent', 'y_extent'):
            if found[key] <= 0:
                raise ValueError(f'{key} must be positive')
        found['V_i'] = found['V_e'] * (found['R_i'] / found['R_e'])
        if not math.isfinite(found['V_i']) or found['V_i'] <= 0:
            raise ValueError('Reconstructed ion velocity scale must be finite and positive')
        return found, bounds
    except (OSError, UnicodeError, ValueError) as error:
        raise DistributionMetadataError(f'Invalid distribution metadata in {path}: {error}') from error


def _read_values(path, spec):
    try:
        tokens = path.read_text(encoding='ascii').split()
        expected = math.prod(spec.raw_shape)
        if len(tokens) != expected:
            raise ValueError(f'Expected {expected} scalars, found {len(tokens)}')
        flat = np.array([_number(token) for token in tokens], dtype=np.float64)
        return flat.reshape(spec.raw_shape, order='F')
    except (UnicodeError, ValueError) as error:
        raise DistributionError(f'Invalid distribution data in {path}: {error}') from error


@dataclass(frozen=True, eq=False)
class ReducedDistribution:
    """Immutable bin masses and ordered axes, without inferred source geometry.

    Numeric arrays have immutable backing; xarray adapters are detached copies.
    Axis tuples describe this reduction, not the simulation dimensionality.
    """

    quantity: str
    species: str
    category: str
    storage_name: str
    position_axis: str | None
    checkpoint_suffix: str
    source_path: Path
    log_path: Path
    legacy_roi_mode: int
    region_kind: str
    region_metadata: Mapping[str, float]
    value_semantics: str
    values: np.ndarray
    axis_names: tuple[str, ...]
    axis_indices: tuple[np.ndarray, ...]
    axis_values: tuple[np.ndarray, ...]
    provenance: Mapping[str, str | float]

    def __post_init__(self):
        object.__setattr__(self, 'values', _immutable_array(self.values))
        for name in ('axis_indices', 'axis_values'):
            object.__setattr__(self, name, tuple(_immutable_array(a) for a in getattr(self, name)))
        object.__setattr__(self, 'axis_names', tuple(self.axis_names))
        for name in ('region_metadata', 'provenance'):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    @property
    def shape(self):
        return self.values.shape

    @property
    def value_sum(self):
        """Diagnostic over every stored bin; never used to repair values."""
        return float(self.values.sum())

    def to_xarray(self):
        """Return detached values, indices, coordinates, and provenance."""
        import xarray as xr

        dims = tuple(name + '_bin' for name in self.axis_names)
        coords = {dim: indices.copy() for dim, indices in zip(dims, self.axis_indices)}
        coords.update({name: (dim, values.copy())
                       for name, dim, values in zip(self.axis_names, dims, self.axis_values)})
        attrs = dict(self.provenance)
        attrs.update(quantity=self.quantity, species=self.species, category=self.category,
                     storage_name=self.storage_name, checkpoint_suffix=self.checkpoint_suffix,
                     source_path=str(self.source_path), log_path=str(self.log_path),
                     legacy_roi_mode=self.legacy_roi_mode, region_kind=self.region_kind,
                     region_metadata=dict(self.region_metadata), value_semantics=self.value_semantics,
                     storage_shape=self.shape, value_sum=self.value_sum)
        if self.position_axis is not None:
            attrs['position_axis'] = self.position_axis
        return xr.DataArray(self.values.copy(), dims=dims, coords=coords,
                            name=self.quantity, attrs=attrs)


def read_distribution(directory, family, species, *, checkpoint, position_axis=None):
    spec = distribution_storage(family, species, position_axis)
    if not isinstance(checkpoint, str) or re.fullmatch(r'[0-9]+', checkpoint) is None:
        raise ValueError('checkpoint must be a numeric suffix string, preserving its spelling')
    files = scan_reduced_products(directory)
    source = files.get((spec.storage_name, checkpoint))
    if source is None:
        raise FileNotFoundError(f'Missing reduced distribution: {spec.storage_name}.{checkpoint}')
    log = files.get(('vd2dgyro', checkpoint))
    if log is None:
        raise DistributionMetadataError(f'Missing reducer log: vd2dgyro.{checkpoint}')
    metadata, bounds = _read_metadata(log)
    values = _read_values(source, spec)
    indices = tuple(np.arange(low, high + 1) for low, high in spec.index_ranges)
    scale = metadata['V_e' if spec.species == 'electron' else 'V_i']
    axes = []
    for name, index in zip(spec.axis_names, indices):
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            if name == 'position':
                axis = metadata[f'{spec.position_axis}_origin'] + (index / 100) * metadata[f'{spec.position_axis}_extent']
            else:
                axis = (index / 200) * scale
        if not np.all(np.isfinite(axis)) or not np.all(np.diff(axis) > 0):
            raise DistributionMetadataError(f'Cannot represent distinct finite {name} coordinates')
        axes.append(axis)
    provenance = dict(metadata)
    provenance.update(serialization_order='F', coordinate_reconstruction=RECONSTRUCTION,
                      producer_normalization=NORMALIZATION, known_selection_limitation=SELECTION,
                      physical_time='unavailable; checkpoint suffix is an identifier',
                      units_status='unavailable; numerical code-space coordinates',
                      metadata_recipe=spec.metadata_recipe)
    return ReducedDistribution(DISTRIBUTION_QUANTITIES[spec.family], spec.species, 'regular',
                               spec.storage_name, spec.position_axis, checkpoint, source, log,
                               0, 'box', bounds, 'normalized_bin_mass', values,
                               spec.axis_names, indices, tuple(axes), provenance)
