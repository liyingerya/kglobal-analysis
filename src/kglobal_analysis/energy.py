"""Faithful eager reading of small legacy reduced energy text products."""

from dataclasses import dataclass
from pathlib import Path
import math
import re
from typing import TYPE_CHECKING

from .reduced import EnergyStorageSpec, energy_storage, scan_reduced_products

if TYPE_CHECKING:
    import xarray as xr

# Provisional public terminology, independent of storage discovery and parsing.
ENERGY_QUANTITY = 'energy_spectrum'
ACTIVE_BINS = 200
_NUMBER = re.compile(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[EeDd][+-]?[0-9]+)?')
_LIMIT = re.compile(r'^\s*(Emin|Emax|Imin|Imax)\s*:\s*(.*?)\s*$')


class EnergySpectrumError(ValueError):
    """Invalid reduced spectrum storage."""


class EnergyMetadataError(EnergySpectrumError):
    """Missing, ambiguous, or invalid reducer-log energy coordinates."""


def _number(token: str) -> float:
    if _NUMBER.fullmatch(token) is None:
        raise ValueError(f'Invalid numeric value {token!r}')
    value = float(token.replace('D', 'E').replace('d', 'e'))
    if not math.isfinite(value):
        raise ValueError(f'Nonfinite numeric value {token!r}')
    return value


def _read_values(path: Path) -> tuple[float, ...]:
    try:
        values = tuple(_number(token) for token in path.read_text(encoding='ascii').split())
    except (ValueError, UnicodeError) as error:
        raise EnergySpectrumError(f'Invalid spectrum data in {path}: {error}') from error
    if len(values) != ACTIVE_BINS + 1:
        raise EnergySpectrumError(f'{path}: expected 201 stored values, found {len(values)}')
    return values


def _read_limits(path: Path, spec: EnergyStorageSpec) -> tuple[float, float]:
    labels = (spec.minimum_label, spec.maximum_label)
    found = {}
    try:
        with path.open(encoding='ascii') as stream:
            for line_number, line in enumerate(stream, 1):
                match = _LIMIT.fullmatch(line.rstrip('\r\n'))
                if match is None or match[1] not in labels:
                    continue
                label, token = match.groups()
                if label in found:
                    raise ValueError(f'Duplicate {label} on line {line_number}')
                found[label] = _number(token)
        missing = [label for label in labels if label not in found]
        if missing:
            raise ValueError('Missing energy limit(s): ' + ', '.join(missing))
        low, high = (found[label] for label in labels)
        if low <= 0 or high <= low:
            raise ValueError('Energy limits must satisfy 0 < minimum < maximum')
        return low, high
    except (OSError, ValueError) as error:
        raise EnergyMetadataError(f'Invalid energy metadata in {path}: {error}') from error


@dataclass(frozen=True)
class EnergySpectrum:
    """Immutable parsed values and provenance; coordinates have no inferred units.

    All numeric sequences are tuples. Index zero is retained as evidence, not
    an active bin. Geometric centers are derived plotting coordinates.
    """

    quantity: str
    species: str
    storage_name: str
    checkpoint_suffix: str
    source_path: Path
    log_path: Path
    storage_values: tuple[float, ...]
    Emin: float
    Emax: float
    edges: tuple[float, ...]

    @property
    def bin_count(self) -> int:
        return ACTIVE_BINS

    @property
    def sentinel_value(self) -> float:
        return self.storage_values[0]

    @property
    def values(self) -> tuple[float, ...]:
        return self.storage_values[1:]

    @property
    def lower_edges(self) -> tuple[float, ...]:
        return self.edges[:-1]

    @property
    def upper_edges(self) -> tuple[float, ...]:
        return self.edges[1:]

    @property
    def widths(self) -> tuple[float, ...]:
        return tuple(upper - lower for lower, upper in zip(self.lower_edges, self.upper_edges))

    @property
    def centers(self) -> tuple[float, ...]:
        # Avoid overflow in the intermediate lower*upper product.
        return tuple(math.sqrt(lower) * math.sqrt(upper)
                     for lower, upper in zip(self.lower_edges, self.upper_edges))

    @property
    def width_integral(self) -> float:
        """Active-bin sum(values * widths), without repairing or rescaling data."""
        return sum(value * width for value, width in zip(self.values, self.widths))

    def to_xarray(self) -> 'xr.DataArray':
        """Return a detached 200-bin DataArray; no physical time is available."""
        import xarray as xr

        return xr.DataArray(
            list(self.values), dims=('energy_bin',), name=self.quantity,
            coords={
                'energy_bin': list(range(1, self.bin_count + 1)),
                'energy_lower': ('energy_bin', list(self.lower_edges)),
                'energy_upper': ('energy_bin', list(self.upper_edges)),
                'energy_center': ('energy_bin', list(self.centers)),
                'energy_width': ('energy_bin', list(self.widths)),
            },
            attrs={
                'quantity': self.quantity, 'species': self.species,
                'storage_name': self.storage_name, 'checkpoint_suffix': self.checkpoint_suffix,
                'source_path': str(self.source_path), 'log_path': str(self.log_path),
                'storage_entries': len(self.storage_values), 'active_bins': self.bin_count,
                'sentinel_value': self.sentinel_value, 'energy_min': self.Emin, 'energy_max': self.Emax,
                'estimator': 'Legacy sum of 1/K contributions per accepted logarithmic energy bin; not raw counts',
                'normalization': 'Producer applies bin-width normalization; reader preserves stored values unchanged',
                'energy_coordinate': '200 logarithmic intervals from logged limits; geometric centers are Python-derived plotting coordinates',
                'physical_time': 'unavailable; checkpoint suffix is an identifier',
            },
        )


def read_energy_spectrum(directory: Path, species: str, *, checkpoint: str) -> EnergySpectrum:
    spec = energy_storage(species)
    if not isinstance(checkpoint, str) or re.fullmatch(r'[0-9]+', checkpoint) is None:
        raise ValueError('checkpoint must be a numeric suffix string, preserving its spelling')
    files = scan_reduced_products(directory)
    source = files.get((spec.storage_name, checkpoint))
    if source is None:
        raise FileNotFoundError(f'Missing reduced spectrum: {directory / (spec.storage_name + "." + checkpoint)}')
    log = files.get(('vd2dgyro', checkpoint))
    if log is None:
        raise EnergyMetadataError(f'Missing reducer log: {directory / ("vd2dgyro." + checkpoint)}')
    values = _read_values(source)
    low, high = _read_limits(log, spec)
    # Algebraically low*(high/low)**(j/200), without overflowing high/low.
    step = (math.log(high) - math.log(low)) / ACTIVE_BINS
    edges = (low,) + tuple(math.exp(math.log(low) + j * step)
                           for j in range(1, ACTIVE_BINS)) + (high,)
    if any(not lower < upper for lower, upper in zip(edges, edges[1:])):
        raise EnergyMetadataError('Energy limits cannot represent 200 distinct float64 intervals')
    return EnergySpectrum(ENERGY_QUANTITY, species, spec.storage_name, checkpoint,
                          source, log, values, low, high, edges)
