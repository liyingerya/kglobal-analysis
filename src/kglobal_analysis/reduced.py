"""Exact legacy reduced-product discovery, separate from movie segments."""

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class EnergyStorageSpec:
    storage_name: str
    minimum_label: str
    maximum_label: str


# Explicit producer mapping; public quantity naming is deliberately elsewhere.
ENERGY_STORAGE = {
    'electron': EnergyStorageSpec('xenergylog', 'Emin', 'Emax'),
    'ion': EnergyStorageSpec('xenergylogi', 'Imin', 'Imax'),
}


def energy_storage(species: str) -> EnergyStorageSpec:
    if not isinstance(species, str) or species not in ENERGY_STORAGE:
        raise ValueError("Supported energy-spectrum species are 'electron' and 'ion'")
    return ENERGY_STORAGE[species]


def parse_reduced_filename(name: str) -> tuple[str, str] | None:
    """Return (storage identifier, checkpoint suffix) for exact supported names."""
    match = re.fullmatch(r'(xenergylog|xenergylogi|vd2dgyro|vdeparperp|vdiparperp|xpepar|xpipar|ypepar|ypipar)\.([0-9]+)', name)
    return match.groups() if match else None


def scan_reduced_products(directory: Path) -> dict[tuple[str, str], Path]:
    """Inspect immediate filenames only; do not open any file."""
    result = {}
    for path in directory.iterdir():
        record = parse_reduced_filename(path.name)
        if record is not None and path.is_file():
            result[record] = path
    return result


def energy_spectrum_suffixes(directory: Path, species: str) -> tuple[str, ...]:
    storage = energy_storage(species).storage_name
    return tuple(sorted((suffix for name, suffix in scan_reduced_products(directory)
                         if name == storage), key=lambda suffix: (int(suffix), suffix)))


@dataclass(frozen=True)
class DistributionStorageSpec:
    storage_name: str
    species: str
    family: str
    position_axis: str | None
    raw_shape: tuple[int, ...]
    index_ranges: tuple[tuple[int, int], ...]
    axis_names: tuple[str, ...]
    metadata_recipe: str


DISTRIBUTION_STORAGE = (
    DistributionStorageSpec('vdeparperp', 'electron', 'parperp', None, (401, 201),
                            ((-200, 200), (0, 200)), ('v_parallel', 'v_perp'), 'legacy_velocity'),
    DistributionStorageSpec('vdiparperp', 'ion', 'parperp', None, (401, 201),
                            ((-200, 200), (0, 200)), ('v_parallel', 'v_perp'), 'legacy_velocity'),
    DistributionStorageSpec('xpepar', 'electron', 'positionpar', 'x', (101, 401),
                            ((0, 100), (-200, 200)), ('position', 'v_parallel'), 'legacy_position_velocity'),
    DistributionStorageSpec('xpipar', 'ion', 'positionpar', 'x', (101, 401),
                            ((0, 100), (-200, 200)), ('position', 'v_parallel'), 'legacy_position_velocity'),
    DistributionStorageSpec('ypepar', 'electron', 'positionpar', 'y', (101, 401),
                            ((0, 100), (-200, 200)), ('position', 'v_parallel'), 'legacy_position_velocity'),
    DistributionStorageSpec('ypipar', 'ion', 'positionpar', 'y', (101, 401),
                            ((0, 100), (-200, 200)), ('position', 'v_parallel'), 'legacy_position_velocity'),
)


def distribution_storage(family, species, position_axis=None):
    for spec in DISTRIBUTION_STORAGE:
        if (family, species, position_axis) == (spec.family, spec.species, spec.position_axis):
            return spec
    raise ValueError('Supported distributions require electron/ion species and, for position products, x/y position_axis')
