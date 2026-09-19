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
    match = re.fullmatch(r'(xenergylog|xenergylogi|vd2dgyro)\.([0-9]+)', name)
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
