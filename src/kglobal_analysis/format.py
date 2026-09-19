"""Explicit storage schema and Fortran volume layout, independent of physics."""

from dataclasses import dataclass
import operator
from .parameters import Parameters


@dataclass(frozen=True)
class VariableSpec:
    """Legacy storage identity, not a canonical quantity/species name."""

    storage_name: str
    log_index: int


@dataclass(frozen=True)
class MovieFormat:
    """An ordered on-disk schema; semantic aliases belong outside decoding."""

    header: str
    variables: tuple[VariableSpec, ...]

    def variable(self, storage_name: str) -> VariableSpec:
        for spec in self.variables:
            if spec.storage_name == storage_name:
                return spec
        raise ValueError(f"Unsupported movie variable {storage_name!r} for {self.header}")


STANDARD_MOVIE_FORMAT = MovieFormat(
    "movie_kglobal3.0.h",
    tuple(VariableSpec(name, index) for index, name in enumerate((
        "ni", "jix", "jiy", "jiz", "bx", "by", "bz", "pi", "neh",
        "pehpar", "pehperp", "epar", "pc", "jhpar", "nih", "pihpar",
        "pihperp", "jihpar",
    ))),
)


def movie_format(parameters: Parameters) -> MovieFormat:
    """Select only the validated standard double-byte configuration."""
    if not parameters.double_byte or parameters.four_byte:
        raise ValueError("Only double_byte without four_byte is supported")
    if parameters.movie_header != STANDARD_MOVIE_FORMAT.header:
        raise ValueError("Only movie_kglobal3.0.h with the standard 18-entry log is supported")
    for name in ("heatfluxmovies", "mult_species", "movie_header2"):
        if name in parameters.definitions:
            raise ValueError(f"Unsupported movie layout option: {name}")
    return STANDARD_MOVIE_FORMAT


@dataclass(frozen=True)
class VolumeLayout:
    """Signed-int16 volume: x fastest, then y, z, frame. No guard cells."""

    Nx: int
    Ny: int
    Nz: int

    def __post_init__(self):
        for name in ("Nx", "Ny", "Nz"):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise TypeError(f"{name} must be an integer, not bool")
            value = operator.index(value)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.Nx, self.Ny, self.Nz

    @property
    def samples_per_frame(self) -> int:
        return self.Nx * self.Ny * self.Nz

    @property
    def frame_bytes(self) -> int:
        return self.samples_per_frame * 2

    @property
    def public_shape(self) -> tuple[int, ...]:
        """Retain the 2D convention only when z is singleton."""
        return self.shape[:2] if self.Nz == 1 else self.shape

    def frame_count(self, size_bytes: int) -> int:
        if isinstance(size_bytes, bool):
            raise TypeError("size_bytes must be an integer, not bool")
        size_bytes = operator.index(size_bytes)
        if size_bytes <= 0 or size_bytes % self.frame_bytes:
            raise ValueError(f"Movie file size {size_bytes} is not a positive multiple of {self.frame_bytes}")
        return size_bytes // self.frame_bytes
