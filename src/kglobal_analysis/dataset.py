"""Strict metadata alignment and composition of lazy movie variables."""

from collections.abc import Iterable
from typing import TYPE_CHECKING

from .format import movie_format
from .times import TimeMetadataError

if TYPE_CHECKING:
    import xarray as xr
    from .case import KGlobalCase


class MovieAlignmentError(ValueError):
    """Requested movie variables cannot share an exact timeline/provenance."""


def movie_dataset(case: "KGlobalCase", variables: Iterable[str], *, byteorder: str) -> "xr.Dataset":
    """Compose existing lazy series only after exact metadata validation."""
    if isinstance(variables, (str, bytes, set, frozenset)):
        raise TypeError("variables must be an ordered iterable of storage names")
    names = tuple(variables)
    if not names:
        raise ValueError("At least one movie variable is required")
    if not all(isinstance(name, str) for name in names):
        raise TypeError("Movie variable names must be strings")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate movie variable names are not allowed")
    schema = movie_format(case.parameters)
    for name in names:
        schema.variable(name)  # Keep unsupported names under format validation.
    series = []
    for name in names:
        if not any(name in entry.movies for entry in case.index.segments.values()):
            raise MovieAlignmentError(f"Movie variable {name!r} has no segments")
        try:
            current = case.movie(name, byteorder=byteorder)
        except TimeMetadataError as error:
            raise MovieAlignmentError(f"Cannot align movie variable {name!r}: {error}") from error
        if series:
            reference = series[0]
            pair = f"{reference.variable.storage_name!r} and {name!r}"
            if current.frame_count != reference.frame_count:
                raise MovieAlignmentError(f"Frame count mismatch between {pair}: "
                                          f"{reference.frame_count} vs {current.frame_count}")
            if current.times != reference.times:
                raise MovieAlignmentError(f"Actual timestamp mismatch between {pair}")
            if current._locations != reference._locations:
                raise MovieAlignmentError(f"Source segment/local frame provenance mismatch between {pair}")
        series.append(current)

    import xarray as xr

    arrays = [item.to_xarray() for item in series]
    # Use unaligned Variables with one explicitly validated coordinate set.
    # Passing DataArrays directly could invoke xarray's implicit alignment.
    first = arrays[0]
    return xr.Dataset(
        data_vars={name: array.variable for name, array in zip(names, arrays)},
        coords=first.coords,
        attrs={key: first.attrs[key] for key in ("movie_header", "encoding", "storage_order")},
    )
