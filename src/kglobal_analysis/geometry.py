"""Explicit simulation geometry, independent of binary layout and rendering."""
from copy import deepcopy
from dataclasses import dataclass
import math
import re

import numpy as np
import xarray as xr

from .format import movie_format
from .parameters import Parameters, _unparenthesize


class GeometryError(ValueError):
    """Missing, unsupported, or conflicting simulation geometry."""


PROFILE = 'kglobal-initrecon-periodic-2d-v1'


def _literal(definitions, name):
    value = definitions.get(name)
    if not isinstance(value, str):
        raise GeometryError(f'Missing literal {name}')
    value = _unparenthesize(value.strip())
    if not re.fullmatch(r'[+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?', value):
        raise GeometryError(f'{name} must be a positive numeric literal, not a CPP expression')
    number = float(re.sub('[dD]', 'e', value))
    if not math.isfinite(number) or number <= 0:
        raise GeometryError(f'{name} must be positive and finite')
    return number


def _equal(left, right):
    if hasattr(left, '__dask_graph__') or hasattr(right, '__dask_graph__'):
        raise GeometryError('Geometry/provenance must be eager metadata')
    return np.array_equal(left, right)


@dataclass(frozen=True)
class MovieGeometry:
    """Declared current initrecon geometry, not executable certification.

    Direct construction explicitly declares this profile for synthetic/external
    data. Case resolution checks the parsed producer metadata. Nz/lz are storage
    provenance only: no scientific z coordinate or dz is supplied.
    """
    Nx: int
    Ny: int
    Nz: int
    lx: float
    ly: float
    lz: float
    geometry_profile: str = PROFILE
    periodic_x: bool = True
    periodic_y: bool = True
    origin_x: float = 0.
    origin_y: float = 0.
    sample_offset_x: float = .5
    sample_offset_y: float = .5
    parameter_source: str = 'explicit declaration'

    def __post_init__(self):
        if self.geometry_profile != PROFILE:
            raise GeometryError('Unsupported geometry profile')
        for name in ('Nx', 'Ny', 'Nz'):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise GeometryError(f'{name} must be a positive integer')
        if self.Nz != 1:
            raise GeometryError('Resolved 3D geometry is not validated; require Nz=1')
        for name in ('lx', 'ly', 'lz'):
            value = getattr(self, name)
            if isinstance(value, (bool, str)) or not isinstance(value, (int, float, np.number)) or not np.isrealobj(value):
                raise GeometryError(f'{name} must be positive and finite')
            if not math.isfinite(value) or value <= 0:
                raise GeometryError(f'{name} must be positive and finite')
        if self.periodic_x is not True or self.periodic_y is not True:
            raise GeometryError('This profile requires explicit periodic x/y boundaries')
        for name, expected in (('origin_x', 0.), ('origin_y', 0.),
                               ('sample_offset_x', .5), ('sample_offset_y', .5)):
            if isinstance(getattr(self, name), bool) or getattr(self, name) != expected:
                raise GeometryError(f'Unsupported {name} for this profile')
        if not isinstance(self.parameter_source, str) or not self.parameter_source:
            raise GeometryError('parameter_source must describe the geometry declaration')
        if not (np.isfinite(self.dx) and self.dx > 0 and np.isfinite(self.dy) and self.dy > 0):
            raise GeometryError('Grid spacing must be positive and finite')

    @classmethod
    def from_parameters(cls, parameters: Parameters, *, parameter_source='parsed parameters'):
        """Resolve literal metadata; no CPP evaluation or inferred boundary default."""
        if not isinstance(parameters, Parameters):
            raise GeometryError('Expected parsed Parameters')
        try:
            movie_format(parameters)
        except ValueError as error:
            raise GeometryError(str(error)) from error
        d = parameters.definitions
        if d.get('movie_header') != f'"{parameters.movie_header}"':
            raise GeometryError('Missing or conflicting movie_header')
        for name in ('double_byte', 'four_byte'):
            if (name in d) != getattr(parameters, name):
                raise GeometryError(f'Conflicting {name}')
        if d.get('init_scheme') != 'initrecon':
            raise GeometryError('Only the declared current initrecon profile is supported')
        if d.get('boundary_condition') != 'periodic':
            raise GeometryError('Explicit boundary_condition periodic is required')
        for name in ('nx', 'ny', 'nz', 'pex', 'pey', 'pez'):
            raw = _unparenthesize(d.get(name, '').strip())
            typed = getattr(parameters, name)
            if (not re.fullmatch(r'[+]?\d+', raw) or int(raw) <= 0
                    or isinstance(typed, bool) or not isinstance(typed, int) or typed != int(raw)):
                raise GeometryError(f'Missing or conflicting positive integer {name}')
        return cls(parameters.Nx, parameters.Ny, parameters.Nz,
                   *(_literal(d, name) for name in ('lx', 'ly', 'lz')),
                   parameter_source=parameter_source)

    @property
    def dx(self):
        return self.lx / self.Nx

    @property
    def dy(self):
        return self.ly / self.Ny

    @property
    def degenerate_z(self):
        return True

    @property
    def x(self):
        return (np.arange(self.Nx, dtype=np.float64) + .5) * self.dx

    @property
    def y(self):
        return (np.arange(self.Ny, dtype=np.float64) + .5) * self.dy

    def _attrs(self):
        return dict(geometry_profile=self.geometry_profile, coordinate_units='code_normalized',
                    grid_layout='uniform_collocated_cell_center', dx=self.dx, dy=self.dy,
                    lx=self.lx, ly=self.ly, lz=self.lz, Nx=self.Nx, Ny=self.Ny, Nz=self.Nz,
                    periodic_x=True, periodic_y=True, degenerate_z=True,
                    init_scheme='initrecon', boundary_condition='periodic',
                    origin_x=0., origin_y=0., sample_offset_x=.5, sample_offset_y=.5,
                    geometry_provenance=self.parameter_source)


def attach_movie_geometry(data, geometry: MovieGeometry):
    """Attach exact x/y centers without reading samples or changing dimension order.

    Existing coordinates must match exactly and be eager metadata. Geometry
    stamps are checked on the container and variables; conflicting units/layout
    are errors. Returns a new object with detached attrs and shared sample data.
    """
    if not isinstance(geometry, MovieGeometry):
        raise GeometryError('Supply an explicit MovieGeometry')
    if not isinstance(data, (xr.DataArray, xr.Dataset)):
        raise GeometryError('Expected an xarray DataArray or Dataset')
    if 'z' in data.dims or 'z' in data.coords:
        raise GeometryError('No resolved/sliced z coordinate is validated by this 2D profile')
    expected = geometry._attrs()
    containers = [data] + (list(data.data_vars.values()) if isinstance(data, xr.Dataset) else [])
    for item in containers:
        for key, value in expected.items():
            if key != 'geometry_provenance' and key in item.attrs and not _equal(item.attrs[key], value):
                raise GeometryError(f'Conflicting geometry metadata: {key}')
    coords = {}
    for axis, size, values in (('x', geometry.Nx, geometry.x), ('y', geometry.Ny, geometry.y)):
        if data.sizes.get(axis) != size:
            raise GeometryError(f'{axis} size must be {size}')
        attrs = {}
        if axis in data.coords:
            coord = data.coords[axis]
            if coord.chunks is not None or coord.dims != (axis,):
                raise GeometryError(f'{axis} must be eager one-dimensional coordinate metadata')
            if not _equal(coord.values, values):
                raise GeometryError(f'Conflicting {axis} coordinates')
            attrs = deepcopy(coord.attrs)
            if attrs.get('units', 'code_normalized') != 'code_normalized':
                raise GeometryError(f'Conflicting {axis} coordinate units')
        attrs['units'] = 'code_normalized'
        coords[axis] = xr.DataArray(values, dims=axis, attrs=attrs)
    result = data.assign_coords(coords)
    result.attrs = deepcopy(data.attrs)
    result.attrs.update(expected)
    if isinstance(result, xr.Dataset):
        for name in result.data_vars:
            result[name].attrs = deepcopy(data[name].attrs)
    return result
