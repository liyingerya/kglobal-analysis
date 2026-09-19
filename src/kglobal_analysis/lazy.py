"""Time-lazy xarray adapter; storage decoding stays in the segment reader.

Whole spatial frames are a task-planning choice here, not part of the movie
schema or time model. A future region loader can change this adapter's tasks.
"""

from typing import TYPE_CHECKING

from .format import VolumeLayout, movie_format

if TYPE_CHECKING:
    import xarray as xr
    from .series import MovieSeries


def series_to_xarray(series: "MovieSeries") -> "xr.DataArray":
    """Build a graph and eager provenance, without reading any movie samples."""
    import numpy as np
    import xarray as xr
    import dask.array as da
    from dask import delayed

    parameters = series._case.parameters
    schema = movie_format(parameters)
    layout = VolumeLayout(parameters.Nx, parameters.Ny, parameters.Nz)
    frames = []
    for suffix, local_index in series._locations:
        segment = series._segments[suffix]
        task = delayed(segment.read_frame, pure=False)(local_index)
        frames.append(da.from_delayed(task, shape=layout.public_shape, dtype=np.float64))
    values = da.stack(frames, axis=0)
    return xr.DataArray(
        values,
        dims=("time", "x", "y", "z")[:values.ndim],
        coords={
            "time": ("time", np.asarray(series.times, dtype=np.float64)),
            "source_segment": ("time", [suffix for suffix, _ in series._locations]),
            "local_frame_index": ("time", [index for _, index in series._locations]),
            "global_frame_index": ("time", np.arange(series.frame_count)),
        },
        name=series.variable.storage_name,
        attrs={
            "storage_name": series.variable.storage_name,
            "movie_header": schema.header,
            "encoding": "double_byte",
            "storage_order": "Fortran: x fastest, then y, then z, then frame",
        },
    )
