"""Case-level metadata coordination and human-readable inspection."""

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, TextIO
import sys

from .index import scan_case
from .parameters import Parameters, read_parameters

if TYPE_CHECKING:
    import numpy as np
    import xarray as xr
    from .distribution import ReducedDistribution
    from .energy import EnergySpectrum
    from .manifest import MovieCaseReport
    from .segment import BxSegment, MovieSegment
    from .series import BxSeries, MovieSeries


class KGlobalCase:
    """Snapshot a case directory without reading any simulation data.

    An explicit relative parameter_file is resolved relative to the case.
    Otherwise prefer 'param', then a unique non-backup 'param_*' file.
    """

    def __init__(self, directory: str | Path, *, parameter_file: str | Path | None = None):
        self.directory = Path(directory).expanduser().resolve()
        self.index = scan_case(self.directory)
        if parameter_file is not None:
            selected = Path(parameter_file).expanduser()
            if not selected.is_absolute():
                selected = self.directory / selected
            self.parameter_file = selected.resolve()
        elif (self.directory / "param").is_file():
            self.parameter_file = self.directory / "param"
        else:
            candidates = sorted(path for path in self.directory.glob("param_*")
                                if path.is_file() and not path.name.endswith(
                                    ("~", ".bak", ".orig", ".swp", ".tmp")))
            if len(candidates) > 1:
                raise ValueError("Multiple parameter files; choose parameter_file explicitly: "
                                 + ", ".join(path.name for path in candidates))
            self.parameter_file = candidates[0] if candidates else None
        self.parameters = (read_parameters(self.parameter_file)
                           if self.parameter_file is not None else Parameters())

    @property
    def variables(self) -> tuple[str, ...]:
        return self.index.variables

    @property
    def suffixes(self) -> tuple[str, ...]:
        return self.index.suffixes

    def parallel_perpendicular_velocity_distribution(
        self, species: str, *, checkpoint: str,
    ) -> "ReducedDistribution":
        """Read regular-category joint bin mass with metadata-derived velocity axes."""
        from .distribution import read_distribution

        return read_distribution(self.directory, 'parperp', species, checkpoint=checkpoint)

    def position_parallel_velocity_distribution(
        self, species: str, *, position_axis: str, checkpoint: str,
    ) -> "ReducedDistribution":
        """Read box-selected position/parallel bin mass; x/y storage currently supported."""
        from .distribution import read_distribution

        return read_distribution(self.directory, 'positionpar', species,
                                 position_axis=position_axis, checkpoint=checkpoint)

    def energy_spectrum(self, species: str, *, checkpoint: str) -> "EnergySpectrum":
        """Read one reduced text spectrum; checkpoint identifies a file, not time."""
        from .energy import read_energy_spectrum

        return read_energy_spectrum(self.directory, species, checkpoint=checkpoint)

    def energy_spectrum_suffixes(self, species: str) -> tuple[str, ...]:
        """Discover current reduced spectrum files, separately from movie segments."""
        from .reduced import energy_spectrum_suffixes

        return energy_spectrum_suffixes(self.directory, species)

    def validate_movie_case(self) -> "MovieCaseReport":
        """Report movie metadata consistency using text and file sizes only.

        No sample reads, checksums, repairs, or cross-variable alignment occur.
        """
        from .manifest import validate_movie_case

        return validate_movie_case(self)

    def movie_frame_count(self, variable: str, segment: str) -> int:
        """Count complete volume frames using file size, without reading samples."""
        from .movie import movie_frame_count

        return movie_frame_count(self, variable, segment)

    def read_movie_frame(self, variable: str, segment: str, frame_index: int,
                         *, byteorder: str) -> "np.ndarray":
        """Read one float64 frame, (Nx, Ny) for Nz=1, else (Nx, Ny, Nz).

        Variable is a validated legacy storage name. Axes are in Fortran order;
        frame indices are zero-based and byteorder must be explicit.
        """
        from .movie import read_movie_frame

        return read_movie_frame(self, variable, segment, frame_index, byteorder=byteorder)

    def movie_segment(self, variable: str, suffix: str, *, byteorder: str) -> "MovieSegment":
        """Access a supported variable's segment with stdout-based times."""
        from .segment import MovieSegment

        return MovieSegment(self, variable, suffix, byteorder=byteorder)

    def movie(self, variable: str, *, byteorder: str) -> "MovieSeries":
        """Build a lazy global sequence for one supported storage variable."""
        from .series import MovieSeries

        return MovieSeries(self, variable, byteorder=byteorder)

    def movie_dataset(self, variables: Iterable[str], *, byteorder: str) -> "xr.Dataset":
        """Combine movie variables with exactly matching timelines/provenance.

        Construction reads metadata only. Values remain time-lazy, with whole
        spatial frames per chunk. Incompatible coverage raises MovieAlignmentError
        rather than joining timelines or filling missing frames.
        """
        from .dataset import movie_dataset

        return movie_dataset(self, variables, byteorder=byteorder)

    def bx_frame_count(self, segment: str) -> int:
        """Count complete 2D double-byte Bx frames using file size only."""
        from .bx import bx_frame_count

        return bx_frame_count(self, segment)

    def bx_segment(self, suffix: str, *, byteorder: str) -> "BxSegment":
        """Access one Bx segment with absolute stdout times and one-frame reads."""
        from .segment import BxSegment

        return BxSegment(self, suffix, byteorder=byteorder)

    def bx(self, *, byteorder: str) -> "BxSeries":
        """Validate a global timeline of all Bx segments; frame reads remain lazy."""
        from .series import BxSeries

        return BxSeries(self, byteorder=byteorder)

    def read_bx_frame(self, segment: str, frame_index: int, *, byteorder: str) -> "np.ndarray":
        """Return one float64 Bx[x, y] frame; byteorder is 'little' or 'big'.

        Frame indices are zero-based. No time or segment concatenation is done.
        The returned array owns its memory and has shape (Nx, Ny), Fortran order.
        """
        from .bx import read_bx_frame

        return read_bx_frame(self, segment, frame_index, byteorder=byteorder)

    def inspect(self, *, file: TextIO | None = None) -> None:
        """Print metadata; missing parameters are reported as unknown."""
        output = sys.stdout if file is None else file
        p = self.parameters

        def display(value: object) -> str:
            return "unknown" if value is None else str(value)

        print(f"KGlobal case: {self.directory}", file=output)
        print(f"Parameter file: {self.parameter_file or 'not found'}", file=output)
        print("Local grid (nx, ny, nz): " + ", ".join(map(display, (p.nx, p.ny, p.nz))), file=output)
        print("Processor grid (pex, pey, pez): " + ", ".join(map(display, (p.pex, p.pey, p.pez))), file=output)
        print("Global grid (Nx, Ny, Nz): " + ", ".join(map(display, (p.Nx, p.Ny, p.Nz))), file=output)
        print(f"dt: {display(p.dt)}; n_movieout: {display(p.n_movieout)}; "
              f"movie_dt: {display(p.movie_dt)}", file=output)
        print(f"movie_header: {display(p.movie_header)}; "
              f"double_byte: {p.double_byte}; four_byte: {p.four_byte}", file=output)
        print("Movie variables: " + (", ".join(self.variables) or "none"), file=output)
        print("Segments: " + (", ".join(self.suffixes) or "none"), file=output)
        for suffix in self.suffixes:
            segment = self.index.segments[suffix]
            print(f"  {suffix}: variables={','.join(segment.variables) or 'none'}; "
                  f"log={segment.movie_log.name if segment.movie_log else 'missing'}; "
                  f"stdout={segment.stdout.name if segment.stdout else 'missing'}", file=output)
