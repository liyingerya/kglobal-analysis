"""Case-level metadata coordination and human-readable inspection."""

from pathlib import Path
from typing import TYPE_CHECKING, TextIO
import sys

from .index import scan_case
from .parameters import Parameters, read_parameters

if TYPE_CHECKING:
    import numpy as np


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

    def bx_frame_count(self, segment: str) -> int:
        """Count complete 2D double-byte Bx frames using file size only."""
        from .bx import bx_frame_count

        return bx_frame_count(self, segment)

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
