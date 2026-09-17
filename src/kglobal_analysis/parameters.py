"""Parse literal KGlobal CPP definitions without executing a preprocessor."""

from dataclasses import dataclass, field
from pathlib import Path
import math
import re


@dataclass(frozen=True)
class Parameters:
    nx: int | None = None
    ny: int | None = None
    nz: int | None = None
    pex: int | None = None
    pey: int | None = None
    pez: int | None = None
    dt: float | None = None
    n_movieout: int | None = None
    movie_header: str | None = None
    double_byte: bool = False
    four_byte: bool = False
    definitions: dict[str, str] = field(default_factory=dict)

    @property
    def Nx(self) -> int | None:
        return None if self.nx is None or self.pex is None else self.nx * self.pex

    @property
    def Ny(self) -> int | None:
        return None if self.ny is None or self.pey is None else self.ny * self.pey

    @property
    def Nz(self) -> int | None:
        return None if self.nz is None or self.pez is None else self.nz * self.pez

    @property
    def movie_dt(self) -> float | None:
        return (None if self.dt is None or self.n_movieout is None
                else self.dt * self.n_movieout)


def _unparenthesize(value: str) -> str:
    while value.startswith("(") and value.endswith(")"):
        value = value[1:-1].strip()
    return value


def parse_parameters(text: str) -> Parameters:
    """Parse object-like definitions, comments, and #undef.

    Selected numeric fields must be literals (optionally parenthesized).
    Conditional/include directives are rejected rather than guessed. Unknown
    object-like macros are preserved verbatim, not evaluated.
    """
    # CPP line splicing precedes comment removal. Reject it before stripping
    # comments so a continued comment cannot expose an inactive definition.
    if re.search(r"\\\r?\n", text):
        raise ValueError("Line continuations are unsupported, including in comments")

    # Preserve quoted strings while removing CPP and Fortran-style comments.
    # Match an unfinished block comment as well, so it cannot expose defines.
    token = r'"(?:\\.|[^"\\])*"|/\*.*?(?:\*/|\Z)|//[^\n]*|![^\n]*'

    def strip_comment(match: re.Match) -> str:
        value = match[0]
        if value.startswith('"'):
            return value
        if value.startswith("/*") and not value.endswith("*/"):
            raise ValueError("Unterminated block comment")
        return " " + "\n" * value.count("\n")

    text = re.sub(token, strip_comment, text, flags=re.DOTALL)
    definitions: dict[str, str] = {}
    for lineno, line in enumerate(text.splitlines(), 1):
        directive = re.match(r"^\s*#\s*(\w+)\b(.*)$", line)
        if directive is None:
            continue
        command, rest = directive.groups()
        if command not in {"define", "undef"}:
            raise ValueError(f"Line {lineno}: unsupported #{command} directive")
        macro = re.fullmatch(r"\s+([A-Za-z_]\w*)(.*)", rest)
        if macro is None:
            raise ValueError(f"Line {lineno}: malformed #{command}")
        name, value = macro.groups()
        if value.startswith("(") or value.rstrip().endswith("\\"):
            raise ValueError(f"Line {lineno}: function-like or continued macro unsupported")
        if command == "undef":
            if value.strip():
                raise ValueError(f"Line {lineno}: malformed #undef")
            definitions.pop(name, None)
        else:
            definitions[name] = value.strip()

    values: dict = {}
    for name in ("nx", "ny", "nz", "pex", "pey", "pez", "n_movieout"):
        if name not in definitions:
            continue
        value = _unparenthesize(definitions[name])
        if not re.fullmatch(r"[+]?\d+", value) or int(value) <= 0:
            raise ValueError(f"{name} must be a positive integer literal: {definitions[name]!r}")
        values[name] = int(value)
    if "dt" in definitions:
        value = _unparenthesize(definitions["dt"])
        if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?", value):
            raise ValueError(f"dt must be a numeric literal: {value!r}")
        values["dt"] = float(re.sub(r"[dD]", "e", value))
        if not math.isfinite(values["dt"]) or values["dt"] == 0:
            raise ValueError("dt must be finite and nonzero")
    if "movie_header" in definitions:
        value = definitions["movie_header"]
        if not re.fullmatch(r'"[^"\n]+"', value):
            raise ValueError("movie_header must be a quoted filename")
        values["movie_header"] = value[1:-1]
    return Parameters(**values, definitions=definitions,
                      double_byte="double_byte" in definitions,
                      four_byte="four_byte" in definitions)


def read_parameters(path: str | Path) -> Parameters:
    """Read only the parameter text file."""
    return parse_parameters(Path(path).read_text(encoding="utf-8"))
