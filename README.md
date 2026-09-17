# kglobal-analysis

Milestone 1: a minimal, standard-library-only Python package for inspecting
KGlobal case structure and parameter metadata. Requires Python 3.10 or later.

No binary movie loading, xarray integration, particle/checkpoint readers, or
physics analysis is implemented. Movie, log, and stdout contents are not read.
Only the selected parameter file is opened as text. Case inspection is read-only.

## Use

From this project directory, install with `python -m pip install -e .`, or use
`PYTHONPATH=src python` without installing anything.

```python
from kglobal_analysis import KGlobalCase

case = KGlobalCase("/path/to/simulation/case")
case.inspect()

print(case.variables)                  # e.g. ('bx', 'ni', 'pehpar')
print(case.suffixes)                   # e.g. ('000', '015'); gaps are valid
segment = case.index.segments["015"]
print(segment.movies)                  # variable -> absolute Path
print(segment.movie_log)               # Path or None
print(segment.stdout)                  # Path or None
print(case.parameters.Nx)
print(case.parameters.movie_dt)
```

The constructor takes a metadata snapshot. Construct a new case to rescan.
`inspect(file=stream)` can send the summary to a text stream.

## Discovery and parameter selection

- Scan immediate files only: `movie.<variable>.<digits>`, `movie.log.<digits>`,
  and `p3d.stdout.<digits>`. Directories and unrelated/backup names are ignored.
- Suffixes remain strings, preserving zero padding. They sort numerically, with
  spelling as a tie-breaker; `002` and `2` remain distinct identifiers.
- Segments can have different variables, missing companions, or only companions.
  `log` is not a movie variable. Suffixes identify run segments, not frames.
- Prefer an exact `param` file, otherwise a unique `param_*` file excluding
  common backup endings (`~`, `.bak`, `.orig`, `.swp`, `.tmp`). Multiple candidates
  raise `ValueError`. Choose explicitly with
  `KGlobalCase(case_dir, parameter_file="param_run")`; relative paths are resolved
  against the case directory. No parameter file is also valid: values are unknown.

## Parameter parser scope

`parse_parameters(text)` is independent of discovery; `read_parameters(path)`
adds text-file reading. The parser supports object-like `#define` statements,
`#undef`, leading whitespace, quoted strings, and `!`, `//`, and `/* ... */`
comments. Repeated definitions use the last value. Unknown object-like macros
are retained in `Parameters.definitions` without evaluation.

Extracted values are `nx`, `ny`, `nz`, `pex`, `pey`, `pez`, `dt`, `n_movieout`,
`movie_header`, `double_byte`, and `four_byte`. Integer fields must be positive
integer literals. `dt` accepts decimal and E/D exponent notation, including
negative time steps, but must be finite and nonzero. Numeric literals may be
parenthesized. `movie_header` must be quoted. Flags reflect macro presence:
`#define double_byte 0` still defines the flag, consistent with `#ifdef`.
Both flags are preserved if present; no binary dtype is inferred in this milestone.

This is not a full CPP interpreter. Macro expressions, function-like macros,
line continuations, conditional directives, and includes are unsupported and
raise `ValueError` rather than silently inferring active values. Unsupported
expressions in unrelated object-like definitions are merely retained as text.
Line continuations are rejected before comment removal, including inside
comments. Unterminated block comments also raise `ValueError`.

Derived values are `Nx=nx*pex`, `Ny=ny*pey`, `Nz=nz*pez`, and
`movie_dt=dt*n_movieout`. Missing inputs produce `None`. `movie_dt` is a signed
nominal interval, not an absolute frame time. There is no `movieout` fallback or
restart-time reconstruction in this milestone.

## Organization

- `index.py`: filename parsing and filesystem discovery.
- `parameters.py`: parameter text parsing and simple metadata derivations.
- `case.py`: coordination, parameter-file selection, and printed summaries.

The neighboring `upstream/` and `legacy/` trees are read-only references and are
not runtime dependencies. Naming/semantics were checked against
`upstream/p3d-kglobal/kglobal/src/movie_kglobal3.0.h` (lines 57–70, 77–114,
184–208), `upstream/p3d-kglobal/kglobal/scripts/p3d_subm_scripts/exp3d_submit_nersc`
(lines 435–452), and upstream/legacy IDL `moviesetupscript_part1` (upstream lines
84–107; legacy lines 63–86). The sample parameter files are configurations, not
universal defaults; matching a parameter file to a particular run remains the
caller's responsibility.

## Tests

Run from this project directory without installing dependencies:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Tests cover filename recognition/rejection, sparse segments and companions,
parameter syntax and derivation, unsupported input, case selection, and summary
output. Temporary test fixtures are created and removed inside `tests/`.
