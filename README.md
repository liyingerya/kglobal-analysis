# kglobal-analysis

Milestones 1–2: KGlobal case structure/parameter inspection and reconstruction
of one double-byte 2D Bx frame from one segment. Requires Python 3.10 or later;
the Bx reader uses NumPy (>=1.23).

Case construction and `inspect()` still read only parameter text, never movie,
log, or stdout contents. Binary reading happens only through an explicit
`read_bx_frame()` call. All case access is read-only. No xarray integration,
multi-frame loading, segment concatenation, other movie variables, four-byte
reading, particle/checkpoint readers, plotting, or physics analysis is provided.

## Use

From this project directory, install with `python -m pip install -e .`, or use
`PYTHONPATH=src python` with NumPy already available. The metadata-only APIs
remain usable without importing NumPy.

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

## Read one Bx frame

```python
from kglobal_analysis import KGlobalCase

case = KGlobalCase("../validation-data/hcs_large_005")
count = case.bx_frame_count("005")  # 20; reads file size, not binary contents
bx = case.read_bx_frame("005", 0, byteorder="little")
print(bx.shape, bx.dtype)           # (8192, 4096), float64
print(bx[0, 1])                    # x=0, y=1; no time or singleton z axis
```

Only `Nz == 1`, `double_byte` without `four_byte`, and the standard
`movie_kglobal3.0.h` 18-entry log are accepted. `heatfluxmovies`, `mult_species`,
and `movie_header2` are rejected. Missing dimensions, missing Bx/log files,
partial or empty binary files, and malformed/mismatched logs raise exceptions.
An invalid frame index raises `IndexError`; negative indices do not wrap.
The segment must exist in the case index; an unknown suffix raises `KeyError`.

The format has no header or Fortran record markers. Signed int16 samples are
stored with **x fastest, then y, then frame**. With zero-based `x`, `y`, and `k`:

```text
frame_bytes = Nx * Ny * 2
frame_offset = k * frame_bytes
sample_offset = frame_offset + 2 * (x + Nx * y)
```

The writer uses native byte order, with no endian marker in the file. Therefore
`byteorder` is mandatory (`"little"` or `"big"`); no automatic detection or host
endianness assumption is made. Validation below explicitly uses little-endian.

The reader seeks to the requested frame and calls `numpy.fromfile` with an
explicit count of `Nx*Ny`. Only one frame's encoded samples are read: 64 MiB for
the real case, not the complete 1.25 GiB movie. The returned array owns its memory,
has shape **`(Nx, Ny)`**, uses **Fortran-contiguous order**, and has **float64**
values. For the validation case the output occupies 256 MiB; encoded input plus
output occupy about 320 MiB during reconstruction. Scaling is performed in place
on the float64 output, avoiding additional full-size arithmetic temporaries.

The standard text log has one min/max pair per variable, 18 pairs per frame:

```text
ni jix jiy jiz bx by bz pi neh pehpar pehperp
epar pc jhpar nih pihpar pihperp jihpar
```

Bx is zero-based entry `18*k + 4` (one-based line `18*k + 5`). The text log must
have exactly `18 * frame_count` entries. The reader validates numeric, finite,
ordered pairs, accepts Fortran D exponents and fixed-width `(E14.6,E14.6)` fields,
and reconstructs using:

```text
Bx = minimum + (float64(q) + 32768) * (maximum - minimum) / 65535
```

Conversion precedes addition to prevent int16 overflow. Equal min/max values
reconstruct to that constant. Float64 preserves the mathematical IDL formula
with less additional arithmetic rounding; results need not be bit-identical to
IDL's float32 calculations. Quantization and the text log's limited precision
still limit physical precision.

Source verification (workspace-relative references):

- `upstream/p3d-kglobal/kglobal/src/movie.F90`, `volume_z`, lines 34–44,
  54–85: ghost-cell exclusion, signed int16 encoding, global min/max, and writes.
- `upstream/p3d-kglobal/kglobal/src/movie_kglobal3.0.h`, `movie_open`, lines
  57–70 and 168–181: global subarray with `MPI_ORDER_FORTRAN`, native byte order.
  `movie_write`, lines 204–238: frame advancement and variable/log order.
- `legacy/idlcodes/movie.pro`, lines 45–71: `intarr(nx,ny)`, frame-byte seek,
  `READU`, and direct x/y indexing without transpose.
- `legacy/idlcodes/movienormalizescriptkglobal`, lines 180–185 and 221–231:
  Bx entry 4, float conversion (line 122), and signed-int16 inverse scaling.

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
Both flags are preserved if present; the Bx reader rejects the combination.

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
- `bx.py`: bounded single-frame Bx reading, 18-entry log validation, and decoding.

The neighboring `upstream/` and `legacy/` trees are read-only references and are
not runtime dependencies. Naming/semantics were checked against
`upstream/p3d-kglobal/kglobal/src/movie_kglobal3.0.h` (lines 57–70, 77–114,
184–208), `upstream/p3d-kglobal/kglobal/scripts/p3d_subm_scripts/exp3d_submit_nersc`
(lines 435–452), and upstream/legacy IDL `moviesetupscript_part1` (upstream lines
84–107; legacy lines 63–86). The sample parameter files are configurations, not
universal defaults; matching a parameter file to a particular run remains the
caller's responsibility.

## Tests

Run from this project directory in a Python environment containing NumPy:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Tests cover filename recognition/rejection, sparse segments and companions,
parameter syntax and derivation, unsupported input, case selection, summary
output, bounded Bx reads, shape/order, log parsing, quantization, and invalid
indices/layouts. Temporary test fixtures are created and removed inside `tests/`.

`test_bx_real.py` uses the read-only external `../validation-data/hcs_large_005`
fixture. Set `KGLOBAL_VALIDATION_CASE` to use another location for this same
fixture. Real-data tests skip if the directory is absent; they ran without skips
in the validation below. They instrument the binary read to assert the exact
sample count, starting offset, and ending offset, and independently seek/read
two-byte samples to verify selected grid points. No full-file binary read is used.

## Milestone 2 validation results

Validated with Python 3.12.14 and NumPy 2.3.5: **31 tests passed**, including the
real fixture. No files under `upstream/`, `legacy/`, or `validation-data/` were
modified. The environment's default Python lacked NumPy, so the existing bundled
Python runtime was used without installing or modifying external dependencies:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /Users/zhiyuyin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m unittest discover -s tests -v
```

For `hcs_large_005`: global grid `8192 × 4096 × 1`, Bx size `1,342,177,280`
bytes, frame size `67,108,864` bytes, exactly **20 frames**. The log has **360**
entries. Validation-only inspection of stdout found **20 actual movie-output
lines**, from `5.0499998724262696` through `5.9999998484272510`, spaced by
approximately `0.05`. No general stdout parser or time-coordinate API was added.

Each call below read just one frame using `byteorder="little"` and returned an
owning, Fortran-contiguous float64 array of shape `(8192, 4096)`.

| Frame | Byte offset | Reconstructed minimum | Reconstructed maximum | Mean |
|---|---:|---:|---:|---:|
| 0 | 0 | -1.3824 | 1.50651 | 0.10000078765166653 |
| 19 | 1,275,068,416 | -1.4281 | 1.486 | 0.09999788661848116 |

Fixed zero-based `(x, y)` samples:

| Point | Frame 0 | Frame 19 |
|---|---:|---:|
| (0, 0) | -0.9609325626001374 | -0.9495535103379873 |
| (1, 0) | -0.9610648084229801 | -0.9496869092851148 |
| (0, 1) | -0.9608443987182422 | -0.9491533134966048 |
| (4096, 2048) | 1.1650512854200046 | 1.14756687113756 |
| (8191, 0) | -0.9608884806591897 | -0.9493756450751506 |
| (0, 4095) | -0.9613733820096133 | -0.9498647745479514 |
| (8191, 4095) | -0.9613733820096133 | -0.9498647745479514 |

These checks verify the explicit little-endian interpretation, offsets,
quantization formula, and x/y ordering against independent scalar reads. They
do not constitute automatic byte-order detection or an independent IDL execution.
