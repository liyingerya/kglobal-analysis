# kglobal-analysis

Milestones 1–4: KGlobal case structure/parameter inspection and a global time-aware
sequence of double-byte 2D Bx segments, reading one frame per call. Requires Python 3.10 or later;
the Bx reader uses NumPy (>=1.23).

Case construction and `inspect()` still read only parameter text, never movie,
log, or stdout contents. Binary reading happens only through an explicit
`read_bx_frame()` call. All case access is read-only. No xarray integration,
bulk frame loading, other movie variables, four-byte
reading, particle/checkpoint readers, plotting, or physics analysis is provided.

## Development setup

Python 3.11 is recommended for development. The supported version is Python
3.10 or later; the project has been verified on Python 3.10.8 and 3.11.13.
From this project directory:

```sh
conda create -n kglobal python=3.11
conda activate kglobal
pip install -e .
```

Run the tests in the activated environment:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

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

## One time-aware Bx segment

```python
segment = case.bx_segment("005", byteorder="little")
# Also available as BxSegment(case, "005", byteorder="little").
print(segment.suffix)          # "005"
print(segment.frame_count)     # 20
print(segment.times[0])        # 5.04999987242627, as recorded in stdout
print(segment.times[-1])       # 5.999999848427251
print(segment.cadence)         # 0.05, nominal dt*n_movieout metadata

index = segment.frame_index_at_time(5.50)  # 9
bx = segment.read_time(5.50)               # one frame; equivalent to read_frame(9)
```

`BxSegment` is a snapshot of one segment's frame count and nominal cadence. It
reads and validates the matching `p3d.stdout.NNN` lazily on first time access,
then caches an immutable tuple of timestamps. Construct a new case/segment if
the files change. Segment construction and time lookup do not read Bx samples.
`read_frame(index)` delegates directly to the existing Milestone 2 reader.

Absolute times come **only from actual stdout event lines**:

```text
 movie output, t=   5.0499998724262696
```

Descriptive lines such as `two-byte movie output` are ignored. Event order
defines the mapping: event 0 belongs to frame 0, event 1 to frame 1, etc. The
suffix is a startup-checkpoint/run identifier, not a physical start time. No
absolute times are inferred from it or synthesized from `dt*n_movieout`.

Before exposing times, the segment checks that timestamp count equals binary
frame count, all timestamps are finite and strictly increasing, and every
observed spacing agrees with nominal cadence when available. Cadence comparison
uses `rtol=1e-6` and `atol=1e-10` in simulation time units. Missing cadence inputs
disable only that consistency check; stdout remains the sole source of times.

Time lookup requires equality within a small floating-point tolerance:
`atol = min(1e-6, minimum observed spacing / 1000)`, with `rtol=0`. For a single
frame, `atol=1e-6`. This accepts decimal labels such as `5.05` despite the small
accumulated timestep roundoff in stdout, without rounding the stored times.
There is no nearest-time fallback or interpolation: e.g. `5.075` raises
`KeyError`. Nonfinite requested times raise `ValueError`.

Missing/unreadable stdout, malformed events, count mismatches, nonfinite or
unordered times, and cadence mismatches raise `TimeMetadataError` (a `ValueError`
subclass) on `.times`, `.frame_index_at_time(...)`, or `.read_time(...)`. They
do not disable `.read_frame(...)` or the original low-level API:

```python
case.read_bx_frame("005", frame_index, byteorder="little")
```

## Global Bx timeline

For a case directory containing multiple Bx segments and their stdout files:

```python
bx = case.bx(byteorder="little")
# Also available as BxSeries(case, byteorder="little").
print(bx.suffixes)              # ("005", "006"), in physical-time order
print(bx.frame_count)           # 40
print(bx.times[0], bx.times[-1]) # approximately 5.05, 7.00
print(bx.locate_time(6.00))     # ("005", 19)
print(bx.locate_time(6.05))     # ("006", 0)
frame = bx.read_time(6.05)      # reads only 006's local frame 0
```

Every discovered segment containing `movie.bx.NNN` participates. Companion-only
or other-variable-only segments do not participate. An empty Bx series raises
`ValueError`. All segments use the case's selected parameter file.

Construction snapshots and validates metadata eagerly: Bx file sizes and each
segment's actual stdout times. **No Bx sample or normalization-log contents are
read during construction, `.times`, or `.locate_time(...)`.** Segments are sorted
by their first validated physical timestamp, never by suffix. Within-segment
event order is preserved; sorting does not repair invalid timestamps. Global
times and suffixes are exposed as immutable tuples.

At each boundary the next segment must start after the preceding segment ends.
Overlaps and duplicate times raise `TimeMetadataError`; no samples are discarded.
For multiple segments, resolved `dt*n_movieout` is required to validate boundary
continuity. A larger spacing raises a gap error, and a smaller spacing raises a
cadence-discontinuity error, using the same `rtol=1e-6`, `atol=1e-10` cadence
comparison as Milestone 3. Missing cadence is an error instead of an inferred
cadence. Final global times are also checked for finite, strictly increasing
values. Missing/invalid stdout in any participating segment aborts construction
with an error identifying the segment.

`.locate_time(time)` returns `(suffix, local_frame_index)`. It uses Milestone 3's
equality rule, evaluated over global spacings:
`atol=min(1e-6, minimum_global_spacing/1000)`, `rtol=0`. Missing times raise
`KeyError`; nonfinite requests raise `ValueError`. There is no nearest-time
fallback, interpolation, or synthetic gap filling. `.read_time(time)` delegates
to exactly one existing `BxSegment.read_frame()` call. Physical arrays are not
concatenated or cached; each result is still one owning float64 `(Nx, Ny)` array.

The original `case.bx_segment(...)` and `case.read_bx_frame(...)` APIs remain
unchanged and can still access individual segments when a global timeline fails
validation. Reconstruct the case/series snapshot after files change.

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
nominal interval, not an absolute frame time. There is no `movieout` fallback.
The segment API reads absolute times directly from stdout, including for restarts.

## Organization

- `index.py`: filename parsing and filesystem discovery.
- `parameters.py`: parameter text parsing and simple metadata derivations.
- `case.py`: coordination, parameter-file selection, and printed summaries.
- `bx.py`: bounded single-frame Bx reading, 18-entry log validation, and decoding.
- `times.py`: actual stdout event parsing and time-metadata validation.
- `segment.py`: one Bx segment, absolute-time lookup, and delegated one-frame reads.
- `series.py`: physically ordered global Bx timeline and boundary validation.

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
indices/layouts, stdout times, count/order/cadence errors, and time-based reads.
Temporary test fixtures are created and removed inside `tests/`.

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
approximately `0.05`. At Milestone 2 this was validation-only; Milestone 3 adds
the narrow movie-event time API described above.

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

## Milestone 3 validation results

The complete suite passed **49 tests**, with no skips, using Python 3.12.14 and
NumPy 2.3.5. This includes the original 31 tests and 18 new synthetic/real tests.
The Milestone 2 binary reader and dependencies are unchanged.

Real segment `005` has 20 frames and 20 stdout movie times:

- First time: `5.04999987242627`.
- Last time: `5.999999848427251`.
- Observed spacing: `0.04999999873689376` for every adjacent pair.
- Nominal `dt*n_movieout`: `0.05`.

| Requested time | Frame index |
|---|---:|
| 5.05 | 0 |
| 5.10 | 1 |
| 5.50 | 9 |
| 5.75 | 14 |
| 6.00 | 19 |

Real reads at times `5.05` and `6.00` matched index-based reads of frames 0 and 19
element-for-element. A request at `5.075` was rejected, not mapped to a neighbor.
Synthetic tests also verify missing stdout, count mismatches, malformed/nonfinite
events, duplicate/decreasing times, cadence mismatches, absent cadence metadata,
delegation to the correct frame, and tolerance handling for very small spacings.
No files under `upstream/`, `legacy/`, or `validation-data/` were modified. No
multi-segment or bulk frame-loading API was added.

## Milestone 4 validation results

The full suite passed **65 tests**, with no skips, using Python 3.12.14 and NumPy
2.3.5: the existing 49 tests plus 16 new synthetic/real series tests. Dependencies
and the Milestone 2/3 readers are unchanged.

The workspace stores `005` in `validation-data/hcs_large_005` and `006` in
`validation-data/hcs_large_multi`. The latter also contains `004` and no parameter
file. `tests/test_series_real.py` creates a temporary case directory inside
`tests/`, linking only 005/006 and `param_hcs_large` from the first directory.
The view is removed after each test. No source fixture is edited or moved.
`KGLOBAL_VALIDATION_CASE` and `KGLOBAL_MULTI_VALIDATION_CASE` can override the two
fixture directories. These tests skip only when a fixture directory is absent;
both ran in this validation.

Real combined timeline:

| Property | Result |
|---|---|
| Participating suffixes, physical order | `('005', '006')` |
| Total frame count | 40 |
| First stdout time | `5.04999987242627` |
| Boundary stdout times | `5.999999848427251 -> 6.049999847164145` |
| Boundary spacing | `0.04999999873689376` (nominal `0.05`) |
| Last stdout time | `6.999999823165126` |
| `locate_time(6.00)` | `('005', 19)` |
| `locate_time(6.05)` | `('006', 0)` |

For both boundary requests, `read_time()` matched the direct segment/frame read
element-for-element. Instrumentation verified exactly one `numpy.fromfile` call
with 33,554,432 int16 samples (**67,108,864 bytes = 64 MiB**), at byte offset
1,275,068,416 for 005/frame19 and offset 0 for 006/frame0. Construction and time
lookup tests forbid opening binary/log files and forbid binary reads.

Synthetic tests verify suffix order differing from physical order, frame counts,
global mappings, boundary crossings, overlaps/duplicates, decreasing timestamps,
missing/invalid stdout, gaps, shorter-than-cadence boundaries, missing cadence,
floating-point tolerance, small spacings, and exactly-one-frame delegation.
All changes are inside `kglobal-analysis/`; no xarray, Dask, plotting, particle
reading, or other-variable support was added.
