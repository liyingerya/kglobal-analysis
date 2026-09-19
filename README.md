# kglobal-analysis

Milestones 1–9: case discovery, parameter metadata, time-lazy movie sequences,
eager one-frame xarray access, strictly aligned multi-variable lazy Datasets,
and read-only movie metadata validation. Reduced energy-spectrum text products
are also supported. The movie format remains the
standard 18-variable double-byte schema. Requires Python >=3.10, NumPy >=1.23,
xarray >=2024.7.0, and dask[array] >=2024.7.0.

Case construction and `inspect()` read only parameter text. Explicit frame reads
load one bounded frame; timeline construction reads sizes and stdout metadata,
not binary samples. All case access is read-only. Each lazy time chunk contains
one whole spatial frame. No spatial chunking, region reader, four-byte reader,
particle/checkpoint reader, plotting, or physics analysis is provided.

The decoder supports a dimension-aware volume layout. **Synthetic Nz > 1 layouts
are tested; real-data validation remains 2D.** This does not validate real 3D
KGlobal simulations. Existing Bx APIs preserve their original 2D behavior.

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
`PYTHONPATH=src python` with the project dependencies already available. The
metadata-only APIs do not import NumPy, xarray, or Dask. NumPy readers do not
import xarray/Dask; xarray and graph-building imports happen when requested.

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

## Generic movie API (Milestone 5)

```python
case = KGlobalCase("/path/to/case")
frame = case.read_movie_frame("jihpar", "004", 0, byteorder="little")
count = case.movie_frame_count("jihpar", "004")
segment = case.movie_segment("jihpar", "004", byteorder="little")
print(segment.variable.storage_name, segment.frame_count, segment.times)
frame = segment.read_time(4.05)

series = case.movie("jihpar", byteorder="little")
print(series.suffixes, series.frame_count, series.times)
print(series.locate_time(5.00))  # ("004", 19) for the real fixture
frame = series.read_time(5.00)
# case.movie("bx", ...), case.movie("ni", ...), etc. use the same machinery.
```

The exported `MovieSegment(case, variable, suffix, *, byteorder)` and
`MovieSeries(case, variable, *, byteorder)` implement the same time semantics
as the Bx interfaces below. Only segments containing the selected variable
participate. Missing/invalid stdout, overlaps, and cadence gaps are errors;
there is no interpolation or nearest-time fallback. Every read delegates to
one generic decoder. No arrays are concatenated or cached.

`STANDARD_MOVIE_FORMAT` is an explicit `MovieFormat` for `movie_kglobal3.0.h`.
Its immutable `VariableSpec` records contain `storage_name` and `log_index`:

```text
0 ni       1 jix       2 jiy       3 jiz       4 bx       5 by
6 bz       7 pi        8 neh       9 pehpar   10 pehperp 11 epar
12 pc     13 jhpar    14 nih      15 pihpar   16 pihperp 17 jihpar
```

These are legacy storage identities, **not canonical physics names**. No species
vocabulary or aliases are defined. Future semantic names can map to these records
without changing decoding. An unknown variable is rejected even when a file with
that name exists. Only this schema and `double_byte` are accepted; `four_byte`,
`movie_header2`, `mult_species`, `heatfluxmovies`, unknown headers, and incorrect
log entry counts are rejected. This milestone does not add a custom-format plugin
interface.

`VolumeLayout(Nx, Ny, Nz)` exposes `shape`, `samples_per_frame`, `frame_bytes`,
`public_shape`, and `frame_count(size_bytes)`. Each frame has `Nx*Ny*Nz` signed
int16 samples and occupies `Nx*Ny*Nz*2` bytes. Disk order is x fastest, then y,
then z, then frame. Zero-based sample `(f,x,y,z)` is at:

```text
byte offset = 2 * (f*Nx*Ny*Nz + x + Nx*(y + Ny*z))
```

The generic reader returns an owning, Fortran-contiguous float64 array of shape
`(Nx, Ny, Nz)` for `Nz > 1`, and `(Nx, Ny)` for `Nz == 1`. Only the z singleton is
removed; x/y singleton axes remain. Byte order is always explicit. Quantization
and full-log validation are shared with the old Bx API. Coordinates and physical
normalizations are not inferred.

The compatibility functions/classes `read_bx_frame`, `BxSegment`, and `BxSeries`
retain their signatures and 2D-only guard. They wrap/subclass the generic code,
not a second decoder or timeline implementation. Use the generic movie API to
exercise synthetic volumetric data. For unchanged on-disk contracts, producer
hardware has no effect on reader selection.

## One materialized xarray frame (Milestone 6)

```python
series = case.movie("bx", byteorder="little")
da = series.read_time_xarray(6.05)
# Global index uses physical-time ordering, never suffix spelling:
da = series.read_frame_xarray(20)  # 006/frame0 for the 005+006 fixture
print(da.name, da.dims, da.shape)  # bx, ('x', 'y'), (8192, 4096)
print(da.coords["time"].item())   # actual stdout time, not the rounded request
print(da.attrs["source_segment"], da.attrs["local_frame_index"])
value = da.isel(x=100, y=50)      # spatial selection by integer index

jihpar = case.movie("jihpar", byteorder="little")
da = jihpar.read_time_xarray(5.00)

segment = case.movie_segment("jihpar", "004", byteorder="little")
da = segment.read_frame_xarray(19)  # local, not global, frame index
# segment.read_time_xarray(5.00) is also available.
```

Each call makes exactly one existing NumPy frame read and wraps that materialized
array in an [xarray.DataArray](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html).
The wrapper does not concatenate or cache other frames, copy the NumPy payload,
or implement another decoder. Existing `read_time()`/`read_frame()` methods
continue to return NumPy arrays. `BxSeries` and `BxSegment` inherit these thin
wrappers while retaining their existing APIs and 2D restriction.

For `Nz == 1`, dimensions are `("x", "y")`; for `Nz > 1` they are
`("x", "y", "z")`. x/y singleton axes are retained. **These are axis labels,
not physical spatial coordinates.** There are no x/y/z coordinate arrays,
units, assumed origins, or cell-center/node/staggering claims. Use `.isel()`;
physical-space `.sel(x=..., y=...)` is not part of this contract.

The sole coordinate is scalar `time`, taken from validated stdout. It is not a
time dimension or a global `.sel(time=...)` interface. Time requests retain the
existing equality tolerance and reject missing times without nearest-neighbor
fallback or interpolation. Even an index-based xarray read requires valid stdout
to attach a trustworthy timestamp. Missing/invalid time metadata raises
`TimeMetadataError` before reading samples; ordinary segment NumPy index reads
remain available without stdout.

Attrs are restricted to known storage metadata: `storage_name`, `movie_header`,
`encoding="double_byte"`, `source_segment`, `local_frame_index`, and
`storage_order`. Series results also include `global_frame_index`. Encoding here
describes the source file; decoded data remains float64. Storage names are not
canonical physics/species names. Local/global indices are zero-based;
negative or out-of-bounds indices raise `IndexError`, and noninteger/bool indices
raise `TypeError`.

The new direct dependency is `xarray>=2024.7.0`; the declared minimum supports
Python 3.10 ([release metadata](https://pypi.org/project/xarray/2024.7.0/)). Its
required transitive dependencies are resolved by pip; no optional parallel extras
or Dask were requested in Milestone 6. Milestone 7 adds the time-lazy
`to_xarray()` interface below while preserving these eager methods. Real
validation remains 2D; only synthetic volumes establish 3D axis ordering.

## Time-lazy xarray series (Milestone 7)

```python
series = case.movie("bx", byteorder="little")
da = series.to_xarray()             # no movie-sample reads
print(da.dims, da.shape, da.chunks)  # metadata only

selected = da.isel(time=40)         # still lazy; for complete 004+005+006
print(selected.source_segment.item(), selected.local_frame_index.item())
frame = selected.compute()         # reads just this one whole frame
# selected.load() or selected.values also trigger this selected frame's read.
```

`MovieSeries.to_xarray()` constructs one delayed call to the existing segment
reader per global frame, wraps each with `dask.array.from_delayed`, and stacks
those tasks along time. It never invokes the decoder to infer dtype or shape.
The [Dask from_delayed API](https://docs.dask.org/en/stable/generated/dask.array.from_delayed.html)
represents each delayed result as one array chunk. Construction and ordinary
indexing perform zero sample reads. Computing selected times executes only the
needed frame tasks. Existing NumPy and eager one-frame xarray APIs are unchanged;
`case.bx(...).to_xarray()` inherits the same generic adapter.

For `Nz == 1`, dimensions are `("time", "x", "y")` and chunks are
`((1, ...), (Nx,), (Ny,))`. Volumes use `("time", "x", "y", "z")` and
`((1, ...), (Nx,), (Ny,), (Nz,))`. There is no spatial chunking: selecting a
single spatial point still reads/decodes its whole source frame. Graph planning
lives in `lazy.py`, separate from the binary reader. A future region loader can
change task granularity without changing the schema, variable names, or timeline.
Whole-frame tasks are a current implementation policy, not a permanent semantic
constraint.

Time and provenance are small eager coordinates along time:

- `time`: actual validated stdout timestamps in physical order.
- `source_segment`: original suffix strings, including zero padding.
- `local_frame_index`: zero-based index in the source segment.
- `global_frame_index`: zero-based index in the physical timeline.

Series attrs contain only `storage_name`, `movie_header`, `encoding`, and
`storage_order`. Varying provenance is never stored as a single series attr.
No x/y/z coordinates, units, species identity, staggering, or canonical physics
names are invented. Field values remain float64 after decoding.

### Exact labels versus tolerant KGlobal lookup

Native xarray/pandas label lookup does **not** inherit `MovieSeries.locate_time()`
tolerance. Stored times are not rounded to nominal cadence. Use the exact stored
label when selecting through xarray:

```python
stored_time = series.times[40]
selected = da.sel(time=stored_time)  # lazy, exact stored label
frame = selected.compute()
```

For example, real `6.049999847164145` is retained; `.sel(time=6.05)` raises
`KeyError`. For a human-entered label, the existing eager
`series.read_time_xarray(6.05)` remains available. If using native xarray nearest
selection, explicitly bound it by the existing global time tolerance:

```python
spacing = min((b-a for a, b in zip(series.times, series.times[1:])),
              default=float("inf"))
tolerance = min(1e-6, spacing / 1000)
selected = da.sel(time=6.05, method="nearest", tolerance=tolerance)
```

Never omit that tolerance for approximate selection. Midpoints and absent times
outside the bound must fail. No interpolation, timestamp canonicalization, or
unrestricted nearest-time convenience API is added by this package. The returned
object is an ordinary DataArray, so callers remain responsible for any native
xarray operations they choose to invoke.

### Materialization and memory

Select times **before** `.compute()`, `.load()`, `.values`, or NumPy conversion.
A whole 60-frame Bx series represents 15 GiB of decoded float64 values; constructing
its graph does not allocate that payload, but computing the full series would.
Each selected real frame reads 64 MiB of int16 samples and decodes to 256 MiB.
Computing several selected frames can execute them concurrently and requires
memory for those frames. `compute(scheduler="synchronous")` is available for
serial task execution. Separate computations can reread frames; this adapter
adds no data cache. Keep source files unchanged while using a case/series graph,
and rebuild snapshots after files change.

The only new direct dependency is `dask[array]>=2024.7.0`, whose minimum release
supports Python 3.10 ([release metadata](https://pypi.org/project/dask/2024.7.0/)).
The existing `xarray>=2024.7.0` requirement is unchanged. No distributed scheduler
package or unrelated optional extras are requested.

## Strict multi-variable movie Dataset (Milestone 8)

```python
ds = case.movie_dataset(["bx", "jihpar"], byteorder="little")
print(list(ds.data_vars))          # caller order: ['bx', 'jihpar']
print(ds.sizes)                    # metadata only
selected = ds.isel(time=19)        # still lazy
frame = selected.compute(scheduler="synchronous")  # one frame per variable

# Selecting a variable culls the other variables' read tasks:
bx_frame = ds["bx"].isel(time=19).compute()
# Two nonadjacent times execute only those variable/time tasks:
# small = ds[["bx", "jihpar"]].isel(time=[0, 19]).compute()
```

The API accepts an ordered iterable of legacy storage names, including generators.
Caller order is preserved in `Dataset.data_vars`. Empty collections and duplicates
raise `ValueError`. A bare string, an unordered set, or non-string names raise
`TypeError`. Unsupported names continue to fail through `MovieFormat` validation.
No new physics names or species semantics are introduced.

Each requested variable first gets an ordinary validated `MovieSeries`. Before
any xarray graphs are composed, the layer checks exact equality of frame count,
actual stdout timestamps, and the source-segment/local-frame pair at every global
index. There is no floating-point tolerance in this alignment check: even a small
timestamp difference is incompatible. Per-series time lookup tolerance is unchanged.

Missing variable coverage, incompatible timelines/provenance, or invalid stdout
time metadata raise the exported `MovieAlignmentError` (a `ValueError` subclass),
with variable/reason context. Ordinary unsupported-format and binary-layout errors
retain their existing errors. No union, intersection, truncation, or NaN filling
is performed. An alignment error is not a physical NaN in the simulation.

```python
from kglobal_analysis import MovieAlignmentError

try:
    ds = case.movie_dataset(["bx", "jihpar"], byteorder="little")
except MovieAlignmentError as error:
    print(error)
```

For example, a case containing Bx 004/005/006 but only jihpar 004 is incompatible.
To examine common 004 data, explicitly construct a case view containing the two
004 variables and their companions (as the real tests do with temporary links).
The Dataset API does not silently choose common coverage.

Once validated, each variable is built through its existing `to_xarray()` adapter.
Composition uses the lazy data variables and one checked coordinate set, avoiding
xarray's implicit timeline alignment. Shared coordinates `time`, `source_segment`,
`local_frame_index`, and `global_frame_index` remain small eager metadata.
Every variable has dimensions `("time", "x", "y")`, or
`("time", "x", "y", "z")` for a volume. No spatial coordinates are invented.
Dataset attrs are only `movie_header`, `encoding`, and `storage_order`; each data
variable retains its own `storage_name` and existing storage attrs.

Construction and indexing read zero movie samples. Computing one time reads one
whole frame for each selected variable. Computing `ds["bx"]` at one time executes
no jihpar task. Exact stored time labels work with `.sel(time=stored_time)`;
rounded decimal labels do not acquire KGlobal's tolerant lookup semantics. The
Dataset adds no nearest-time convenience method; the same explicitly bounded
native-xarray guidance above applies.

**Select variables and times before materialization.** A two-variable, 20-frame
8192×4096 Dataset represents 10 GiB of float64 values. Do not compute/load/convert
the full real Dataset just to inspect it. One selected two-variable time slice
still requires 512 MiB for its decoded values, plus read/compute working memory.
Time-only chunking is unchanged: even
`ds["bx"].isel(time=19, x=100, y=50).compute()` reads a complete source frame.
No region reads or spatial chunks are added. Real validation remains 2D, with
synthetic Nz > 1 tests only. This milestone adds no dependency.

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
- `format.py`: explicit storage schema, configuration restrictions, and volume layout.
- `movie.py`: one generic bounded decoder, log selection, and inverse quantization.
- `bx.py`: compatibility wrappers for the original 2D Bx API.
- `times.py`: actual stdout event parsing and time-metadata validation.
- `segment.py`: generic segment/time lookup, one-frame xarray wrapping, and Bx compatibility subclass.
- `series.py`: generic timeline/boundary validation, global frame mapping for xarray, and Bx compatibility subclass.
- `lazy.py`: time-lazy Dask/xarray graph adapter, separate from binary decoding.
- `dataset.py`: strict timeline/provenance checks and lazy multi-variable composition.

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

## Milestone 5 validation results

All **80 tests passed, with no skips**, using Python 3.12.14 and NumPy 2.3.5.
The original 65 tests are unchanged, including independent real Bx values and
bounded reads for 005/006. New tests cover all 18 schema mappings and distinct
file/log selection, generic timelines and delegation, rejected layouts/variables,
and a two-frame 3×2×2 volume. Fixed voxels verify x-fastest ordering and 24-byte
frame offsets, beyond aggregate statistics. This is synthetic 3D validation only.

Before relying on `movie.jihpar.004`, file inspection established:

| Property | Observed value |
|---|---:|
| Exact binary size | 1,342,177,280 bytes |
| Global shape | 8192 × 4096 × 1 |
| Frame size | 67,108,864 bytes |
| Complete frames / remainder | 20 / 0 bytes |
| Log entries | 360 (18 per frame) |
| Actual stdout movie events | 20 |
| First / last time | 4.0499998976883944 / 4.9999998736893758 |
| jihpar log index | 17 |

Real jihpar frames 0 and 19 decode successfully; time-based reads at 4.05 and
5.00 match direct index reads element-for-element. Each read is instrumented to
verify exactly one 64 MiB binary read, the correct filename, and exact start/end
offsets. Independent two-byte seeks at five fixed grid points check the values
against log entry 17. Reconstructed ranges are [-0.356986, 0.479592] and
[-0.486011, 0.508915], respectively. Real Bx 004 frames 0/19 also match between
compatibility and generic calls through the same decoder.

`test_movie_real.py` creates temporary read-only symlinks inside `tests/` to
004's Bx/jihpar/log/stdout files in `hcs_large_multi` and the parameter file in
`hcs_large_005`. It uses the same fixture-location environment overrides as the
older real tests. Temporary views are removed. No reference or validation-data
files were modified, and no dependency was added. Future real 3D output still
requires validation, but does not require a new case/segment/time abstraction.

## Milestone 6 validation results

All **95 tests passed with no skips**: the existing 82 tests plus 11 synthetic
and two real-data xarray tests. Validation used Python 3.11.13, NumPy 2.4.6,
and the declared minimum xarray 2024.7.0 in a project-local `.venv`.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
  -m unittest discover -s tests -v
```

| Real request | Source | Global index | Scalar stdout time |
|---|---|---:|---:|
| Bx `read_time_xarray(6.05)` | 006, local frame 0 | 20 in the 005/006 sequence | 6.049999847164145 |
| jihpar `read_time_xarray(5.00)` | 004, local frame 19 | 19 | 4.999999873689376 |

Both results are named `DataArray` objects with dimensions `("x", "y")`, shape
`(8192, 4096)`, and float64 NumPy data. They match direct NumPy reads
element-for-element, including `.isel(x=100, y=50)`. Instrumentation verifies
one bounded 67,108,864-byte (64 MiB) binary read per xarray request, the correct
file, and exact offsets. Only temporary links inside `tests/` are created; the
real fixtures remain read-only.

Synthetic tests cover all 18 storage-variable names, global ordering opposite
to suffix order, segment-boundary mapping, local/global bounds, stored versus
rounded timestamps, no nearest-time fallback, missing stdout, inherited Bx
wrappers, absence of spatial/physics metadata, and sharing the decoded NumPy
payload without copying. The two-frame 3×2×2 fixture checks fixed voxels and a
nonzero local-frame byte offset; its dimensions are `("x", "y", "z")`. Real
3D KGlobal output remains unvalidated.

Only xarray was added as a direct dependency; its required transitive dependencies
were installed in the isolated environment without parallel extras or Dask.
Existing NumPy APIs and the decoder are unchanged. No full-series xarray,
physical coordinates, simulation changes, or Milestone 7 work was added.

## Milestone 7 validation results

All **106 tests passed with no skips**: the unchanged 95-test suite plus nine
synthetic lazy tests and two real-data lazy tests. Validation used Python
3.11.13, NumPy 2.4.6, xarray 2024.7.0, and Dask 2024.7.0 in `.venv`, testing
both declared xarray/Dask minimum versions together. No `distributed` package
was installed.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
  -m unittest discover -s tests -v
```

Real Bx 004/005/006 was assembled using temporary links, without copying or
changing source data. The resulting lazy array has:

- 60 frames, dimensions `("time", "x", "y")`, shape `(60, 8192, 4096)`.
- Chunks `((1,) * 60, (8192,), (4096,))`.
- Actual stdout time coordinates and per-time source/local/global indices.
- Zero sample reads during series/array construction and time selection.

The test derives the index for `locate_time(6.05)` from the constructed timeline
and verifies it is global frame 40, segment `006`, local frame 0. Computing only
that selected frame invokes one bounded 67,108,864-byte read and matches the
existing eager `read_time_xarray(6.05)` values element-for-element. Native exact
`.sel(time=6.05)` fails because the stored label is `6.049999847164145`; selecting
that actual label works. The full 60-frame payload was never computed.

Real jihpar 004 exposes lazy shape `(20, 8192, 4096)` and whole-frame time chunks.
Construction/selection reads zero samples. Computing only frame 19 reads one
64 MiB raw frame and matches the direct generic NumPy decoder exactly.

Synthetic tests verify zero-read construction/indexing, one-read single-time
compute, and exactly two reads for two nonadjacent time indices across reversed
suffix/physical ordering (`090` before `002`). They also test selected `.load()`
and `.values`, eager provenance only, exact labels versus explicitly bounded
nearest selection, the inherited Bx entry point, and whole-frame reads even for
spatial-point selection. The synthetic 3×2×2 volume preserves fixed-voxel order
with dimensions `("time", "x", "y", "z")` and chunks
`((1,1,1,1), (3,), (2,), (2,))`; selecting one time computes one frame.

Only `dask[array]>=2024.7.0` was added as a direct dependency. Existing eager
readers, storage schema, and time lookup semantics are unchanged. Real 3D output
remains unvalidated. No spatial chunking, region reads, new physical semantics,
or subsequent milestone work was implemented.

## Milestone 8 validation results

All **119 tests passed with no skips**: the unchanged 106-test suite plus eleven
synthetic Dataset tests and two real Dataset tests. The environment remains
Python 3.11.13, NumPy 2.4.6, xarray 2024.7.0, and Dask 2024.7.0; no dependency
was added or changed.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
  -m unittest discover -s tests -v
```

Synthetic read-count checks with the synchronous scheduler establish:

| Operation | Movie frame reads |
|---|---:|
| Dataset construction and time indexing | 0 |
| One time, both variables | 2 (one per variable) |
| One time, Bx only | 1 (no jihpar read) |
| Two nonadjacent times, both variables | 4 (only selected variable/time pairs) |

Alignment tests reject missing segments, inconsistent binary/stdout frame counts,
nonidentical timestamps (including a 1e-9 difference), different source suffixes,
and an injected different local-index mapping before graph construction or sample
I/O where applicable. The local-index test injects metadata because independent
local mappings under identical suffix/stdout files cannot otherwise be represented
by the current filesystem model. Tests also cover empty/duplicate/unknown names,
ordered generators, shared eager provenance, exact stored labels, and restricted
attrs. No xarray union/intersection is used.

Real tests link only Bx/jihpar/log/stdout 004 and the matching parameter file into
a temporary case view. Each lazy variable has shape `(20, 8192, 4096)`, dimensions
`("time", "x", "y")`, and chunks `((1,)*20, (8192,), (4096,))`. Construction and
selection perform zero sample reads. Computing the two-variable slice at frame 19
(actual stdout time approximately `4.999999873689376`) performs one bounded
67,108,864-byte read for Bx and one for jihpar. Both match the existing generic
NumPy reader element-for-element. Separately computing only Bx frame 19 performs
one Bx read and zero jihpar reads. The full real Dataset is never materialized.

The two-variable synthetic 3×2×2 Dataset has time/x/y/z dimensions and no spatial
coordinate values. Selecting one time and jihpar computes one 12-sample frame;
fixed voxels confirm x-fastest Fortran ordering. Spatial-point selection in 2D
still reads the whole six-sample synthetic source frame, as intended.

Only Dataset composition, validation, tests, and documentation were added.
Existing decoders and eager/lazy single-variable APIs remain unchanged. All
reference/validation data remains read-only. No spatial chunking, plotting,
physics diagnostics, or subsequent milestone work was implemented.

## Movie validation and provenance manifest

```python
from kglobal_analysis import KGlobalCase

case = KGlobalCase("/path/to/case")
report = case.validate_movie_case()
print(report.summary())
print(report.ok, report.errors, report.warnings)
for variable in report.variables:
    print(variable.storage_name, variable.suffixes,
          variable.total_frame_count, variable.first_time, variable.last_time)
metadata = report.to_dict()  # JSON-compatible; no file is written
```

`MovieCaseReport` and its nested summaries are frozen dataclasses:

- `parameters`: parameter-file path and parsed definitions, global dimensions,
  dt, n_movieout, nominal cadence, header, encoding flags, selected schema names,
  and bytes per frame when the format/dimensions are supported.
- `segments`: suffix, binary paths/sizes/frame counts, stdout path/event count/
  validated times, and log path/entry count/frame count/validation status.
- `variables`: storage name, suffixes (physical-time order when valid), counts
  per segment, total frame count, timeline validity, and first/last valid times.
- `errors` / `warnings`: issues with stable category codes, messages, and
  applicable suffix, storage name, or path. `absent_schema_variables` lists
  missing standard variables. `.ok` means there are no metadata errors.

Errors include partial binary frames, incompatible binary/stdout/log counts,
invalid timestamps or min/max ranges, missing required companions, unsupported
formats, and overlapping, duplicate, gapped, or cadence-inconsistent variable
segments. Missing standard variables and narrower internally valid coverage are
coverage facts/warnings: all 18 variables are **not** required. Companion-only
segments are retained. The report never repairs data or aligns variables into a
Dataset; strict `movie_dataset()` alignment remains unchanged.

Counts come from `VolumeLayout` and binary file sizes. Times come exclusively
from actual stdout events, with existing time and `MovieSeries` boundary
validators. Suffixes are identifiers, never physical times. The decoder and
manifest share log-range parsing; log widths come from the selected MovieFormat.
Unknown/unverifiable quantities remain `None`. `stdout_valid` describes the
stdout timeline itself; binary count mismatches are separate errors.
`timeline_valid` describes binary/time coverage independently of log validity.
Malformed parameter syntax still raises during `KGlobalCase` construction.
Construct a new case to rediscover changed files or parameter definitions.

**Zero movie-sample I/O:** validation reads parameter/stdout/log text and stats
binary sizes only. It does not open binary sample streams or compute checksums.
Full binary hashing is deliberately deferred to keep this metadata-fast. An OK
report is not evidence that the sample values are correct or the files unchanged.

Real validation uses a temporary symlink view of Bx 004/005/006, jihpar 004,
three log/stdout pairs, and `param_hcs_large`: 8192 × 4096 × 1, standard
`movie_kglobal3.0.h`, double-byte encoding, nominal cadence 0.05. Bx has 60
frames at approximately 4.05–7.00; jihpar has 20 at approximately 4.05–5.00.
The case is valid with unequal-coverage and absent-variable warnings. Tests
forbid binary opens, NumPy sample reads, and frame-reader calls during manifest
construction. Synthetic volume metadata is tested; real 3D remains unvalidated.

The report records backend-neutral format, dimensions, variables, counts,
actual timestamps, and parameter provenance for future CPU/GPU run comparisons.
It does not infer a backend, units, staggering, spatial coordinates, or species,
and does not perform numerical comparisons between runs.

## Legacy reduced energy spectra

```python
from kglobal_analysis import KGlobalCase

case = KGlobalCase("/path/to/staging")
print(case.energy_spectrum_suffixes("electron"))
e = case.energy_spectrum("electron", checkpoint="016")
i = case.energy_spectrum("ion", checkpoint="016")
print(e.storage_name, e.sentinel_value, e.width_integral)
da = e.to_xarray()
```

This eager reader opens only small reduced text products: `xenergylog.NNN`
(electron), `xenergylogi.NNN` (ion), and the corresponding `vd2dgyro.NNN` log.
It never opens particle checkpoints. Discovery uses exact names with numeric
suffixes, preserves leading zeros, and allows gaps. Reduced-product suffixes
are discovered separately from movie segments; the discovery method rescans
current filenames and can list a spectrum whose required log is missing.

`EnergySpectrum` is a frozen object with tuple-backed numeric sequences and
source/log paths. It preserves all **201 stored entries** in `storage_values`.
Index 0 is exposed as `sentinel_value`; `values` contains the **200 active bins**.
A nonzero sentinel is retained as evidence. Finite negative values and spectra
whose integral differs from one also remain unchanged.

Limits come from `Emin/Emax` or `Imin/Imax` in the reducer log, including Fortran
D exponents. The 201 producer edges use
`Emin * (Emax/Emin)**(j/200)` for j=0..200, with endpoints fixed to the parsed
limits. The denominator is **200, not 201**. `lower_edges`, `upper_edges`,
`widths`, and `centers` are available; geometric centers are Python-derived
plotting coordinates, not explicitly stored producer coordinates.

`to_xarray()` returns a detached DataArray named by the provisional quantity
`energy_spectrum`, with dimension `energy_bin`, indices 1..200, and coordinates
`energy_lower`, `energy_upper`, `energy_center`, and `energy_width`. Its attrs
retain storage identity, species/category, suffix, paths, limits, and sentinel.
No energy units or physical time are inferred: **checkpoint suffix is not
physical time**, and legacy reduced output does not provide physical time.

The values are **not raw counts**: the legacy producer accumulates 1/K
contributions and applies bin-width normalization. `width_integral` computes
`sum(values * widths)` over active bins as a diagnostic. The reader never
renormalizes or repairs stored values. An integral near one is expected for
normalized output, but is not proof that the legacy reducer was free of
accumulator defects. Real historical 2D reduced files validate format
compatibility and self-consistency only.

`xenergylog` and `xenergylogi` are legacy storage identifiers, not canonical
scientific names. A small explicit mapping selects storage and log labels by
`electron`/`ion`; public quantity naming is separate. Semantic terminology may
evolve (for example, to `energy_distribution`) without changing discovery,
parsing, filenames, or provenance. Future reduced-product APIs should preserve
this separation rather than infer physics from filename spelling.

Missing spectrum files raise `FileNotFoundError`. Invalid numeric content,
nonfinite values, or lengths other than 201 raise `EnergySpectrumError`.
Missing/invalid/duplicate required log limits raise `EnergyMetadataError`
(a subclass of `EnergySpectrumError`). Limits must be finite with
`0 < Emin < Emax` and represent 200 distinct floating-point intervals.
Unsupported species or non-string/non-numeric checkpoint suffixes raise
`ValueError`. No parameter-based fallback for missing log limits is applied.

## Legacy reduced distributions

Six regular-category text products are supported through two provisional
semantic APIs:

```python
case = KGlobalCase("/path/to/staging")
vv = case.parallel_perpendicular_velocity_distribution("electron", checkpoint="016")
xv = case.position_parallel_velocity_distribution(
    "ion", position_axis="x", checkpoint="016",
)
print(vv.storage_name)
print(vv.region_kind)
print(vv.value_sum)
da = vv.to_xarray()
```

| Storage identifier | Species | Provisional quantity | Position axis |
|---|---|---|---|
| `vdeparperp` | electron | `parallel_perpendicular_velocity_distribution` | — |
| `vdiparperp` | ion | `parallel_perpendicular_velocity_distribution` | — |
| `xpepar` | electron | `position_parallel_velocity_distribution` | x |
| `xpipar` | ion | `position_parallel_velocity_distribution` | x |
| `ypepar` | electron | `position_parallel_velocity_distribution` | y |
| `ypipar` | ion | `position_parallel_velocity_distribution` | y |

Storage names are provenance, not canonical scientific terminology. Quantity,
species, storage identity, region selection, and simulation dimensionality are
separate concepts. Exact digit suffix spelling is preserved; gaps are allowed.
**Checkpoint suffix is not physical time.** Each read opens only the selected
reduced text file and matching `vd2dgyro` log, never raw particle checkpoints or
movie samples. Discovery is separate from movie discovery.

The current interpreted contract supports a validated inclusive rectangular
box (`region_kind="box"`). The legacy selector is retained as
`legacy_roi_mode=0` provenance; other modes raise `DistributionMetadataError`.
This is a validation boundary, not a permanent region API. Available normalized
`XMIN/XMAX`, `YMIN/YMAX`, and `ZMIN/ZMAX` are retained in `region_metadata`.
Future region selections may include genuinely 3D volumes. A two-axis reduced
histogram does **not** imply a 2D source simulation or a 2D selected region.
Current x/y position products do not permanently restrict future spatial axes.

Values are **normalized joint bin mass** (`value_semantics="normalized_bin_mass"`)
for the historical regular `weight==1` category. The producer increments by
one, then independently divides each array by its own sum when nonzero.
Values are not raw counts, densities, continuous probability densities, or
gyrotropic phase-space densities. `value_sum` includes every bin and is only a
diagnostic: zero, non-unit sums, and finite negative values are preserved without
repair or renormalization. Position/parallel products do not apply the
perpendicular-index cutoff, so they are **not guaranteed marginals** of the
parallel/perpendicular products.

Every stored bin is active, including zero indices and endpoints. There is no
sentinel row or column. First-index-fastest text is reconstructed in Fortran
order without transposing: `(401, 201)` for parallel/perpendicular and
`(101, 401)` for position/parallel. Wrong counts, malformed/nonfinite values,
missing or ambiguous metadata, unsupported bin counts, and invalid scales or
box extents fail explicitly.

`ReducedDistribution` is frozen, with immutable numeric backing and ordered
`axis_names`, `axis_indices`, and `axis_values`. The log supplies nominal
code-space centers: velocity index times the species scale divided by 200;
position origin plus index times extent divided by 100. The ion scale is
`V_e * (R_i / R_e)`; the logged converted electron scale is not generally the
code-light-speed scale. No parameter fallback is used. Nearest-index endpoint
support can extend nearly half a velocity bin beyond the last nominal center;
position endpoint supports are clipped by the box. Floating-point ties are
producer-dependent. No edges, Jacobian corrections, or density conversions are
provided.

`to_xarray()` returns detached data and coordinates, named by the provisional
quantity. Velocity dimensions are `("v_parallel_bin", "v_perp_bin")` with integer
indices `-200..200`, `0..200` and attached `v_parallel`, `v_perp` coordinates.
Position dimensions are `("position_bin", "v_parallel_bin")` with indices
`0..100`, `-200..200` and attached `position`, `v_parallel` coordinates;
`position_axis` distinguishes x/y in attrs. Attrs retain paths, storage shape,
serialization, region bounds including z when supplied, and reconstruction and
normalization provenance. No units, time coordinate, or simulation-dimensionality
claim is inferred. Historical compatibility does not certify particle-level
numerical correctness.
