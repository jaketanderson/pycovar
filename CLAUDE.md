# This file is a good recap for AI agents working in this repository for the first time.

# pycovar

Python port of a MATLAB script that builds a 4D NMR spectrum by covariance
(generalized indirect covariance, GIC) of 3D spectra that share one
dimension. The current test case is an HCcH-TOCSY self-covariance: the 3D
(1Hy, 13C, 1H) is covaried with itself over the directly detected 1H, giving a
4D (1Hy, 13C, 1Hys, 13Cs). Both inputs and outputs are NMRPipe files.

Repo: https://github.com/jaketanderson/pycovar (owner: Jake Anderson,
Frueh lab).

## Files

| File | What it is |
|---|---|
| `pycovar/` | The Python port, as a small importable package (`axes`, `io`, `header`, `core`, `cli`). Covaries one spectrum with itself or two spectra that share a dimension. |
| `covar.py` | Shim so `python covar.py ...` still works; same as `python -m pycovar`. |
| `pyproject.toml` | Makes the package installable (`pip install -e .`). |
| `tests/` | pytest suite on tiny synthetic NMRPipe files (`python -m pytest tests`). pytest is an optional dev dependency. |
| `verification/covariance_4D_rev2fix.m` | The original MATLAB script (rev2, 2018, BH/AK/DF), with debug shape printing and a `use_derivative` switch added. Supports several spectrum pairs, extraction limits and MRS weighting, which `pycovar` does not. Leave it and the rest of `verification/` alone: it is the reference for checking the port. |
| `verification/compare_modes.py` | **Out of date.** Runs the old `covar.py <mode>` with hard-coded inputs, which no longer exists. Use `python -m pycovar` once per mode instead. Left unedited on purpose. |
| `testing.ipynb` | Scratch notebook for reading outputs with nmrglue. |
| `requirements.txt` | The lab's shared package list (from Luca M.), much broader than this project needs. `pycovar` only needs `nmrglue`, `numpy`, `scipy`, `tqdm`. |
| `covar.tar.gz` | Old snapshot of `covar.py` + the `.m` file for sharing (before the package). Not in git. |
| `*.dat`, `H.C.pipe` | 2D projections of a 4D (H.C, H.Hs, C.Cs, ...). They carry MATLAB's header values, so they were probably made from a MATLAB 4D. |

## Running

```bash
python -m pycovar SPEC_A.pipe -o out/4D.pipe              # self-covariance
python -m pycovar SPEC_A.pipe SPEC_B.pipe -o out/4D.pipe  # cross-covariance
python -m pycovar --help
```

Put the options after the spectra. Main options: `--mode {auto,real,complex,split}`,
`--lambda-factor` (0.5 = matrix square root, 1.0 = plain covariance),
`--derivative` (first difference along the shared dimension before the SVD),
`--downsample N` or `N N N N` (keep every n-th point per 4D axis), `--labels`,
`--axes-a/--axes-b`, `--no-prompt`, `--overwrite`, `--log/--no-log`. A `%` pair
in the output name writes 2D planes instead of one file.

In MATLAB, run `verification/covariance_4D_rev2fix` (it uses `waitbar`, so it
expects a display). It still has Jake's paths in its "EDIT THESE PARAMETERS"
block, and calls `get_header_data` and `write_covar2pipe`, which are **not in
this repo** (ask Jake).

Environment: Jake's conda env is
`/public/groups/frueh/jaanderson/miniconda3/envs/pycovar` (Python 3.11,
nmrglue 0.12, numpy 1.26.4, scipy 1.10.1). Others need `nmrglue numpy scipy
tqdm` (`pip install -e .` pulls them in).

As a library (`import pycovar` does no I/O and configures no logging; the
package logs to `logging.getLogger("pycovar")`):

```python
import pycovar
a, b = pycovar.load_pair("hcch.pipe")        # or two files; axes_a=/axes_b= to force
res = pycovar.covariance_4D(a, b, downsample=2, lambda_factor=0.5)
pycovar.write_4D(res, "out/4D.pipe")         # res.spec is indexed (I, J, L, M)
```

Example input for Jake:
`/public/groups/frueh/jaanderson/data/Ybt/PCP1/apo/nmr/251001_Apo_HNC_PCP1_1402-1482/20/LR/`
with `HCcH_real.pipe` (150 1Hy x 160 13C x 2048 1H, real) and `HCcH_complex.pipe`
(the same with 1H complex).

## Conventions

- The 3Ds are indexed (I, J, K) and (L, M, K); K is the shared dimension
  and is summed out. The 4D is (I, J, L, M). It is written in NMRPipe order
  (M, L, J, I) with X = F2, Y = F1, Z = F3, A = F4. Default 4D labels come
  from the input headers (e.g. `1Hy 13C 1Hys 13Cs`, an `s` added where
  labels would repeat); the old names are `--labels H C Hs Cs`.
- Output size = I x J x L x M x 4 bytes + a 2048-byte header, whatever the
  mode. For the example input that is 150 x 160 x 150 x 160 -> 2.3 GB. The
  whole 4D is held in RAM as float32.
- **Axes.** The shared dimension is taken to be the direct dimension, header
  block FDF2 (NMRPipe convention, whatever the storage order). Array axis `a`
  is described by block `FDF{FDDIMORDER[2-a]}`; the example files store 1H as
  the slowest axis, so they are permuted with `(2,1,0)`. If something looks
  wrong (shared axis not FDF2 or not the largest, nuclei or ppm windows that
  differ between A and B) the CLI shows the axes and asks; without a terminal
  or with `--no-prompt` it fails and prints the `--axes-a/--axes-b` to use.
  A size mismatch of the shared dimension is always an error (no
  interpolation or truncation).
- **Downsampling** is decimation, applied only to the output: the SVD uses the
  full-resolution data and only the kept 4D points are computed. The
  overflow rescale is computed from the full-resolution factors, so a
  downsampled 4D is a subsample of the full one.
- Not supported (the MATLAB script does): more than two spectra, MRS weighting,
  extraction limits.

## Modes (how the shared dimension enters)

- `real`: reads a spectrum whose shared dimension is real. Same math as MATLAB.
- `complex`: reads a spectrum whose shared dimension is complex and does the
  SVD of the complex matrix. Only the real part of each 4D plane is computed
  and written.
- `split`: real and imaginary parts placed side by side as separate real
  points. Experimental.
- `auto` (default): `real` or `complex` following the header quad flag.

An explicit mode that contradicts the input is refused. A complex shared axis
is combined into complex numbers whether nmrglue already did so (axis stored
fastest) or the file holds interleaved real/imaginary points (as the example
`HCcH_complex.pipe` does).

## Attribution

Attribution of work and ideas is taken seriously in this project. Credit
people and sources wherever reasonable:

- **Git commits**: use authorship and trailers (e.g. `Co-Authored-By`) so
  that everyone who contributed, human or AI, is credited.
- **Code comments**: when an algorithm, formula, or piece of code comes from
  someone else (the original MATLAB script by BH/AK/DF, a paper, a
  colleague's suggestion, an AI-generated suggestion), say so in a comment
  next to it.
- **Docs and notes**: cite the origin of methods, data, and results (papers,
  lab members, upstream scripts) here and in READMEs or notebooks.
- When unsure who deserves credit, ask rather than omit it.

## Known results and gotchas

- **Python matches MATLAB.** Given the same input and settings, `covar.py
  real` reproduced MATLAB's 4D to float32 rounding: 99.9998% of points were
  bit-identical, and the largest difference was ~1e-11 of the maximum
  (checked 2026-10-28). If the outputs differ, check the inputs and settings
  first.
- **The package refactor is bit-identical.** The `pycovar` package reproduced the old `covar.py` exactly at full resolution for `real` and `complex` (data identical, header differing only in the default axis labels; 2026-11-04). The old `covar.py` itself is what was checked against MATLAB.
- **Never give the MATLAB script a complex spectrum.** `read_spectrum`
  ignores quad flags, so it reads interleaved real/imaginary points as
  consecutive real points. This fails silently and gives a meaningless 4D.
- **MATLAB's 4D header has wrong FDMAX/FDMIN** (always 5985812480 /
  -222082304, even though negatives are clipped to 0). FDREALSIZE,
  FDSLICECOUNT and FDF2LB are also carried over from elsewhere. The data is
  correct. When comparing spectra visually, set contour levels by hand rather
  than trusting header-based scaling. Python writes the correct values.
- **`complex` vs `real`:** with the same input, the peak shapes are nearly
  identical (correlation >= 0.999 on strong planes). Intensities are about
  sqrt(2) higher in `complex` mode, and weak/noise regions rise somewhat
  more. A large visual difference usually means the inputs or contour levels
  differ.
- **Rescaling:** both scripts divide by a power of 2 if values could exceed
  2^40. The bound is computed slightly differently (Python also counts
  negative x negative products), so in rare cases the overall intensity can
  differ by a factor of 2.
- **Zero-filling** in the 1Hy dimension counts twice in the 4D (H and Hs):
  doubling it makes the output 4x larger without adding information.
