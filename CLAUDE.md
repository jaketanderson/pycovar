# pycovar

Python port of a MATLAB script that builds a 4D NMR spectrum by covariance
(generalized indirect covariance, GIC) of 3D spectra that share one
dimension. The current test case is an HCcH-TOCSY self-covariance: the 3D
(1Hy, 13C, 1H) is covaried with itself over the directly detected 1H, giving a
4D (H, C, Hs, Cs). Both inputs and outputs are NMRPipe files.

Repo: https://github.com/jaketanderson/pycovar (owner: Jake Anderson,
Frueh lab).

## Files

| File | What it is |
|---|---|
| `covar.py` | The Python port. Self-covariance only (one spectrum used twice). |
| `covariance_4D_rev2fix.m` | The original MATLAB script (rev2, 2018, BH/AK/DF), with debug shape printing and a `use_derivative` switch added. Supports several spectrum pairs, extraction limits and MRS weighting, which `covar.py` does not. |
| `compare_modes.py` | Runs `covar.py` once per mode (`real`, `complex`, `split`). Its docstring's output name is out of date; outputs go wherever `filename_4D` in `covar.py` points. |
| `testing.ipynb` | Scratch notebook for reading outputs with nmrglue. |
| `requirements.txt` | The lab's shared package list (from Luca M.), much broader than this project needs. `covar.py` itself only needs `nmrglue`, `numpy`, `scipy`, `tqdm`. |
| `covar.tar.gz` | Snapshot of `covar.py` + the `.m` file for sharing. Not in git. |
| `{no,yes}_SR_{no,yes}_deriv/` | Output 4Ds for each parameter combination (see below). Not in git (`*.pipe` is ignored). |
| `*.dat`, `H.C.pipe` | 2D projections of a 4D (H.C, H.Hs, C.Cs, ...). They carry MATLAB's header values, so they were probably made from a MATLAB 4D. |

## Paths are Jake's defaults: change them for your own data

Every path in the scripts points into Jake's folders on NMRbox. They work as
is only for Jake. Anyone else must edit them before running:

- **Input spectrum** - `covar.py`: `specdir` and `specfile` near the top.
  `covariance_4D_rev2fix.m`: `filenames` in the "EDIT THESE PARAMETERS" block.
  Default input directory:
  `/public/groups/frueh/jaanderson/data/Ybt/PCP1/apo/nmr/251001_Apo_HNC_PCP1_1402-1482/20/LR/`
  - `HCcH_real.pipe`: 150 (1Hy) x 160 (13C) x 2048 (1H, real)
  - `HCcH_complex.pipe`: the same spectrum with 1H complex (2048 complex points)
- **Output file** - `filename_4D` in both scripts. It is relative to the
  current directory, and its folder must already exist.
- **Python environment** - Jake's conda env is
  `/public/groups/frueh/jaanderson/miniconda3/envs/pycovar` (Python 3.11,
  nmrglue 0.12, numpy 1.26.4, scipy 1.10.1). Other users need their own env
  with `nmrglue numpy scipy tqdm`.
- **MATLAB helpers** - the `.m` script calls `get_header_data` and
  `write_covar2pipe`, which are **not in this repo**. They must be on the
  MATLAB path. Where they live has not been recorded here; ask Jake.

## Running

```bash
python covar.py real      # or: complex, split  (default: real)
python compare_modes.py   # all three modes, one after another
```

Run from the repo root so the relative output paths resolve. In MATLAB, run
`covariance_4D_rev2fix` (it uses `waitbar`, so it expects a display).

Settings are edited in the source, not passed as arguments:

| Setting | `covar.py` | `.m` script | Meaning |
|---|---|---|---|
| Matrix power | `lambda_factor` | `lambda` | 0.5 = matrix square root ("SR"), 1.0 = plain covariance |
| Derivative | `use_derivative` | `use_derivative` | First difference along the shared dimension before the SVD |
| Downsampling | `downsample` | `downsample` | Keep every n-th point per 4D axis |
| Output | `filename_4D` | `filename_4D` | `%` in the name writes planes instead of one file |

The output folders are named after these settings:
`yes_SR` = lambda 0.5, `no_SR` = lambda 1.0, `yes_deriv`/`no_deriv` =
derivative on/off.

## Conventions

- The 3Ds are indexed (I, J, K) and (L, M, K); K is the shared dimension
  and is summed out. The 4D is (I, J, L, M) = (H, C, Hs, Cs). It is written
  in NMRPipe order (M, L, J, I) with X = F2, Y = F1, Z = F3, A = F4.
- Output size = I x J x L x M x 4 bytes + a 2048-byte header, whatever the
  mode. For the default input that is 150 x 160 x 150 x 160 -> 2.3 GB. The
  whole 4D is held in RAM as float32.
- `covar.py` transposes the input with `axes=(2,1,0)` so that 1H is the last
  axis. This is hardcoded for this spectrum, whose 1H is NMRPipe's Z
  dimension. Check the transpose for any other spectrum.

## Modes (how the shared 1H dimension enters)

- `real`: reads a spectrum whose 1H is real. Same math as MATLAB.
- `complex`: reads a spectrum whose 1H is complex and does the SVD of the
  complex matrix. Only the real part of each 4D plane is computed and written.
- `split`: real and imaginary parts placed side by side as separate real
  points. Experimental.

`covar.py` checks the input's 1H quad flag and refuses a spectrum of the
wrong kind for the mode. Each mode reads its own file: `real` reads
`HCcH_real.pipe`, the others read `HCcH_complex.pipe`.

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
