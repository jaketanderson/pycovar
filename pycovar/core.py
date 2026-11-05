"""Generalized indirect covariance of two 3D spectra sharing one dimension.

The math is a port of the MATLAB script covariance_4D_rev2fix.m (rev2, 2018,
BH/AK/DF). Everything is computed at full resolution; downsampling only
selects which rows of the SVD factors are turned into 4D points.
"""
import logging
import os
import time
from dataclasses import dataclass

import nmrglue as ng
import numpy as np
import scipy.linalg
from tqdm import tqdm

from .axes import ShapeMismatchError
from .header import apply_layout, create_4D_dic

logger = logging.getLogger("pycovar")

MODES = ("auto", "real", "complex", "split")
MAX_LABEL = 8  # NMRPipe's label field


@dataclass
class Covariance4D:
    data: np.ndarray    # float32, indexed (M, L, J, I): C order = NMRPipe order
    header: dict        # NMRPipe header of a single-file data stream
    labels: list        # I, J, L, M labels
    mode: str           # resolved mode: real, complex or split
    scale: float        # data was divided by this power of 2 to avoid overflow
    downsample: tuple

    @property
    def spec(self):
        """View indexed (I, J, L, M)."""
        return self.data.transpose(3, 2, 1, 0)


def default_labels(spec_a, spec_b):
    """Header labels of I, J, L, M; an 's' is added where they would collide."""
    labels = [spec_a.axes[0].label, spec_a.axes[1].label,
              spec_b.axes[0].label, spec_b.axes[1].label]
    for n in (2, 3):
        if labels[n] in labels[:n]:
            labels[n] += "s"
    return [lab[:MAX_LABEL] for lab in labels]


def _expand(lo, hi, cplx):
    """Extremes of the real components a complex factor is split into."""
    if not cplx:
        return lo, hi
    return (np.concatenate([lo.real, -hi.imag]), np.concatenate([hi.real, -lo.imag]))


def _extremes(a, cplx):
    """Per-column min/max of a, as (lo, hi), with real and imaginary parts
    kept separate in complex mode. Complex min/max would compare magnitudes."""
    if cplx:
        lo = a.real.min(axis=0) + 1j * a.imag.min(axis=0)
        hi = a.real.max(axis=0) + 1j * a.imag.max(axis=0)
    else:
        lo, hi = a.min(axis=0), a.max(axis=0)
    return lo, hi


def rescale_scale(U1, U2, s2l, cplx, mps):
    """Power-of-2 divisor that keeps every 4D value below mps.

    Computed from the full-resolution factors, U1 (the I,J rows of U) and U2
    (the L,M rows), so that a downsampled 4D is exactly a subsample of the
    full one. Largest positive value each component can contribute to any
    point: a product of two entries is largest at an extreme of each factor,
    which includes two very negative entries multiplying to a large positive.
    """
    lo1, hi1 = _extremes(U1, cplx)
    lo1, hi1 = _expand(lo1 * s2l, hi1 * s2l, cplx)
    # UX2 is the conjugate transpose of U2: its imaginary part is -Im(U2)
    lo2, hi2 = _extremes(U2, cplx)
    lo2, hi2 = _expand(lo2, hi2, cplx)
    bound = np.max([hi1 * hi2, lo1 * lo2, hi1 * lo2, lo1 * hi2], axis=0).sum()
    ratio = bound / mps
    return 2.0 ** np.ceil(np.log2(ratio)) if ratio > 1 else 1.0


def _as_tuple4(downsample):
    ds = tuple(int(d) for d in np.atleast_1d(downsample))
    if len(ds) == 1:
        ds = ds * 4
    if len(ds) != 4 or min(ds) < 1:
        raise ValueError(f"downsample needs four integers >= 1, got {downsample}")
    return ds


def covariance_4D(spec_a, spec_b=None, *, lambda_factor=1.0, derivative=False,
                  mode="auto", downsample=(1, 1, 1, 1), labels=None,
                  block_rows=500, progress=False):
    """4D covariance of spec_a (I, J, K) with spec_b (L, M, K), summed over K.

    spec_a, spec_b: pycovar.Spectrum (from read_spectrum / load_pair). If
        spec_b is None the spectrum is covaried with itself.
    lambda_factor: 1.0 is plain covariance, 0.5 the matrix square root.
    derivative: first difference along K before the SVD.
    mode: "auto" (real or complex, following the shared dimension), "real",
        "complex" (SVD of the complex matrix; only the real part is kept), or
        "split" (real and imaginary parts side by side as real points;
        experimental).
    downsample: keep every n-th point of each 4D axis (I, J, L, M); one
        integer applies to all four. The covariance itself is not affected.
    labels: four labels for I, J, L, M (default: see default_labels).
    """
    if spec_b is None:
        spec_b = spec_a
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; choose from {MODES}")
    downsample = _as_tuple4(downsample)

    I, J, K = spec_a.shape
    L, M, K2 = spec_b.shape
    if K != K2:
        raise ShapeMismatchError(f"Shared dimension sizes differ: {K} and {K2}")
    if spec_a.is_complex != spec_b.is_complex:
        raise ShapeMismatchError("Shared dimension is complex in only one spectrum")
    if mode == "auto":
        mode = "complex" if spec_a.is_complex else "real"
    if (mode != "real") != spec_a.is_complex:
        kind = "real" if mode == "real" else "complex"
        raise ValueError(f"Mode '{mode}' needs a spectrum whose shared dimension "
                         f"is {kind}, but {spec_a.path} is not")
    cplx = mode == "complex"

    I_4D, J_4D = len(range(0, I, downsample[0])), len(range(0, J, downsample[1]))
    L_4D, M_4D = len(range(0, L, downsample[2])), len(range(0, M, downsample[3]))
    labels = list(labels) if labels is not None else default_labels(spec_a, spec_b)
    if len(labels) != 4:
        raise ValueError("labels needs four entries (I, J, L, M)")
    logger.info("Mode %s, lambda %g, derivative %s, downsample %s", mode,
                lambda_factor, derivative, downsample)
    logger.info("4D shape (I, J, L, M) = %s of full %s; %.2f GB",
                (I_4D, J_4D, L_4D, M_4D), (I, J, L, M),
                I_4D * J_4D * L_4D * M_4D * 4 / 1e9)

    stacked = np.vstack([spec_a.data.reshape(I * J, K, order="F"),
                         spec_b.data.reshape(L * M, K, order="F")])
    # MATLAB reads float32 data into doubles, so work in double precision to match
    stacked = stacked.astype(np.float64 if mode == "real" else np.complex128)
    if derivative:
        stacked = np.diff(stacked, n=1, axis=1)
    if mode == "split":
        stacked = np.hstack([stacked.real, stacked.imag])

    logger.info("Performing SVD of a %s matrix...", stacked.shape)
    t0 = time.time()
    U, S, _ = scipy.linalg.svd(stacked, full_matrices=False)
    logger.info("SVD completed in %.1f s", time.time() - t0)
    del stacked

    # Scale data to keep peak heights reasonable and prevent float overflow
    n_pairs = 1
    mps = 2 ** (40 / n_pairs)  # Max value allowed per spectrum
    S2l = S ** (2 * lambda_factor)
    scale = rescale_scale(U[:I * J], U[I * J:], S2l, cplx, mps)
    if scale != 1.0:
        logger.info("Dividing by %g so values stay below 2^40", scale)

    # Row indices to keep in the downsampled 4D (column-major, like MATLAB)
    IJ_write = (np.arange(0, I, downsample[0])[:, None]
                + np.arange(0, J, downsample[1])[None, :] * I).ravel(order="F")
    LM_write = (np.arange(0, L, downsample[2])[:, None]
                + np.arange(0, M, downsample[3])[None, :] * L).ravel(order="F")
    UX1_S = U[IJ_write, :] * S2l
    UX2 = U[I * J + LM_write, :].conj().T
    del U  # Clear some memory
    if cplx:
        # Only the real part of UX1_S @ UX2 is kept, which is this product of
        # real matrices with twice as many components
        UX1_S = np.hstack([UX1_S.real, -UX1_S.imag])
        UX2 = np.vstack([UX2.real, UX2.imag])
    if scale != 1.0:
        # A power of 2 scales exactly, so it does not matter which one
        if UX1_S.size < UX2.size:
            UX1_S = UX1_S / scale
        else:
            UX2 = UX2 / scale

    # Row lm of UX2.T @ UX1_S.T is one flattened (I_4D x J_4D) plane. Calculate
    # it a block of rows at a time, straight into the float32 array that gets
    # written, so the full product never exists in float64.
    # NMRPipe stores float32 with the first dimension varying fastest, which
    # is C order for an array indexed (M, L, J, I)
    data = np.empty((L_4D * M_4D, I_4D * J_4D), dtype=np.float32)
    starts = range(0, L_4D * M_4D, block_rows)
    if progress:
        starts = tqdm(starts, desc="Calculating blocks...")
    for start in starts:
        block = UX2.T[start:start + block_rows] @ UX1_S.T
        # Anything negative is an artifact
        np.maximum(block, 0, out=data[start:start + block_rows], casting="same_kind")
    data = data.reshape(M_4D, L_4D, J_4D, I_4D)

    blocks = [spec_a.axes[0].block, spec_a.axes[1].block,
              spec_b.axes[0].block, spec_b.axes[1].block]
    header = create_4D_dic([spec_a.dic, spec_b.dic], blocks, [I, J, L, M],
                           downsample, labels, data)
    return Covariance4D(data, header, labels, mode, scale, downsample)


def write_4D(result, path, overwrite=False):
    """Write a Covariance4D as NMRPipe data.

    A path without '%' gives one file holding the whole 4D (a data stream).
    A path with two '%' gives 2D planes. One '%' is not supported (MATLAB
    writes 3D cubes for it, but nmrglue would write 2D planes).
    """
    path = str(path)
    n_pct = path.count("%")
    if n_pct == 1:
        raise ValueError("One '%' is not supported: MATLAB writes 3D cubes for "
                         "it, but nmrglue would write 2D planes")
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    dic = apply_layout(dict(result.header), single_file=(n_pct == 0))
    if n_pct == 0:
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True")
        # Same file as ng.pipe.write makes, but without its two extra
        # in-memory copies of the data
        with open(path, "wb") as f:
            f.write(ng.pipe.dic2fdata(dic).tobytes())
            result.data.tofile(f)
    else:
        ng.pipe.write(path, dic, result.data, overwrite=overwrite)
    logger.info("Wrote %s", path)
