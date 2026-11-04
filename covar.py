# covar.py
#
# v0.0
# Jake Anderson, October 2026

import sys

import nmrglue as glue
import numpy as np
import scipy

from tqdm import tqdm

# Currently this script only does self-covariance (for quick testing)

# How the shared dimension enters the covariance:
#   "real"    - a real spectrum is read and used as is (same as the MATLAB script)
#   "complex" - SVD of the complex matrix; the real part of each plane is kept
#   "split"   - real and imaginary parts side by side as separate real points. This is weird and for testing.
# Pass it on the command line (e.g. "python covar.py complex"); the default is "real"
# If you are using anything but "real", you will need to provide 3Ds that have complex values in the covaried dimension


lambda_factor = 1.0  # Use 0.5 for matrix square rooting
downsample = [1, 1, 1, 1]
use_derivative = False


mode = sys.argv[1] if len(sys.argv) > 1 else "real"
if mode not in ("real", "complex", "split"):
    raise ValueError(f"Unknown mode: {mode}")

specdir = "/public/groups/frueh/jaanderson/data/Ybt/PCP1/apo/nmr/251001_Apo_HNC_PCP1_1402-1482/20/LR/"
specfile = specdir + ("HCcH_real.pipe" if mode == "real" else "HCcH_complex.pipe")
specheader, specarr = glue.fileio.pipe.read(specfile)
# This is a fragile fix for my one spectrum! Needs fixing for generality.
specheader, specarr = glue.fileio.pipe.transpose_3D(specheader, specarr, axes=(2,1,0))  # Transpose so 1H is third dim
udic = glue.pipe.guess_udic(specheader, specarr)
if udic[2]["complex"] != (mode != "real"):
    kind = "real" if mode == "real" else "complex"
    raise ValueError(f"Mode '{mode}' needs a spectrum whose 1H dimension is "
                     f"{kind}, but {specfile} is not")
if mode != "real":
    # Complex 1H is stored as alternating real and imaginary points
    specarr = specarr[..., 0::2] + 1j * specarr[..., 1::2]
print(f"{'Spectrum dimensions:':<22}{specarr.shape}")
dimlabels = [udic[i]["label"] for i in range(udic["ndim"])]
print(f"{'Dimension labels:':<22}{dimlabels}")

def rescale_pair(UX1_S, UX2, mps):
    """Scale one pair's factors so no plane value can exceed mps."""
    # Largest positive value each component can contribute to any plane. A
    # product of two entries is largest at an extreme of each factor, which
    # includes two very negative entries multiplying to a large positive.
    lo1, hi1 = UX1_S.min(axis=0), UX1_S.max(axis=0)
    lo2, hi2 = UX2.min(axis=1), UX2.max(axis=1)
    bound = np.max([hi1 * hi2, lo1 * lo2, hi1 * lo2, lo1 * hi2], axis=0).sum()
    ratio = bound / mps
    if ratio > 1:
        # Scale by a power of 2 (to prevent rounding errors)
        scale = 2.0 ** np.ceil(np.log2(ratio))
        if UX1_S.size < UX2.size:
            UX1_S = UX1_S / scale
        else:
            UX2 = UX2 / scale
    return UX1_S, UX2


# This is the most complex and difficult part of the script. It was written by Claude
def create_4D_dic(dics, sizes, downsample, labels, data, filename):
    """Build the NMRPipe header of the 4D from the headers of a spectrum pair.

    dics are the headers of the two 3D spectra (axes I,J,K and L,M,K), sizes
    are I, J, L, M before downsampling, and data is the float32 4D array in
    NMRPipe order (M, L, J, I).
    """
    def block(dic, axis):  # header block describing an axis of a 3D array
        return "FDF%d" % dic["FDDIMORDER"][2 - axis]

    # (header, block) each 4D axis comes from, and the block it is written to.
    # NMRPipe's standard order is X=F2, Y=F1, Z=F3, A=F4.
    sources = [(dics[0], block(dics[0], 0)), (dics[0], block(dics[0], 1)),
               (dics[1], block(dics[1], 0)), (dics[1], block(dics[1], 1))]
    targets = ["FDF2", "FDF1", "FDF3", "FDF4"]
    M_4D, L_4D, J_4D, I_4D = data.shape

    dic = dict(dics[0])
    names = {k[4:] for k in dic if k[:4] in targets} - {"SIZE"}
    for (src, src_fn), fn, SIZE, DS, label in zip(sources, targets, sizes,
                                                  downsample, labels):
        # Copy every per-dimension parameter (a few exist for F2 only)
        for name in names:
            if fn + name in dic:
                dic[fn + name] = src.get(src_fn + name, 0.0)
        dic[fn + "LABEL"] = label

        # The whole dimension is used, then every DS-th point is kept
        ORIG, SW = src[src_fn + "ORIG"], src[src_fn + "SW"]
        XN = SIZE
        NEW_XN = XN - (SIZE - 1) % DS
        NEW_SIZE = np.ceil(SIZE / DS)
        dic[fn + "X1"] = 1.0
        dic[fn + "XN"] = NEW_XN
        dic[fn + "ORIG"] = ORIG + SW / SIZE * (XN - NEW_XN)
        dic[fn + "SW"] = SW / SIZE * NEW_SIZE * DS

    # Sizes and layout: real data, X is the fastest varying dimension
    dic["FDDIMCOUNT"] = 4.0
    dic["FDDIMORDER"] = [2.0, 1.0, 3.0, 4.0]
    for n, fn in enumerate(dic["FDDIMORDER"]):
        dic["FDDIMORDER%d" % (n + 1)] = fn
    dic["FDSIZE"] = float(I_4D)
    dic["FDSPECNUM"] = float(J_4D)
    dic["FDF3SIZE"] = float(L_4D)
    dic["FDF4SIZE"] = float(M_4D)
    dic["FDREALSIZE"] = dic["FDF2TDSIZE"]
    dic["FDQUADFLAG"] = 1.0
    dic["FDTRANSPOSED"] = 0.0

    # One file holding the whole 4D is a data stream; otherwise 2D planes
    single_file = filename.count("%") == 0
    dic["FDPIPEFLAG"] = 1.0 if single_file else 0.0
    dic["FDFILECOUNT"] = 1.0 if single_file else float(L_4D * M_4D)
    dic["FDSLICECOUNT"] = 0.0
    dic["FDSLICECOUNT1"] = 0.0

    # Intensity range of the data actually written
    dic["FDMAX"] = dic["FDDISPMAX"] = float(data.max())
    dic["FDMIN"] = dic["FDDISPMIN"] = float(data.min())
    dic["FDSCALEFLAG"] = 1.0

    # These described the acquisition of the shared dimension, which is gone
    dic["FDDMXVAL"] = 0.0
    dic["FDDMXFLAG"] = 0.0
    for n in range(1, 7):
        dic["FDUSER%d" % n] = 0.0
    return dic


spectra = [specarr, specarr]
I, J, K = spectra[0].shape
L, M, _ = spectra[1].shape

stacked = np.vstack([spectra[0].reshape(I * J, K, order="F"),
                     spectra[1].reshape(L * M, K, order="F")])

# MATLAB reads float32 data into doubles, so work in double precision to match
dtype = np.float64 if mode == "real" else np.complex128
stacked = stacked.astype(dtype)
if use_derivative:
    stacked = np.diff(stacked, n=1, axis=1)
if mode == "split":
    stacked = np.hstack([stacked.real, stacked.imag])

print("Performing SVD...")
U, S, _ = scipy.linalg.svd(stacked, full_matrices=False)
print("SVD completed.")

# Row indices to keep in the downsampled 4D (column-major, like MATLAB)
IJ_write = (np.arange(0, I, downsample[0])[:, None]
            + np.arange(0, J, downsample[1])[None, :] * I).ravel(order="F")
LM_write = (np.arange(0, L, downsample[2])[:, None]
            + np.arange(0, M, downsample[3])[None, :] * L).ravel(order="F")

UX1_S = U[IJ_write, :] * S ** (2 * lambda_factor)
UX2 = U[I * J + LM_write, :].conj().T
del stacked, U  # Clear some memory
if mode == "complex":
    # Only the real part of UX1_S @ UX2 is kept, which is this product of
    # real matrices with twice as many components
    UX1_S = np.hstack([UX1_S.real, -UX1_S.imag])
    UX2 = np.vstack([UX2.real, UX2.imag])

# Scale data to keep peak heights reasonable and prevent float overflow
n_pairs = 1
mps = 2 ** (40 / n_pairs)  # Max value allowed per spectrum
UX1_S, UX2 = rescale_pair(UX1_S, UX2, mps)

I_4D, J_4D = len(range(0, I, downsample[0])), len(range(0, J, downsample[1]))
L_4D, M_4D = len(range(0, L, downsample[2])), len(range(0, M, downsample[3]))

# Row lm of UX2.T @ UX1_S.T is one flattened (I_4D x J_4D) plane. Calculate
# it a block of rows at a time, straight into the float32 array that gets
# written, so the full product never exists in float64.
# NMRPipe stores float32 with the first dimension varying fastest, which is
# C order for an array indexed (M, L, J, I)
data_4D = np.empty((L_4D * M_4D, I_4D * J_4D), dtype=np.float32)
for start in tqdm(range(0, L_4D * M_4D, 500), desc="Calculating blocks..."):
    block = UX2.T[start:start + 500] @ UX1_S.T
    # Anything negative is an artifact
    np.maximum(block, 0, out=data_4D[start:start + 500], casting="same_kind")
data_4D = data_4D.reshape(M_4D, L_4D, J_4D, I_4D)
spec_4D = data_4D.transpose(3, 2, 1, 0)  # View indexed (I, J, L, M)

filename_4D = f"no_SR_no_deriv/4D_HCcH_python_{mode}.pipe"
labels_4D = ["H", "C", "Hs", "Cs"]

if filename_4D.count("%") == 1:
    raise ValueError("One '%' is not supported: MATLAB writes 3D cubes for "
                     "it, but nmrglue would write 2D planes")

dic_4D = create_4D_dic([specheader, specheader], [I, J, L, M], downsample,
                       labels_4D, data_4D, filename_4D)
if filename_4D.count("%") == 0:
    # Same file as glue.pipe.write makes, but without its two extra in-memory
    # copies of the data
    with open(filename_4D, "wb") as f:
        f.write(glue.pipe.dic2fdata(dic_4D).tobytes())
        data_4D.tofile(f)
else:
    glue.pipe.write(filename_4D, dic_4D, data_4D, overwrite=True)
print(f"Wrote {filename_4D}")
