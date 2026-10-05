import nmrglue as glue
import numpy as np
import scipy

specheader, specarr = glue.fileio.pipe.read("HCcH.pipe")
specheader, specarr = glue.fileio.pipe.transpose_3D(specheader, specarr, axes=(2,1,0))  # Transpose so 1H is third dim
print(f"{'Spectrum dimensions:':<22}{specarr.shape}")
udic = glue.pipe.guess_udic(specheader, specarr)
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

# MATLAB reads float32 data into doubles, so work in float64 to match
stacked = np.diff(stacked.astype(np.float64), n=1, axis=1)

U, S, _ = scipy.linalg.svd(stacked, full_matrices=False)

lambda_factor = 0.5
downsample = [1, 1, 1, 1]

# Row indices to keep in the downsampled 4D (column-major, like MATLAB)
IJ_write = (np.arange(0, I, downsample[0])[:, None]
            + np.arange(0, J, downsample[1])[None, :] * I).ravel(order="F")
LM_write = (np.arange(0, L, downsample[2])[:, None]
            + np.arange(0, M, downsample[3])[None, :] * L).ravel(order="F")

UX1_S = U[IJ_write, :] * S ** (2 * lambda_factor)
UX2 = U[I * J + LM_write, :].T
del stacked, U  # Clear some memory

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
for start in range(0, L_4D * M_4D, 500):
    block = UX2.T[start:start + 500] @ UX1_S.T
    # Anything negative is an artifact
    np.maximum(block, 0, out=data_4D[start:start + 500], casting="same_kind")
data_4D = data_4D.reshape(M_4D, L_4D, J_4D, I_4D)
spec_4D = data_4D.transpose(3, 2, 1, 0)  # View indexed (I, J, L, M)

filename_4D = "4D_HCCH_python.pipe"
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
