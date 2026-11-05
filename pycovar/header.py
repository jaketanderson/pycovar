"""NMRPipe header of the 4D output."""
import numpy as np


# This is the most complex and difficult part of the package. It was written
# by Claude (as create_4D_dic in covar.py) and is generalized here to take the
# header block of each axis explicitly. The downsampled ORIG/SW arithmetic
# follows the original MATLAB script (BH/AK/DF).
def create_4D_dic(dics, blocks, sizes, downsample, labels, data):
    """Build the NMRPipe header of the 4D from the headers of a spectrum pair.

    dics are the headers of the two 3D spectra (axes I,J,K and L,M,K), blocks
    the header block ("FDF3", ...) describing each of I, J, L, M, sizes are
    I, J, L, M before downsampling, and data is the float32 4D array in
    NMRPipe order (M, L, J, I). The header describes a single-file data
    stream; see apply_layout for planes.
    """
    # (header, block) each 4D axis comes from, and the block it is written to.
    # NMRPipe's standard order is X=F2, Y=F1, Z=F3, A=F4.
    sources = [(dics[0], blocks[0]), (dics[0], blocks[1]),
               (dics[1], blocks[2]), (dics[1], blocks[3])]
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
    dic["FDSLICECOUNT"] = 0.0
    dic["FDSLICECOUNT1"] = 0.0
    apply_layout(dic, single_file=True)

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


def apply_layout(dic, single_file):
    """Set the header fields that differ between one data-stream file and a
    series of 2D planes (in place). Plane count is F3 size times F4 size."""
    dic["FDPIPEFLAG"] = 1.0 if single_file else 0.0
    dic["FDFILECOUNT"] = 1.0 if single_file else dic["FDF3SIZE"] * dic["FDF4SIZE"]
    return dic
