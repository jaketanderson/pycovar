"""pycovar: 4D covariance (generalized indirect covariance) of two NMRPipe 3D
spectra that share one dimension.

    import pycovar
    a, b = pycovar.load_pair("hcch.pipe")          # one file: self-covariance
    result = pycovar.covariance_4D(a, b, downsample=2)
    pycovar.write_4D(result, "out/4D.pipe")

Importing this package does no I/O and configures no logging.
"""
from .axes import (AxisError, AxisInfo, Issue, ShapeMismatchError, check_pair,
                   describe_axes, propose_axes)
from .core import Covariance4D, covariance_4D, default_labels, write_4D
from .io import Spectrum, load_pair, read_spectrum

__version__ = "0.1.0"

__all__ = ["AxisError", "AxisInfo", "Covariance4D", "Issue", "ShapeMismatchError",
           "Spectrum", "check_pair", "covariance_4D", "default_labels",
           "describe_axes", "load_pair", "propose_axes", "read_spectrum",
           "write_4D"]
