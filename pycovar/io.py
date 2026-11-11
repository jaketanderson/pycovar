"""Reading NMRPipe 3D spectra into the (I, J, K) layout used for covariance."""
import logging
from dataclasses import dataclass

import nmrglue as ng
import numpy as np

from .axes import (AxisError, check_pair, describe_axes, format_axes,
                   propose_axes, validate_permutation)

logger = logging.getLogger("pycovar")


@dataclass
class Spectrum:
    """A 3D spectrum arranged so the shared dimension is the last axis."""
    path: str
    data: np.ndarray    # (I, J, K); complex if the shared dimension is complex
    dic: dict           # header as read (not modified)
    axes: list          # AxisInfo for I, J, K, in that order
    perm: tuple         # array-axis permutation that produced `data`

    @property
    def shape(self):
        return self.data.shape

    @property
    def is_complex(self):
        return np.iscomplexobj(self.data)


def _read_raw(path):
    dic, arr = ng.pipe.read(path)
    return dic, arr, describe_axes(dic, arr)


def _arrange(path, dic, arr, axes, perm):
    """Permute so the shared axis is last, and make a complex shared axis complex."""
    perm = validate_permutation(perm)
    data = arr.transpose(perm)
    shared = axes[perm[2]]
    if shared.complex and not np.iscomplexobj(data):
        # Interleaved real and imaginary points along the shared axis (nmrglue
        # only combines them itself when that axis is stored fastest)
        data = data[..., 0::2] + 1j * data[..., 1::2]
    return Spectrum(path, data, dic, [axes[p] for p in perm], perm)


def read_spectrum(path, axes=None):
    """Read one spectrum. `axes` is an array-axis permutation; if omitted the
    shared dimension is taken to be the direct one (FDF2) and put last.

    Unlike load_pair, this does no cross-spectrum checking.
    """
    dic, arr, desc = _read_raw(path)
    perm = tuple(axes) if axes is not None else propose_axes(desc)
    for issue in check_pair(desc, perm, desc, perm):
        if issue.severity == "fatal":
            raise AxisError(issue.message)
    return _arrange(path, dic, arr, desc, perm)


def load_pair(path_a, path_b=None, *, axes_a=None, axes_b=None, confirm_axes=None):
    """Read two spectra (or one, for self-covariance) ready for covariance.

    axes_a / axes_b: explicit permutations; they are not second-guessed.
    confirm_axes: callback(descs, perms, issues) -> perms, called when
        something about the proposed arrangement looks wrong and at least one
        permutation was not given explicitly. descs is [axes_a, axes_b]. It
        returns the permutations to use, [perm_a, perm_b]. If it is None, a
        suspicious arrangement raises AxisError, with the permutations to pass
        as axes_a / axes_b in the message. Fatal problems always raise.
    Returns (Spectrum, Spectrum); the same object twice for self-covariance.
    """
    self_cov = path_b is None
    dic_a, arr_a, desc_a = _read_raw(path_a)
    if self_cov:
        dic_b, arr_b, desc_b = dic_a, arr_a, desc_a
        axes_b = axes_a
    else:
        dic_b, arr_b, desc_b = _read_raw(path_b)
    explicit = [axes_a is not None, axes_b is not None]
    perm_a = validate_permutation(axes_a) if explicit[0] else propose_axes(desc_a)
    perm_b = validate_permutation(axes_b) if explicit[1] else propose_axes(desc_b)

    issues = check_pair(desc_a, perm_a, desc_b, perm_b)
    warns = [i for i in issues if i.severity == "warn"]
    if warns and not all(explicit):
        for i in warns:
            logger.warning(i.message)
        if confirm_axes is None:
            # Prompt the user to consider changing their axes
            raise AxisError(
                "The axes look wrong:\n  " + "\n  ".join(i.message for i in warns)
                + f"\nGive the permutations explicitly (axes_a / --axes-a, and "
                  f"axes_b / --axes-b); proposed: A {perm_a}, B {perm_b}.\n"
                + format_axes("A", desc_a, perm_a)
                + ("" if self_cov else "\n" + format_axes("B", desc_b, perm_b)))
        perm_a, perm_b = (validate_permutation(p) for p in
                          confirm_axes([desc_a, desc_b], [perm_a, perm_b], warns))
        issues = check_pair(desc_a, perm_a, desc_b, perm_b)
    fatal = [i.message for i in issues if i.severity == "fatal"]
    if fatal:
        raise AxisError("; ".join(fatal))

    spec_a = _arrange(path_a, dic_a, arr_a, desc_a, perm_a)
    spec_b = spec_a if self_cov else _arrange(path_b, dic_b, arr_b, desc_b, perm_b)
    logger.info("Spectrum A %s: axes %s -> shape %s", path_a, perm_a, spec_a.shape)
    if not self_cov:
        logger.info("Spectrum B %s: axes %s -> shape %s", path_b, perm_b, spec_b.shape)
    return spec_a, spec_b
