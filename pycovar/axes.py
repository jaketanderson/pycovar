"""Axis bookkeeping for NMRPipe 3D spectra: what each array axis is, which
one is the shared (covaried) dimension, and whether a pair of spectra can be
covaried. No I/O and no prompting here.
"""
import re
from dataclasses import dataclass

import numpy as np


class AxisError(ValueError):
    """The axes of a spectrum (or pair) cannot be used as given."""


class ShapeMismatchError(AxisError):
    """The shared dimensions of two spectra do not have the same size."""


@dataclass(frozen=True)
class AxisInfo:
    index: int      # position in the array as read (before any permutation)
    block: str      # header block describing it, e.g. "FDF2"
    label: str
    size: int       # points; complex points count once
    sw: float
    obs: float
    orig: float
    car: float
    complex: bool   # header says complex (quad flag != 1)


@dataclass(frozen=True)
class Issue:
    severity: str   # "fatal" cannot proceed; "warn" ask the user
    message: str


def describe_axes(dic, arr):
    """One AxisInfo per array axis of a 3D NMRPipe array.

    Array axis a is described by header block FDF{FDDIMORDER[2 - a]}. If the
    array is real-typed but the header says an axis is complex, the points
    are stored interleaved (real, imaginary) and the stored length is twice
    the size reported here. If nmrglue already returned a complex array, the
    stored length is the size.
    """
    if arr.ndim != 3:
        raise AxisError(f"Only 3D spectra are supported, got {arr.ndim}D")
    axes = []
    for a in range(3):
        block = "FDF%d" % int(dic["FDDIMORDER"][2 - a])
        is_complex = dic[block + "QUADFLAG"] != 1.0
        size = arr.shape[a]
        if is_complex and not np.iscomplexobj(arr):
            if size % 2:
                raise AxisError(f"Axis {a} ({block}) is flagged complex but has "
                                f"an odd number of points ({size})")
            size //= 2
        axes.append(AxisInfo(a, block, str(dic[block + "LABEL"]), size,
                             float(dic[block + "SW"]), float(dic[block + "OBS"]),
                             float(dic[block + "ORIG"]), float(dic[block + "CAR"]),
                             is_complex))
    return axes


def shared_index(axes):
    """Array index of the acquisition dimension (header block FDF2).

    By NMRPipe convention FDF2 is the direct dimension whatever order the
    data is stored in. (The first entry of FDDIMORDER is not: for a
    transposed file it can be an indirect dimension.)
    """
    for ax in axes:
        if ax.block == "FDF2":
            return ax.index
    raise AxisError("No FDF2 block found")


def propose_axes(axes, shared=None):
    """Permutation putting the shared axis last.

    The other two axes follow in reversed array order (NMRPipe's fastest
    dimension first), so a spectrum read as (1H, 13C, 1Hy) gives (2, 1, 0).
    """
    k = shared_index(axes) if shared is None else shared
    return tuple([a for a in (2, 1, 0) if a != k] + [k])


def validate_permutation(perm):
    perm = tuple(int(p) for p in perm)
    if sorted(perm) != [0, 1, 2]:
        raise AxisError(f"{perm} is not a permutation of the axes (0, 1, 2)")
    return perm


def parse_permutation(text):
    """'2,1,0' (or '2 1 0') -> (2, 1, 0)."""
    try:
        return validate_permutation(re.split(r"[,\s]+", text.strip()))
    except ValueError:
        raise AxisError(f"Cannot read {text!r} as an axis permutation, "
                        "e.g. 2,1,0") from None


def nucleus(label):
    """'1Hy' -> '1H', '13C' -> '13C', '15N' -> '15N' (best effort)."""
    m = re.match(r"\d*[A-Za-z]", label)
    return m.group(0) if m else label


def _ppm_window(ax):
    return ax.orig / ax.obs, (ax.orig + ax.sw) / ax.obs


def check_pair(axes_a, perm_a, axes_b, perm_b):
    """Everything that makes a pairing look wrong, as a list of Issues.

    axes_* are describe_axes() results (array order as read); perm_* say how
    each will be permuted so the shared axis is last.
    """
    issues = []
    for name, axes, perm in (("A", axes_a, perm_a), ("B", axes_b, perm_b)):
        shared = axes[perm[2]]
        for p in perm[:2]:
            if axes[p].complex:
                issues.append(Issue("fatal", f"Spectrum {name}: non-shared axis "
                                    f"{axes[p].label} ({axes[p].block}) is complex; "
                                    "only the shared dimension may be complex"))
        if shared.block != "FDF2":
            issues.append(Issue("warn", f"Spectrum {name}: shared axis "
                                f"{shared.label} is {shared.block}, not FDF2 (the "
                                "direct dimension)"))
        if shared.size < max(axes[p].size for p in perm[:2]):
            issues.append(Issue("warn", f"Spectrum {name}: shared axis "
                                f"{shared.label} ({shared.size} points) is not the "
                                "largest axis"))

    sa, sb = axes_a[perm_a[2]], axes_b[perm_b[2]]
    if sa.size != sb.size:
        issues.append(Issue("fatal", f"Shared dimensions differ in size: A has "
                            f"{sa.size} points ({sa.label}), B has {sb.size} ({sb.label})"))
    if sa.complex != sb.complex:
        issues.append(Issue("fatal", "Shared dimension is complex in one spectrum "
                            "and real in the other"))
    if nucleus(sa.label) != nucleus(sb.label):
        issues.append(Issue("warn", f"Shared dimension nuclei differ: {sa.label} "
                            f"vs {sb.label}"))
    if not np.isclose(sa.obs, sb.obs, rtol=1e-4) or not np.isclose(sa.sw, sb.sw, rtol=1e-4):
        issues.append(Issue("warn", f"Shared dimension OBS/SW differ: "
                            f"A {sa.obs:g} MHz / {sa.sw:g} Hz, "
                            f"B {sb.obs:g} MHz / {sb.sw:g} Hz"))
    else:
        (lo_a, hi_a), (lo_b, hi_b) = _ppm_window(sa), _ppm_window(sb)
        step = (hi_a - lo_a) / max(sa.size, 1)
        if abs(lo_a - lo_b) > step or abs(hi_a - hi_b) > step:
            issues.append(Issue("warn", "Shared dimension covers different ppm "
                                f"windows: A {lo_a:.3f} to {hi_a:.3f}, "
                                f"B {lo_b:.3f} to {hi_b:.3f}"))
    return issues


def format_axes(name, axes, perm):
    """Human-readable table of one spectrum's axes and the proposed order."""
    lines = [f"Spectrum {name}: array axes as read"]
    lines.append(f"  {'idx':<4}{'block':<7}{'label':<9}{'size':>7}{'SW (Hz)':>12}"
                 f"{'OBS (MHz)':>11}  kind")
    for ax in axes:
        lines.append(f"  {ax.index:<4}{ax.block:<7}{ax.label:<9}{ax.size:>7}"
                     f"{ax.sw:>12.2f}{ax.obs:>11.2f}  "
                     f"{'complex' if ax.complex else 'real'}")
    labels = [axes[p].label for p in perm]
    lines.append(f"  Proposed order {perm}: I={labels[0]}, J={labels[1]}, "
                 f"shared K={labels[2]}")
    return "\n".join(lines)
