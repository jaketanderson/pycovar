"""Tests on tiny synthetic NMRPipe files; no real data needed.

Run with: python -m pytest tests   (from the repo root)
"""
import logging
import subprocess
import sys
from pathlib import Path

import nmrglue as ng
import numpy as np
import pytest

import pycovar
from pycovar.axes import AxisError, ShapeMismatchError

ROOT = Path(__file__).resolve().parents[1]


def _axis(n, cplx, label, obs, sw=1000.0):
    return {"size": n, "complex": cplx, "sw": sw, "obs": obs, "car": 100.0,
            "label": label, "time": False, "freq": True, "encoding": "states"}


def make_pipe(path, data, labels=("1Hy", "13C", "1H"), shared_complex=False,
              obs=(600.0, 150.0, 600.0), sws=(1000.0, 1000.0, 1000.0)):
    """Write a 3D stream in standard order. data is indexed (F3, F1, F2) with
    F2 (fastest, last) the direct dimension; complex64 if shared_complex."""
    n3, n1, n2 = data.shape
    udic = {"ndim": 3,
            0: _axis(n3, False, labels[0], obs[0], sws[0]),
            1: _axis(n1, False, labels[1], obs[1], sws[1]),
            2: _axis(n2, shared_complex, labels[2], obs[2], sws[2])}
    dic = ng.pipe.create_dic(udic)
    dic["FDPIPEFLAG"] = 1.0
    dic["FDFILECOUNT"] = 1.0
    ng.pipe.write(str(path), dic, data, overwrite=True)
    return str(path)


def rand(shape, seed, cplx=False):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=shape)
    if cplx:
        return (a + 1j * rng.normal(size=shape)).astype("complex64")
    return a.astype("float32")


def test_import_has_no_side_effects():
    code = ("import logging, pycovar; "
            "assert not logging.getLogger('pycovar').handlers; "
            "assert not logging.getLogger().handlers")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


def test_propose_axes_transposed_header():
    # Header like HCcH_*.pipe: direct 1H (FDF2) is the slowest array axis
    dic = ng.pipe.create_dic({"ndim": 3, 0: _axis(6, False, "1Hy", 600.),
                              1: _axis(4, False, "13C", 150.),
                              2: _axis(8, False, "1H", 600.)})
    dic["FDDIMORDER"] = [3.0, 1.0, 2.0, 4.0]
    axes = pycovar.describe_axes(dic, np.zeros((8, 4, 6), "float32"))
    assert [a.label for a in axes] == ["1H", "13C", "1Hy"]  # F2, F1, F3
    assert pycovar.propose_axes(axes) == (2, 1, 0)


def test_propose_axes_standard_order(tmp_path):
    p = make_pipe(tmp_path / "a.pipe", rand((6, 4, 8), 0))
    spec = pycovar.read_spectrum(p)
    assert spec.shape == (4, 6, 8)
    assert [a.label for a in spec.axes] == ["13C", "1Hy", "1H"]


def test_self_covariance_matches_numpy(tmp_path):
    a = rand((6, 4, 8), 1)
    p = make_pipe(tmp_path / "a.pipe", a)
    sa, sb = pycovar.load_pair(p)
    assert sa is sb
    res = pycovar.covariance_4D(sa)
    A = sa.data.astype(float)
    expect = np.maximum(np.einsum("ijk,lmk->ijlm", A, A), 0)
    assert res.scale == 1.0
    assert np.allclose(res.spec, expect, rtol=1e-4, atol=1e-4 * expect.max())
    assert res.labels == ["13C", "1Hy", "13Cs", "1Hys"]


def test_cross_covariance_matches_numpy(tmp_path):
    pa = make_pipe(tmp_path / "a.pipe", rand((6, 4, 8), 2))
    pb = make_pipe(tmp_path / "b.pipe", rand((5, 3, 8), 3),
                   labels=("1Hy", "15N", "1H"), obs=(600.0, 60.0, 600.0))
    sa, sb = pycovar.load_pair(pa, pb)
    res = pycovar.covariance_4D(sa, sb)
    expect = np.maximum(np.einsum("ijk,lmk->ijlm", sa.data.astype(float),
                                  sb.data.astype(float)), 0)
    assert res.spec.shape == (4, 6, 3, 5)
    assert np.allclose(res.spec, expect, rtol=1e-4, atol=1e-4 * expect.max())
    assert res.labels == ["13C", "1Hy", "15N", "1Hys"]
    # Passing the same spectrum twice is the same as self-covariance
    assert np.array_equal(pycovar.covariance_4D(sa, sa).data, pycovar.covariance_4D(sa).data)


def test_shared_size_mismatch_is_an_error(tmp_path):
    pa = make_pipe(tmp_path / "a.pipe", rand((6, 4, 8), 4))
    pb = make_pipe(tmp_path / "b.pipe", rand((6, 4, 16), 5))
    with pytest.raises(AxisError, match="differ in size"):
        pycovar.load_pair(pa, pb)


def test_wrong_looking_axes_raise_without_callback_and_use_callback(tmp_path):
    pa = make_pipe(tmp_path / "a.pipe", rand((6, 4, 8), 6))
    pb = make_pipe(tmp_path / "b.pipe", rand((6, 4, 8), 7), labels=("1Hy", "13C", "15N"),
                   obs=(600.0, 150.0, 60.0))
    with pytest.raises(AxisError, match="axes_a"):
        pycovar.load_pair(pa, pb)
    seen = {}

    def confirm(descs, perms, issues):
        seen["issues"] = issues
        return perms  # user accepts what was proposed

    sa, sb = pycovar.load_pair(pa, pb, confirm_axes=confirm)
    assert any("nuclei" in i.message for i in seen["issues"])
    # Explicit axes are not second-guessed
    pycovar.load_pair(pa, pb, axes_a=(1, 0, 2), axes_b=(1, 0, 2))


def test_complex_shared_axis_stored_fastest(tmp_path):
    a = rand((6, 4, 8), 8, cplx=True)
    p = make_pipe(tmp_path / "a.pipe", a, shared_complex=True)
    spec = pycovar.read_spectrum(p)
    assert spec.is_complex and spec.shape == (4, 6, 8)
    res = pycovar.covariance_4D(spec)
    assert res.mode == "complex"
    A = spec.data.astype("complex128")
    expect = np.maximum(np.einsum("ijk,lmk->ijlm", A, A.conj()).real, 0)
    assert np.allclose(res.spec, expect, rtol=1e-4, atol=1e-4 * expect.max())
    with pytest.raises(ValueError, match="needs a spectrum"):
        pycovar.covariance_4D(spec, mode="real")


def test_complex_shared_axis_interleaved(tmp_path):
    # Real-typed data with the complex axis not stored fastest (like HCcH_complex)
    a = rand((6, 4, 8), 9, cplx=True)
    p = make_pipe(tmp_path / "a.pipe", a, shared_complex=True)
    dic, arr = ng.pipe.read(p)
    # Make an equivalent file in which the real-typed array holds interleaved points
    inter = np.empty((6, 4, 16), "float32")
    inter[..., 0::2], inter[..., 1::2] = arr.real, arr.imag
    axes = pycovar.describe_axes(dic, inter)
    assert axes[2].size == 8 and axes[2].complex
    from pycovar.io import _arrange
    spec = _arrange(p, dic, inter, axes, (1, 0, 2))
    assert np.array_equal(spec.data, arr.transpose(1, 0, 2))


def test_complex_nonshared_axis_is_fatal():
    dic = ng.pipe.create_dic({"ndim": 3, 0: _axis(6, True, "1Hy", 600.),
                              1: _axis(4, False, "13C", 150.),
                              2: _axis(8, False, "1H", 600.)})
    axes = pycovar.describe_axes(dic, np.zeros((12, 4, 8), "float32"))
    issues = pycovar.check_pair(axes, (1, 0, 2), axes, (1, 0, 2))
    assert any(i.severity == "fatal" and "non-shared" in i.message for i in issues)


def test_downsample_is_subsample_of_full(tmp_path):
    p = make_pipe(tmp_path / "a.pipe", rand((9, 7, 20), 10))
    spec = pycovar.read_spectrum(p)
    full = pycovar.covariance_4D(spec)
    ds = pycovar.covariance_4D(spec, downsample=(2, 3, 2, 3))
    assert ds.scale == full.scale
    assert np.allclose(ds.spec, full.spec[::2, ::3, ::2, ::3], rtol=1e-5,
                       atol=1e-6 * full.spec.max())


def test_cli_end_to_end_and_log(tmp_path):
    p = make_pipe(tmp_path / "a.pipe", rand((6, 4, 8), 11))
    out = tmp_path / "sub" / "out.pipe"
    cmd = [sys.executable, "-m", "pycovar", p, "-o", str(out), "--downsample", "2"]
    subprocess.run(cmd, check=True, cwd=ROOT, stdin=subprocess.DEVNULL)
    assert out.exists() and (tmp_path / "sub" / "out.log").exists()
    dic, data = ng.pipe.read(str(out))
    assert data.shape == (3, 2, 3, 2)  # (M, L, J, I): I=4, J=6 downsampled by 2
    # Refuses to overwrite, and --no-log makes no log
    r = subprocess.run(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, capture_output=True)
    assert r.returncode != 0
    out2 = tmp_path / "out2.pipe"
    subprocess.run([sys.executable, "-m", "pycovar", p, "-o", str(out2), "--no-log"],
                   check=True, cwd=ROOT, stdin=subprocess.DEVNULL)
    assert out2.exists() and not (tmp_path / "out2.log").exists()


def test_cli_doubtful_axes_fail_without_tty(tmp_path):
    pa = make_pipe(tmp_path / "a.pipe", rand((6, 4, 8), 12))
    pb = make_pipe(tmp_path / "b.pipe", rand((6, 4, 8), 13), labels=("1Hy", "13C", "15N"),
                   obs=(600.0, 150.0, 60.0))
    r = subprocess.run([sys.executable, "-m", "pycovar", pa, pb, "-o",
                        str(tmp_path / "o.pipe"), "--no-log"], cwd=ROOT,
                       stdin=subprocess.DEVNULL, capture_output=True, text=True)
    assert r.returncode == 1 and "--axes-a" in r.stdout + r.stderr


def test_prompt_parsing(monkeypatch):
    from pycovar.cli import prompt_for_axes
    dic = ng.pipe.create_dic({"ndim": 3, 0: _axis(6, False, "1Hy", 600.),
                              1: _axis(4, False, "13C", 150.),
                              2: _axis(8, False, "1H", 600.)})
    axes = pycovar.describe_axes(dic, np.zeros((6, 4, 8), "float32"))
    answers = iter(["nonsense", "0,1,2", ""])  # bad, then a permutation, then accept B
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    perms = prompt_for_axes([axes, list(axes)], [(1, 0, 2), (1, 0, 2)], [])
    assert perms == [(0, 1, 2), (1, 0, 2)]
    answers = iter(["13C", "1Hy"])  # by shared-axis label
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    perms = prompt_for_axes([axes, list(axes)], [(1, 0, 2), (1, 0, 2)], [])
    assert perms == [(2, 0, 1), (2, 1, 0)]
