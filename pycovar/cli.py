"""Command line interface. The argparse / logging layout follows Jake
Anderson's CSP_analysis.py (github.com/jaketanderson/CSP_analysis)."""
import argparse
import logging
import os
import pprint
import platform
import sys

import nmrglue
import numpy
import scipy

from . import __version__
from .axes import AxisError, format_axes, parse_permutation, propose_axes
from .core import MODES, covariance_4D, write_4D
from .io import load_pair

logger = logging.getLogger("pycovar")


def build_parser():
    p = argparse.ArgumentParser(
        prog="pycovar",
        description="4D covariance NMR: covary one or two NMRPipe 3D spectra "
                    "over the dimension they share. One spectrum is covaried "
                    "with itself.")
    p.add_argument("spectra", nargs="+", metavar="SPECTRUM",
                   help="One or two NMRPipe 3D spectra (A [B]). Plane series "
                        "with %% in the name are fine.")
    p.add_argument("-o", "--output", required=True,
                   help="Output 4D. A name without %% is one file; a name with "
                        "two %% writes 2D planes.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite the output if it exists.")
    p.add_argument("--mode", choices=MODES, default="auto",
                   help="How the shared dimension enters: auto follows the "
                        "header (real or complex); split is experimental and "
                        "needs a complex dimension. (default: %(default)s)")
    p.add_argument("--lambda-factor", type=float, default=1.0,
                   help="Matrix power: 1.0 is plain covariance, 0.5 the matrix "
                        "square root. (default: %(default)s)")
    p.add_argument("--derivative", action="store_true",
                   help="First difference along the shared dimension before the SVD.")
    p.add_argument("--downsample", type=int, nargs="+", default=[1], metavar="N",
                   help="Keep every N-th point of the output axes. Give one "
                        "value for all four axes or four (I J L M). The "
                        "covariance is computed at full resolution first.")
    p.add_argument("--labels", nargs=4, metavar=("I", "J", "L", "M"),
                   help="Labels for the four output axes (default: from the "
                        "input headers, with an 's' where they would repeat).")
    p.add_argument("--axes-a", metavar="P0,P1,P2",
                   help="Array-axis permutation for spectrum A (shared axis "
                        "last), e.g. 2,1,0. Skips the automatic check.")
    p.add_argument("--axes-b", metavar="P0,P1,P2", help="Same for spectrum B.")
    p.add_argument("--log", default=None,
                   help="Log file (default: next to the output, <output>.log).")
    p.add_argument("--no-log", action="store_true", help="Do not write a log file.")
    p.add_argument("--verbose", action="store_true", help="Also log debug messages.")
    p.add_argument("--version", action="version", version=f"pycovar {__version__}")
    return p


def setup_logging(logfile, verbose):
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(stream)
    if logfile:
        fh = logging.FileHandler(logfile)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)


def prompt_for_axes(descs, perms, issues):
    """Show the axes and ask which one is shared. Used as load_pair's confirm_axes."""
    print("\nThe axes of the input look wrong:")
    for issue in issues:
        print(f"  - {issue.message}")
    perms = list(perms)
    for n, name in enumerate("AB"):
        if n == 1 and descs[1] is descs[0]:
            break
        print("\n" + format_axes(name, descs[n], perms[n]))
        while True:
            answer = input(f"Spectrum {name}: press Enter to accept, or give a "
                           "permutation (e.g. 2,1,0) or the label of the shared "
                           "axis: ").strip()
            if not answer:
                break
            labels = [ax.label for ax in descs[n]]
            try:
                if answer in labels:
                    if labels.count(answer) > 1:
                        raise AxisError(f"{answer} is ambiguous; give a permutation")
                    perms[n] = propose_axes(descs[n], shared=labels.index(answer))
                else:
                    perms[n] = parse_permutation(answer)
                print(f"  Using {perms[n]}")
                break
            except AxisError as e:
                print(f"  {e}")
    if descs[1] is descs[0]:
        perms[1] = perms[0]
    return perms


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if len(args.spectra) > 2:
        parser.error("Give one or two spectra")
    if len(args.downsample) not in (1, 4):
        parser.error("--downsample takes one or four values")
    if args.output.count("%") == 1:
        parser.error("One '%' in the output is not supported; use none or two")

    logfile = None
    if not args.no_log:
        base = args.output.split("%")[0] if "%" in args.output else args.output
        logfile = args.log or (os.path.splitext(base)[0].rstrip("_-.") or "covar") + ".log"
        os.makedirs(os.path.dirname(os.path.abspath(logfile)), exist_ok=True)
    setup_logging(logfile, args.verbose)

    logger.info("You ran: %s", " ".join([sys.executable] + sys.argv))
    if logfile:
        logger.info("This log file is located at %s", os.path.abspath(logfile))
    logger.info("pycovar %s; python %s; numpy %s, scipy %s, nmrglue %s", __version__,
                platform.python_version(), numpy.__version__, scipy.__version__,
                nmrglue.__version__)
    logger.info("Using the following args:\n%s\n", pprint.pformat(vars(args)))

    for path in args.spectra:
        if "%" not in path and not os.path.exists(path):
            parser.error(f"{path} does not exist")
    if (args.output.count("%") == 0 and os.path.exists(args.output)
            and not args.overwrite):
        parser.error(f"{args.output} exists; use --overwrite")

    axes_a = parse_permutation(args.axes_a) if args.axes_a else None
    axes_b = parse_permutation(args.axes_b) if args.axes_b else None
    is_interactive: bool = sys.stdin.isatty()
    try:
        spec_a, spec_b = load_pair(
            args.spectra[0], args.spectra[1] if len(args.spectra) == 2 else None,
            axes_a=axes_a, axes_b=axes_b,
            confirm_axes=is_interactive)
        result = covariance_4D(spec_a, spec_b, lambda_factor=args.lambda_factor,
                               derivative=args.derivative, mode=args.mode,
                               downsample=args.downsample, labels=args.labels,
                               progress=True)
        write_4D(result, args.output, overwrite=args.overwrite)
    except (AxisError, ValueError, FileExistsError) as e:
        logger.error("%s", e)
        return 1
    return 0
