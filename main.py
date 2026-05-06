"""Entry point for the NBA Shot Bayesian analysis pipeline.

Usage examples:
    python main.py --all                   # full pipeline, both priors
    python main.py --download              # step 0: fetch raw data
    python main.py --clean                 # step 1: clean raw CSV
    python main.py --eda                   # step 2: exploratory analysis
    python main.py --prepare               # step 3: feature prep & splits
    python main.py --fit                   # step 4: fit both priors
    python main.py --fit --prior A         # step 4: fit Prior A only
    python main.py --compare               # step 5: LOO model comparison
    python main.py --posterior             # step 6: posterior analysis, both
    python main.py --posterior --prior B   # step 6: Prior B only
    python main.py --sensitivity           # step 7: sensitivity analysis
    python main.py --validate              # step 8: held-out validation, both
    python main.py --validate --prior A    # step 8: Prior A only
    python main.py --fit --prior A --force # force re-fit even if file exists
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys
import time
from typing import Callable

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _priors(args: argparse.Namespace) -> list[str]:
    """Return the list of priors to process for steps that accept --prior."""
    return [args.prior] if args.prior else ["A", "B"]


@contextlib.contextmanager
def _timed(name: str):
    """Context manager that logs step start and elapsed time on exit."""
    logger.info("─── START %-20s ───", name)
    t0 = time.perf_counter()
    yield
    logger.info("─── DONE  %-20s  (%.1f s) ───", name, time.perf_counter() - t0)


def _run(name: str, fn: Callable[[argparse.Namespace], None], args: argparse.Namespace) -> None:
    """Run one pipeline step with timing; catch missing-prerequisite errors."""
    try:
        with _timed(name):
            fn(args)
    except FileNotFoundError as exc:
        logger.error(
            "Prerequisite missing for step '%s':\n  %s\n"
            "Run the earlier pipeline steps first, or use `python main.py --all`.",
            name, exc,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Step functions  (deferred imports keep startup fast)
# ---------------------------------------------------------------------------

def step_download(args: argparse.Namespace) -> None:
    from src.download_data import download
    download(force=args.force)


def step_clean(args: argparse.Namespace) -> None:
    from src.clean_data import clean
    clean()


def step_eda(args: argparse.Namespace) -> None:
    from src.eda import run_eda
    run_eda()


def step_prepare(args: argparse.Namespace) -> None:
    from src.prepare_features import prepare
    prepare()


def step_fit(args: argparse.Namespace) -> None:
    from src.fit_model import fit
    for prior in _priors(args):
        logger.info("Fitting Prior %s ...", prior)
        fit(prior_name=prior, force=args.force)


def step_compare(args: argparse.Namespace) -> None:
    from src.compare_models import compare
    compare()


def step_posterior(args: argparse.Namespace) -> None:
    from src.posterior_analysis import analyze
    for prior in _priors(args):
        logger.info("Posterior analysis — Prior %s ...", prior)
        analyze(prior_name=prior)


def step_sensitivity(args: argparse.Namespace) -> None:
    from src.sensitivity_analysis import run_sensitivity
    run_sensitivity()


def step_validate(args: argparse.Namespace) -> None:
    from src.validate_model import validate, validate_all
    if args.prior:
        validate(args.prior)
    else:
        validate_all()


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

_ORDERED_STEPS: list[tuple[str, Callable]] = [
    ("download",    step_download),
    ("clean",       step_clean),
    ("eda",         step_eda),
    ("prepare",     step_prepare),
    ("fit",         step_fit),
    ("compare",     step_compare),
    ("posterior",   step_posterior),
    ("sensitivity", step_sensitivity),
    ("validate",    step_validate),
]

_STEP_FLAGS = [name for name, _ in _ORDERED_STEPS]


def run_full_pipeline(args: argparse.Namespace) -> None:
    """Execute every pipeline step in order."""
    t_start = time.perf_counter()
    logger.info("════ Full pipeline start ════")
    for name, fn in _ORDERED_STEPS:
        _run(name, fn, args)
    logger.info("════ Full pipeline done (%.1f s) ════", time.perf_counter() - t_start)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Bayesian Multilevel Logistic Regression — NBA Shot Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Step flags
    p.add_argument("--all",         action="store_true", help="Run the full pipeline end-to-end")
    p.add_argument("--download",    action="store_true", help="Step 0: download raw data from Kaggle")
    p.add_argument("--clean",       action="store_true", help="Step 1: clean raw CSV")
    p.add_argument("--eda",         action="store_true", help="Step 2: exploratory data analysis")
    p.add_argument("--prepare",     action="store_true", help="Step 3: feature prep and train/test split")
    p.add_argument("--fit",         action="store_true", help="Step 4: fit multilevel model")
    p.add_argument("--compare",     action="store_true", help="Step 5: LOO model comparison")
    p.add_argument("--posterior",   action="store_true", help="Step 6: posterior summaries and plots")
    p.add_argument("--sensitivity", action="store_true", help="Step 7: prior sensitivity analysis")
    p.add_argument("--validate",    action="store_true", help="Step 8: held-out validation metrics")

    # Modifiers
    p.add_argument(
        "--prior", choices=["A", "B"], default=None, metavar="A|B",
        help="Which prior to use for --fit / --posterior / --validate. "
             "Omit to run both A and B.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-run steps even if their outputs already exist (download, fit).",
    )

    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    config.setup_logging()

    # Print help if no actionable flag was given
    if not args.all and not any(getattr(args, flag) for flag in _STEP_FLAGS):
        parser.print_help()
        sys.exit(0)

    if args.all:
        run_full_pipeline(args)
        return

    for flag, fn in _ORDERED_STEPS:
        if getattr(args, flag):
            _run(flag, fn, args)


if __name__ == "__main__":
    main()
