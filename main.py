"""Entry point for the NBA shot Bayesian analysis pipeline."""

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Bayesian Multilevel Logistic Regression — NBA Shot Data"
    )
    p.add_argument("--all",         action="store_true", help="run the full pipeline end-to-end")
    p.add_argument("--download",    action="store_true", help="download raw data from Kaggle")
    p.add_argument("--clean",       action="store_true", help="clean and validate raw data")
    p.add_argument("--eda",         action="store_true", help="run exploratory data analysis")
    p.add_argument("--prepare",     action="store_true", help="prepare features and train/test split")
    p.add_argument("--fit",         action="store_true", help="fit the full multilevel model")
    p.add_argument("--prior",       action="store_true", help="run prior predictive checks")
    p.add_argument("--compare",     action="store_true", help="compare candidate models (LOO/WAIC)")
    p.add_argument("--posterior",   action="store_true", help="generate posterior predictive plots")
    p.add_argument("--sensitivity", action="store_true", help="run prior sensitivity analysis")
    p.add_argument("--validate",    action="store_true", help="run posterior predictive validation")
    p.add_argument("--force",       action="store_true", help="overwrite existing outputs")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    steps = [
        "all", "download", "clean", "eda", "prepare",
        "fit", "prior", "compare", "posterior", "sensitivity", "validate",
    ]

    if not any(getattr(args, s) for s in steps):
        parser.print_help()
        sys.exit(0)

    for step in steps:
        if getattr(args, step):
            print(f"--{step}: not implemented")


if __name__ == "__main__":
    main()
