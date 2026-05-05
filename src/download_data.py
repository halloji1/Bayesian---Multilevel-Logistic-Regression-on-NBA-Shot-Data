"""Download the NBA Shot Logs dataset from Kaggle to data/raw/."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import kagglehub

import config

logger = logging.getLogger(__name__)

DATASET_SLUG = "dansbecker/nba-shot-logs"


def download(force: bool = False) -> Path:
    """Download shot_logs.csv from Kaggle and place it in DATA_RAW_DIR.

    Uses kagglehub.dataset_download(), which caches the dataset locally and
    returns the path to the cached directory. The CSV is then copied into
    config.DATA_RAW_DIR / config.RAW_CSV.

    Args:
        force: Re-download and overwrite even if the destination file already
               exists. Default is False.

    Returns:
        Absolute Path to the CSV file at config.DATA_RAW_DIR / config.RAW_CSV.

    Side effects:
        Creates config.DATA_RAW_DIR if it does not exist.
        Writes (or overwrites) config.DATA_RAW_DIR / config.RAW_CSV.
    """
    dest = config.DATA_RAW_DIR / config.RAW_CSV

    if dest.exists() and not force:
        logger.info("File already exists, skipping: %s", dest)
        return dest

    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading dataset '%s' via kagglehub ...", DATASET_SLUG)
    cache_path = Path(kagglehub.dataset_download(DATASET_SLUG))
    logger.info("Path to dataset files: %s", cache_path)

    candidates = list(cache_path.rglob("shot_logs.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"shot_logs.csv not found anywhere under {cache_path}"
        )
    src = candidates[0]

    shutil.copy2(src, dest)

    size_mb = dest.stat().st_size / 1_048_576
    logger.info("Download complete: %s (%.2f MB)", dest, size_mb)
    return dest


if __name__ == "__main__":
    config.setup_logging()
    parser = argparse.ArgumentParser(
        description="Download NBA shot logs from Kaggle to data/raw/."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if data/raw/shot_logs.csv already exists.",
    )
    args = parser.parse_args()
    path = download(force=args.force)
    logger.info("Ready: %s", path)
