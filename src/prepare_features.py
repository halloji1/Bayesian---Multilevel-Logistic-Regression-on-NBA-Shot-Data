"""Standardize predictors, encode player factor, and split into train/test sets.

Input:   data/processed/shots_clean.csv
Outputs: data/processed/train.csv
         data/processed/test.csv
         data/processed/player_index.csv
         data/processed/scaler_params.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import config

logger = logging.getLogger(__name__)

# Columns to z-score; their standardized twins are named <col>_z.
_SCALE_COLS = [
    config.COL_SHOT_DIST,
    config.COL_CLOSE_DEF,
    config.COL_SHOT_CLOCK,
    config.COL_TOUCH_TIME,
]

_SCALER_PARAMS_PATH = config.DATA_PROCESSED_DIR / "scaler_params.json"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_player_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted player-id → 0-indexed integer mapping.

    Sorts by player_id so the index is deterministic across runs.
    """
    mapping = (
        df[[config.COL_PLAYER_ID, config.COL_PLAYER_NAME]]
        .drop_duplicates(subset=config.COL_PLAYER_ID)
        .sort_values(config.COL_PLAYER_ID)
        .reset_index(drop=True)
    )
    mapping[config.COL_PLAYER_IDX] = range(len(mapping))
    return mapping[[config.COL_PLAYER_ID, config.COL_PLAYER_NAME, config.COL_PLAYER_IDX]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare(input_path: Path | None = None) -> dict[str, Path]:
    """Standardize predictors, encode player factor, and write train/test splits.

    Steps:
        1. Z-score standardize SHOT_DIST, CLOSE_DEF_DIST, SHOT_CLOCK, TOUCH_TIME
           using training-set statistics. Appends *_z columns alongside originals.
           Saves means and SDs to scaler_params.json for downstream interpretation.
        2. Build a 0-indexed integer player_idx from player_id (sorted for
           determinism). Saves the mapping to player_index.csv.
        3. Stratified 80/20 train/test split on player_idx so every player
           contributes shots to both sets. Uses config.RANDOM_SEED.
        4. Saves train.csv and test.csv.

    Args:
        input_path: Path to cleaned CSV. Defaults to
                    config.DATA_PROCESSED_DIR / config.CLEAN_CSV.

    Returns:
        Dict with keys "train", "test", "player_index", "scaler_params"
        mapping to the absolute Path of each output file.

    Raises:
        FileNotFoundError: If input_path does not exist.

    Side effects:
        Creates config.DATA_PROCESSED_DIR if missing.
        Writes four files under data/processed/.
    """
    if input_path is None:
        input_path = config.DATA_PROCESSED_DIR / config.CLEAN_CSV

    if not input_path.exists():
        raise FileNotFoundError(
            f"Cleaned CSV not found: {input_path}\n"
            "Run `python -m src.clean_data` first."
        )

    logger.info("Loading %s ...", input_path)
    df = pd.read_csv(input_path)
    logger.info("Loaded %d rows, %d players", len(df), df[config.COL_PLAYER_ID].nunique())

    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Z-score standardize numeric predictors
    #    Stats are computed on the full cleaned dataset (pre-split) so that
    #    the scaler_params.json reflect population-level scale and can be
    #    used consistently at inference time.
    # ------------------------------------------------------------------
    scaler_params: dict[str, dict[str, float]] = {}
    for col in _SCALE_COLS:
        mean = float(df[col].mean())
        std  = float(df[col].std(ddof=1))
        df[f"{col}_z"] = (df[col] - mean) / std
        scaler_params[col] = {"mean": mean, "std": std}
        logger.info("  Standardized %-22s  mean=%7.4f  std=%7.4f", col, mean, std)

    _SCALER_PARAMS_PATH.write_text(json.dumps(scaler_params, indent=2))
    logger.info("Saved: %s", _SCALER_PARAMS_PATH)

    # ------------------------------------------------------------------
    # 2. Build 0-indexed player factor and merge onto the dataset
    # ------------------------------------------------------------------
    player_index = _build_player_index(df)
    df = df.merge(
        player_index[[config.COL_PLAYER_ID, config.COL_PLAYER_IDX]],
        on=config.COL_PLAYER_ID,
        how="left",
    )

    player_index_path = config.DATA_PROCESSED_DIR / config.PLAYER_INDEX_CSV
    player_index.to_csv(player_index_path, index=False)
    logger.info("Saved: %s  (%d players)", player_index_path, len(player_index))

    # ------------------------------------------------------------------
    # 3. Stratified train/test split
    #    stratify=player_idx guarantees every player appears in both sets,
    #    since sklearn draws proportional samples from each stratum.
    # ------------------------------------------------------------------
    train_df, test_df = train_test_split(
        df,
        test_size=1.0 - config.TRAIN_TEST_SPLIT,
        stratify=df[config.COL_PLAYER_IDX],
        random_state=config.RANDOM_SEED,
    )

    train_players = train_df[config.COL_PLAYER_IDX].nunique()
    test_players  = test_df[config.COL_PLAYER_IDX].nunique()
    all_in_both   = (
        set(train_df[config.COL_PLAYER_IDX]) == set(test_df[config.COL_PLAYER_IDX])
    )

    logger.info(
        "Train: %d rows | Test: %d rows | Split: %.0f/%.0f",
        len(train_df), len(test_df),
        100 * config.TRAIN_TEST_SPLIT, 100 * (1 - config.TRAIN_TEST_SPLIT),
    )
    logger.info("Train players: %d | Test players: %d", train_players, test_players)
    logger.info(
        "All players in both splits: %s",
        "YES" if all_in_both else "NO — investigate players with very few shots",
    )

    # ------------------------------------------------------------------
    # 4. Save train and test CSVs
    # ------------------------------------------------------------------
    train_path = config.DATA_PROCESSED_DIR / config.TRAIN_CSV
    test_path  = config.DATA_PROCESSED_DIR / config.TEST_CSV

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info("Saved: %s", train_path)
    logger.info("Saved: %s", test_path)

    return {
        "train":         train_path,
        "test":          test_path,
        "player_index":  player_index_path,
        "scaler_params": _SCALER_PARAMS_PATH,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config.setup_logging()
    paths = prepare()
    for key, path in paths.items():
        logger.info("Output [%-14s]: %s", key, path)
