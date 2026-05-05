import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_RAW_DIR       = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR        = PROJECT_ROOT / "outputs" / "figures"
MODELS_DIR         = PROJECT_ROOT / "outputs" / "models"
REPORTS_DIR        = PROJECT_ROOT / "outputs" / "reports"

# ---------------------------------------------------------------------------
# File names
# ---------------------------------------------------------------------------
RAW_CSV          = "shot_logs.csv"
CLEAN_CSV        = "shots_clean.csv"
TRAIN_CSV        = "shots_train.csv"
TEST_CSV         = "shots_test.csv"
PLAYER_INDEX_CSV = "player_index.csv"

# ---------------------------------------------------------------------------
# MCMC settings
# ---------------------------------------------------------------------------
N_CHAINS      = 4
N_DRAWS       = 2000
N_TUNE        = 1000
TARGET_ACCEPT = 0.95
RANDOM_SEED   = 42

# ---------------------------------------------------------------------------
# Cleaning / split thresholds
# ---------------------------------------------------------------------------
MIN_SHOTS_PER_PLAYER = 100
TRAIN_TEST_SPLIT     = 0.8

# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------
COL_PLAYER_ID    = "player_id"
COL_PLAYER_NAME  = "player_name"
COL_SHOT_RESULT  = "shot_result"        # binary target: 1 = made, 0 = missed
COL_SHOT_DIST    = "shot_dist"          # distance from basket (feet)
COL_CLOSE_DEF    = "close_def_dist"     # closest defender distance (feet)
COL_TOUCH_TIME   = "touch_time"         # time holding ball before shot (s)
COL_DRIBBLES     = "dribbles"           # dribbles before shot
COL_SHOT_CLOCK   = "shot_clock"         # shot-clock time remaining (s)
COL_PERIOD       = "period"             # game period (1–4+OT)
COL_GAME_ID      = "game_id"
COL_PLAYER_IDX   = "player_idx"         # integer index used in PyMC model


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
