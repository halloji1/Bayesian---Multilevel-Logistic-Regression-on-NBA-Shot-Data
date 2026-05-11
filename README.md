# Bayesian Multilevel Logistic Regression on NBA Shot Data

A hierarchical Bayesian model for predicting NBA shot outcomes, accounting for the nested structure of shots within players within teams.

## Overview

NBA shot data has a natural hierarchical structure: shots are nested within players, players within teams. Standard logistic regression violates the independence assumption and underestimates standard errors. This project uses **Bayesian multilevel logistic regression** to properly model this structure while incorporating prior knowledge from existing basketball analytics research.

## Dataset

**Source:** [NBA Shot Logs](https://www.kaggle.com/datasets/dansbecker/nba-shot-logs) (Kaggle, uploaded by Dan Becker, scraped from the NBA API)

**Coverage:** Every regular-season shot attempt in the **2014–15 NBA season**

**Size:** ~128,000 rows × 21 columns, ~16 MB CSV

**File:** `shot_logs.csv`

### Key Columns Used

| Column | Description | Role in Model |
|--------|-------------|---------------|
| `FGM` | Field goal made (1) or missed (0) | **Target** |
| `SHOT_DIST` | Shot distance from basket (feet) | Level-1 predictor |
| `CLOSE_DEF_DIST` | Distance to nearest defender (feet) | Level-1 predictor |
| `SHOT_CLOCK` | Seconds left on shot clock at release | Level-1 predictor |
| `TOUCH_TIME` | Seconds the shooter held the ball | Level-1 predictor |
| `PERIOD` | Quarter (1–4, 5+ for OT) | Used to derive `CLUTCH` |
| `GAME_CLOCK` | Time remaining in the period (`MM:SS`) | Used to derive `CLUTCH` |
| `FINAL_MARGIN` | Final score margin of the game | Used to derive `CLUTCH` (proxy) |
| `player_id`, `player_name` | Shooter identity | Level-2 grouping |
| `LOCATION` | Home (H) or Away (A) | Optional contextual covariate |
| `PTS_TYPE` | 2-point or 3-point attempt | Optional covariate |
| `SHOT_RESULT` | "made" / "missed" (redundant with FGM) | EDA only |

Other columns (`GAME_ID`, `MATCHUP`, `SHOT_NUMBER`, `DRIBBLES`, `CLOSEST_DEFENDER`, `CLOSEST_DEFENDER_PLAYER_ID`, `PTS`, `W`) are not used by the model but are retained for EDA and sanity checks.

### Known Data Quality Issues

- `SHOT_CLOCK` has ~5,500 missing values (shots taken with shot clock off, e.g. < 24s remaining in a period). These rows are dropped.
- `TOUCH_TIME` contains negative values and values above 24 seconds — both physically impossible. Negative values are dropped; values capped at 24s in some prior analyses.
- `GAME_CLOCK` is stored as a `MM:SS` string and must be parsed to seconds.
- The dataset only provides `FINAL_MARGIN` (game-level), not the live margin at the time of each shot. The `CLUTCH` flag therefore approximates clutch context using final margin ≤ 5 as a proxy — a documented limitation of the model.

### Why This Dataset

The 2014–15 NBA shot logs are the most widely used public dataset for shot-quality analysis and provide:

- **Defender proximity** — rare in publicly available shot data
- **Shot clock and touch time** — context for shot difficulty
- **Player-level grouping** — sufficient sample size per player (median ~150 shots) to support multilevel estimation after the `MIN_SHOTS_PER_PLAYER` filter

## Motivation

- Quantify how much shot success variation is attributable to player skill, team system, and situational randomness
- Provide shrinkage-based estimates for low-volume shooters (rookies, bench players)
- Output full posterior distributions for probabilistic inference (e.g., "probability that Player A is more accurate than Player B")
- Avoid Type I error inflation from ignoring the nesting structure

## Research Hypotheses

| ID | Hypothesis |
|----|-----------|
| H1 | Shot distance is negatively associated with FG% |
| H2 | Closest defender distance is positively associated with FG% |
| H3 | Clutch-time shots have lower FG% than regular shots |
| H4 | Player-level random intercept variance is significantly > 0 |
| H5 | The effect of distance varies significantly across players (random slope) |

## Model Specification

### Hierarchical Structure

```
Level 1: Shot i      (distance, defender distance, shot clock, clutch flag)
Level 2: Player j    (position, height, career FG%)
Level 3: Team k      (optional, depending on sample size)
```

### Likelihood

```
Y_ij ~ Bernoulli(p_ij)
logit(p_ij) = β_0j + β_1j · Distance_ij + β_2 · DefDist_ij
              + β_3 · ShotClock_ij + β_4 · Clutch_ij
```

### Random Effects (Player Level)

```
(β_0j, β_1j) ~ N((γ_00, γ_10), Σ)
```

### Model Building Strategy

1. **Model 0** — Null model (intercept-only) → compute ICC
2. **Model 1** — Random intercept + Level-1 fixed effects
3. **Model 2** — Add Level-2 player covariates
4. **Model 3** — Random slopes for distance
5. **Model 4** — Cross-level interactions

## Prior Specification

Two prior configurations are fitted in parallel and their posteriors compared. **Prior A** is the primary specification; **Prior B** serves as a contrast to assess robustness and the influence of prior strength on inference.

### Prior A — Weakly Informative (Primary)

A loose, regularizing prior that lets the data drive the posterior while ruling out extreme values. Aligned with Gelman et al.'s default recommendations for logistic regression.

| Parameter | Prior | Rationale |
|-----------|-------|-----------|
| Intercept γ_00 | `Normal(-0.2, 1)` | NBA league avg FG% ≈ 45.5%, logit ≈ -0.18; SD = 1 allows wide adjustment |
| Standardized slopes β_k | `Normal(0, 0.5)` | Gelman's default weakly informative prior |
| Categorical coefficients | `Normal(0, 1)` or `Student-t(3, 0, 1)` | Heavy tails for occasional large effects |
| Random effect SD τ_00 | `Half-Normal(0, 1)` | Avoids pathologies of Inverse-Gamma (Gelman 2006) |
| Correlation matrix Ω | `LKJ(η = 2)` | Mild regularization toward identity |
| Team-level SD τ_team | `Half-Normal(0, 0.5)` | Reflects expected small marginal team effect |

### Prior B — Strongly Informative / Literature-Based (Contrast)

A tight prior centered on effect sizes reported in published basketball analytics work. Used to test whether the data confirm prior literature, and to demonstrate how strong priors affect inference for low-volume players (where shrinkage is most active).

| Parameter | Prior | Rationale |
|-----------|-------|-----------|
| Intercept γ_00 | `Normal(-0.18, 0.2)` | Tightly centered on observed league-average logit FG% |
| Distance coefficient β_dist | `Normal(-0.07, 0.02)` | Per-foot effect from Chang et al. (2014), Goldsberry (2012) |
| Defender distance β_def | `Normal(0.05, 0.02)` | Positive effect of defender separation, magnitude from prior shot-quality studies |
| Shot clock β_sc | `Normal(0.02, 0.01)` | Small positive effect; rushed shots underperform |
| Clutch indicator β_clutch | `Normal(-0.10, 0.05)` | Modest negative effect consistent with clutch-performance literature |
| Random effect SD τ_00 | `Half-Normal(0, 0.4)` | Reflects that between-player logit-FG% SD rarely exceeds ~0.4 in observed data |
| Correlation matrix Ω | `LKJ(η = 4)` | Stronger pull toward independence between random intercept and slope |
| Team-level SD τ_team | `Half-Normal(0, 0.2)` | Tight: team effect is small once player skill is controlled |

### Comparison Strategy

| Comparison Axis | Prior A | Prior B |
|-----------------|---------|---------|
| Information strength | Weak | Strong |
| Source of prior | Generic regularization | Published NBA analytics |
| Expected role of data | Dominant | Balanced with prior |
| Shrinkage on low-volume players | Mild | Strong |
| Sensitivity to outlier shooters | Higher | Lower |

**Sensitivity analysis** reports for each parameter:
- Posterior mean and 95% CrI under Prior A vs. Prior B
- Difference in LOO-CV ELPD between the two specifications
- Any sign flips or substantive shifts in conclusions about H1–H5
- A vague reference prior (`Normal(0, 10)`) is also fit as a sanity check to confirm data dominance under Prior A

## Analysis Pipeline

### 1. Data Acquisition (`src/download_data.py`)
- Download from Kaggle: `dansbecker/nba-shot-logs` (see [Dataset](#dataset) section)
- Use `kagglehub` API; fall back to `kaggle` CLI
- Save `shot_logs.csv` to `data/raw/`
- Skip if file exists unless `--force` flag is passed

### 2. Data Cleaning (`src/clean_data.py`)

Input: `data/raw/shot_logs.csv` → Output: `data/processed/shots_clean.csv`

**2.1 Column selection.** Retain only columns relevant to modeling and EDA: `SHOT_DIST`, `CLOSE_DEF_DIST`, `SHOT_CLOCK`, `TOUCH_TIME`, `PERIOD`, `GAME_CLOCK`, `FGM`, `player_name`, `player_id`, `LOCATION`, `SHOT_RESULT`, `FINAL_MARGIN`, `PTS_TYPE`. Drop unused columns (`GAME_ID`, `MATCHUP`, `DRIBBLES`, `CLOSEST_DEFENDER`, `CLOSEST_DEFENDER_PLAYER_ID`, `PTS`, `W`, `SHOT_NUMBER`).

**2.2 Missing value handling.** Drop rows with missing `SHOT_CLOCK` (~5,500 rows, mostly shots taken with under 24s left in a period when the shot clock is off). No imputation — these shots are systematically different from regular-clock shots and would bias the estimate.

**2.3 Implausible value filtering.** Drop rows where:
- `TOUCH_TIME < 0` (data entry errors)
- `SHOT_DIST < 0` (impossible)
- `CLOSE_DEF_DIST < 0` (impossible)

Cap `TOUCH_TIME > 24` at 24 seconds (a possession cannot exceed the shot clock; values slightly above 24 are rounding artifacts).

**2.4 Time parsing.** Convert `GAME_CLOCK` from `"MM:SS"` string to total seconds (`GAME_CLOCK_SEC`).

**2.5 Clutch flag construction.** Create binary `CLUTCH` indicator following NBA's standard definition:
```
CLUTCH = (PERIOD >= 4) AND (GAME_CLOCK_SEC <= 300) AND (abs(FINAL_MARGIN) <= 5)
```
**Limitation:** the dataset only provides game-final margin, not the live margin at shot time. This is documented as a known approximation — a shot in a tied 4th quarter that became a blowout is incorrectly excluded, and vice versa.

**2.6 Player volume filter.** Drop players with fewer than `MIN_SHOTS_PER_PLAYER = 100` attempts.

> **Why 100?** This is a configurable empirical threshold balancing three concerns:
> - **Statistical precision:** With $n = 100$ and $p \approx 0.45$, the standard error on a player's FG% is ~5 percentage points — enough to distinguish good from poor shooters but not for fine ranking. Lower thresholds yield random effects dominated by noise rather than signal.
> - **Random slope identifiability:** H5 requires per-player distance slopes. Disentangling distance effects from baseline skill needs sufficient variance in distance per player.
> - **Computational cost:** Player count $J$ scales the random effect parameter space linearly. At $n \geq 100$, $J \approx 280$–300 players (NBA rotation regulars). At $n \geq 50$, $J$ jumps to ~360 with marginal information gain. At $n \geq 200$, $J$ drops to ~200 and excludes ~⅓ of rotation players.
>
> Sensitivity to this threshold is checked by re-running with $n \geq 50$ and $n \geq 200$ and confirming that fixed-effect posteriors are stable.

**2.7 Logging.** Record row counts before/after each filter, final shape, and number of unique players retained.

### 3. Exploratory Data Analysis (`src/eda.py`)
- Shot distance distribution, FG% by distance bucket, FG% by player (top/bottom 20), defender distance distribution, correlation heatmap
- Summary statistics written to `outputs/reports/eda_summary.txt`
- **Note:** EDA informs sanity checks but does NOT inform priors — priors are set from external literature to avoid double-dipping

### 4. Feature Preparation (`src/prepare_features.py`)

Input: `data/processed/shots_clean.csv` → Outputs: `train.csv`, `test.csv`, `player_index.csv`, `scaler_params.json`

**4.1 Standardization.** z-score the four continuous predictors, saved as `*_z` columns alongside originals:
- `SHOT_DIST_z`, `CLOSE_DEF_DIST_z`, `SHOT_CLOCK_z`, `TOUCH_TIME_z`

Standardization serves two purposes:
- Makes the prior scale (`Normal(0, 0.5)` for slopes) comparable across predictors with different units
- Reduces posterior correlation between intercept and slopes, improving MCMC mixing

Means and SDs are persisted to `scaler_params.json` for back-transformation when interpreting results.

**4.2 Player factor encoding.** Convert `player_id` to a contiguous 0-indexed integer factor `player_idx` (required for PyMC indexing). Save the mapping `player_id ↔ player_name ↔ player_idx` to `player_index.csv`.

**4.3 Train/test split.** 80/20 stratified split on `player_idx` so every player appears in both splits — necessary because random effects are player-specific and a player absent from training cannot be predicted in test. Use `RANDOM_SEED` from `config.py` for reproducibility.

**4.4 Sanity checks.** Verify:
- All players in test set also appear in training set
- Standardized predictors have mean ≈ 0, SD ≈ 1 on training set
- Class balance of `FGM` is preserved across splits (within ±1%)

### 5. Model Estimation (`src/fit_model.py`)
- PyMC implementation with **non-centered parameterization** for player random effects (avoids divergent transitions)
- Sampling: 4 chains, 2000 draws + 1000 tune, `target_accept=0.95`
- Save `InferenceData` to `outputs/models/fit_prior_{A|B}.nc`
- Convergence diagnostics: R-hat < 1.01, ESS > 400, zero divergent transitions, trace plots

### 6. Model Comparison (`src/compare_models.py`)
- **LOO-CV** (Pareto-smoothed importance sampling) — primary
- **WAIC** — secondary
- **Bayes Factor** — for hypothesis testing (with caution re: prior sensitivity)

### 7. Posterior Analysis (`src/posterior_analysis.py`)
- Posterior means, medians, 95% credible intervals
- Forest plots of fixed effects, caterpillar plots of player random intercepts
- Posterior predictive checks (PPC), trace plots

### 8. Prior Sensitivity Analysis (`src/sensitivity_analysis.py`)
- Side-by-side comparison of Prior A vs Prior B posteriors
- Mean shifts, HDI overlap, sign agreement on H1–H5
- Overlay density plots for each fixed effect

### 9. Validation (`src/validate_model.py`)
- Held-out test set: AUC, Brier score, log loss, calibration plots
- Benchmark against frequentist `lme4` model, plain logistic regression, random forest

## Tech Stack

- **Language:** R (preferred) or Python
- **Modeling:** `brms` / `rstanarm` (R), `PyMC` / `NumPyro` (Python)
- **Diagnostics:** `bayesplot`, `loo`, `ArviZ`
- **Visualization:** `ggplot2`, `tidybayes` (R), `matplotlib`, `seaborn` (Python)