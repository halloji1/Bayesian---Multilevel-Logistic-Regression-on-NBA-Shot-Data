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
(β_0j, β_1j) ~ MVN((γ_00, γ_10), Σ)
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

### 1. Data Acquisition & Cleaning
- Source: Kaggle NBA Shot Logs (see [Dataset](#dataset) section above)
- Drop rows with missing `SHOT_CLOCK`, negative `TOUCH_TIME`, or negative distances
- Parse `GAME_CLOCK` from `MM:SS` to seconds
- Construct `CLUTCH` flag from `PERIOD`, `GAME_CLOCK`, and `FINAL_MARGIN`
- Filter to players with ≥ 100 shot attempts to stabilize random effect estimation

### 2. Exploratory Data Analysis
- Shot charts, FG% by player/team/position, distribution checks, correlation analysis
- **Note:** Avoid double-dipping — priors should be set from external knowledge, not the current dataset

### 3. Variable Preparation
- z-score standardization of continuous predictors (critical for prior scale comparability)
- Dummy / effect coding for categorical variables

### 4. Model Estimation (MCMC)
- **R:** `brms` (Stan backend)
- **Python:** `PyMC` or `NumPyro`
- Convergence diagnostics: R-hat < 1.01, ESS > 400, zero divergent transitions, trace plots

### 5. Model Comparison
- **LOO-CV** (Pareto-smoothed importance sampling) — primary
- **WAIC** — secondary
- **Bayes Factor** — for hypothesis testing (with caution re: prior sensitivity)

### 6. Posterior Analysis
- Posterior means, medians, 95% credible intervals
- Posterior density plots, caterpillar plots for player effects
- Posterior predictive checks (PPC)

### 7. Prior Sensitivity Analysis
- Compare posteriors across the three prior specifications

### 8. Validation
- Held-out test set: AUC, Brier score, calibration plots
- Benchmark against frequentist `lme4` model, plain logistic regression, random forest

## Tech Stack

- **Language:** R (preferred) or Python
- **Modeling:** `brms` / `rstanarm` (R), `PyMC` / `NumPyro` (Python)
- **Diagnostics:** `bayesplot`, `loo`, `ArviZ`
- **Visualization:** `ggplot2`, `tidybayes` (R), `matplotlib`, `seaborn` (Python)

## Sample Code (brms)

```r
library(brms)

# ---- Prior A: Weakly Informative ----
priors_A <- c(
  prior(normal(-0.2, 1),  class = "Intercept"),
  prior(normal(0, 0.5),   class = "b"),
  prior(normal(0, 1),     class = "sd"),
  prior(lkj(2),           class = "cor")
)

# ---- Prior B: Strongly Informative (Literature-Based) ----
priors_B <- c(
  prior(normal(-0.18, 0.2),  class = "Intercept"),
  prior(normal(-0.07, 0.02), class = "b", coef = "Distance"),
  prior(normal(0.05, 0.02),  class = "b", coef = "DefDist"),
  prior(normal(0.02, 0.01),  class = "b", coef = "ShotClock"),
  prior(normal(-0.10, 0.05), class = "b", coef = "Clutch"),
  prior(normal(0, 0.4),      class = "sd"),
  prior(lkj(4),              class = "cor")
)

formula <- made ~ Distance + DefDist + ShotClock + Clutch +
                  (1 + Distance | player_id)

fit_A <- brm(formula, data = shots, family = bernoulli(),
             prior = priors_A, chains = 4, iter = 4000,
             warmup = 1000, cores = 4,
             control = list(adapt_delta = 0.95))

fit_B <- brm(formula, data = shots, family = bernoulli(),
             prior = priors_B, chains = 4, iter = 4000,
             warmup = 1000, cores = 4,
             control = list(adapt_delta = 0.95))

# ---- Compare ----
loo_compare(loo(fit_A), loo(fit_B))
```

## Expected Deliverables

- Full posterior distributions for all parameters under **both Prior A and Prior B**
- Side-by-side comparison of posterior estimates and 95% CrIs across the two priors
- LOO-CV / WAIC comparison between Prior A and Prior B fits
- Player ability rankings with credible intervals (highlighting where shrinkage differs between priors)
- Situational shot probability prediction tool
- Posterior predictive checks for both specifications
- Sensitivity analysis report quantifying prior influence on H1–H5

## Known Challenges

| Challenge | Mitigation |
|-----------|-----------|
| MCMC divergent transitions | Non-centered parameterization, raise `adapt_delta` to 0.99 |
| Long sampling time on ~128k rows | Variational inference (VI) for prototyping |
| Subjective prior critique | Document sensitivity analysis, justify priors from literature |
| Low-volume players | Bayesian shrinkage handles this naturally; optionally apply minimum-shot threshold |

## References

- Gelman, A. (2006). Prior distributions for variance parameters in hierarchical models.
- Lewandowski, D., Kurowicka, D., & Joe, H. (2009). Generating random correlation matrices based on vines and extended onion method.
- Chang, Y. H., et al. (2014). Quantifying shot quality in the NBA.
- Goldsberry, K. (2012). CourtVision: New visual and spatial analytics for the NBA.
- Nakagawa, S., & Schielzeth, H. (2013). A general and simple method for obtaining R² from generalized linear mixed-effects models.