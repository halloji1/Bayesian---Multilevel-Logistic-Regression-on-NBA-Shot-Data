# Bayesian Multilevel Logistic Regression on NBA Shot Data

A hierarchical Bayesian model for predicting NBA shot outcomes, accounting for the nested structure of shots within players within teams.

## Overview

NBA shot data has a natural hierarchical structure: shots are nested within players, players within teams. Standard logistic regression violates the independence assumption and underestimates standard errors. This project uses **Bayesian multilevel logistic regression** to properly model this structure while incorporating prior knowledge from existing basketball analytics research.

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

All priors are weakly informative by default, with optional informative versions based on basketball analytics literature.

| Parameter | Prior | Rationale |
|-----------|-------|-----------|
| Intercept γ_00 | `Normal(-0.2, 1)` | NBA league avg FG% ≈ 45.5%, logit ≈ -0.18 |
| Standardized slopes β_k | `Normal(0, 0.5)` | Gelman's default weakly informative prior |
| Distance coefficient (informative) | `Normal(-0.07, 0.03)` | Based on Chang et al. (2014), Goldsberry (2012) |
| Categorical coefficients | `Normal(0, 1)` or `Student-t(3, 0, 1)` | Heavy tails for occasional large effects |
| Random effect SD τ_00 | `Half-Normal(0, 1)` | Avoids pathologies of Inverse-Gamma (Gelman 2006) |
| Correlation matrix Ω | `LKJ(η = 2)` | Mild regularization toward identity |
| Team-level SD τ_team | `Half-Normal(0, 0.5)` | Reflects expected small marginal team effect |

### Prior Sensitivity Analysis

Three versions are tested for each model:

1. **Weakly informative** (default, as above)
2. **Tighter informative** (SDs halved, more domain knowledge)
3. **Vague reference** (`Normal(0, 10)` to validate data dominance)

## Analysis Pipeline

### 1. Data Acquisition & Cleaning
- Source: Kaggle NBA Shot Logs (2014-15, ~128k shots) or `nba_api`
- Handle missing defender distances, drop technical free throws, merge player attribute tables

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

priors <- c(
  prior(normal(-0.2, 1),    class = "Intercept"),
  prior(normal(0, 0.5),     class = "b"),
  prior(normal(-0.07, 0.03), class = "b", coef = "Distance"),
  prior(normal(0, 1),       class = "sd"),
  prior(lkj(2),             class = "cor")
)

fit <- brm(
  made ~ Distance + DefDist + ShotClock + Clutch +
         (1 + Distance | player_id),
  data    = shots,
  family  = bernoulli(),
  prior   = priors,
  chains  = 4,
  iter    = 4000,
  warmup  = 1000,
  cores   = 4,
  control = list(adapt_delta = 0.95)
)
```

## Expected Deliverables

- Full posterior distributions for all parameters
- LOO-CV model comparison report
- Prior sensitivity analysis report
- Player ability rankings with credible intervals
- Situational shot probability prediction tool
- Posterior predictive visualizations

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

## License

TBD

## Contact

TBD
