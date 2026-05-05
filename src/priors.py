"""Prior specifications for the Bayesian multilevel logistic regression model.

Two named priors are available:
  "A" — Weakly informative: broad distributions, minimal domain assumptions.
  "B" — Strongly informative: tightened around published NBA shooting literature.

Each prior dict is consumed directly by the PyMC model builder in model.py.
"""

from __future__ import annotations

import json


def get_prior_A() -> dict:
    """Return weakly informative prior specification (Prior A).

    All slopes are centred at zero with modest shrinkage (sigma=0.5 on
    z-scored predictors).  The intercept reflects a league FG% slightly
    below 50% (logit(-0.2) ≈ 45%).

    Returns:
        Dict with keys: name, description, intercept, slopes,
        random_effect_sd, lkj_eta.
    """
    return {
        "name": "A",
        "description": "Weakly informative — broad Normal(0, 0.5) slopes, HalfNormal(1) random-effect SD",
        "intercept": {"mu": -0.2, "sigma": 1.0},
        "slopes": {
            "distance":   {"mu": 0.0, "sigma": 0.5},
            "def_dist":   {"mu": 0.0, "sigma": 0.5},
            "shot_clock": {"mu": 0.0, "sigma": 0.5},
            "clutch":     {"mu": 0.0, "sigma": 0.5},
        },
        "random_effect_sd": {"sigma": 1.0},
        "lkj_eta": 2,
    }


def get_prior_B() -> dict:
    """Return strongly informative prior specification (Prior B).

    Slopes are anchored to published estimates of how each predictor
    affects FG% on z-scored inputs:
      - distance:   negative effect, ~7 pp per SD
      - def_dist:   positive effect (open shots make more), ~5 pp per SD
      - shot_clock: small positive effect (more time ≈ better looks), ~2 pp per SD
      - clutch:     negative effect (pressure reduces accuracy), ~10 pp

    The tighter random-effect SD (HalfNormal(0.4)) reflects that player
    skill differences, while real, are moderate on the log-odds scale.
    Higher LKJ eta (4) shrinks random-effect correlations toward zero.

    Returns:
        Dict with keys: name, description, intercept, slopes,
        random_effect_sd, lkj_eta.
    """
    return {
        "name": "B",
        "description": "Strongly informative — literature-based slopes, HalfNormal(0.4) random-effect SD",
        "intercept": {"mu": -0.18, "sigma": 0.2},
        "slopes": {
            "distance":   {"mu": -0.07, "sigma": 0.02},
            "def_dist":   {"mu":  0.05, "sigma": 0.02},
            "shot_clock": {"mu":  0.02, "sigma": 0.01},
            "clutch":     {"mu": -0.10, "sigma": 0.05},
        },
        "random_effect_sd": {"sigma": 0.4},
        "lkj_eta": 4,
    }


def get_prior(name: str) -> dict:
    """Dispatch to the named prior specification.

    Args:
        name: "A" for weakly informative, "B" for strongly informative.

    Returns:
        Prior specification dict.

    Raises:
        ValueError: If name is not "A" or "B".
    """
    registry: dict[str, callable] = {
        "A": get_prior_A,
        "B": get_prior_B,
    }
    if name not in registry:
        raise ValueError(
            f"Unknown prior name {name!r}. Valid options: {sorted(registry)}"
        )
    return registry[name]()


if __name__ == "__main__":
    for label in ("A", "B"):
        prior = get_prior(label)
        print(f"Prior {label}: {prior['description']}")
        print(json.dumps(prior, indent=2))
        print()
