from __future__ import annotations
import json


def get_prior_A() -> dict:
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
