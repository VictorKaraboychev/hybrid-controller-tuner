"""
Shared utilities for analyzing hybrid step responses.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def compute_step_metrics(
    t: np.ndarray, y: np.ndarray, reference: float = 1.0
) -> Dict[str, float]:
    """
    Extract steady-state, overshoot, settling-time, and peak metrics.
    """

    if len(t) == 0 or len(y) == 0:
        raise ValueError("Time and response vectors must be non-empty.")

    steady_state = y[-1]
    peak = np.max(y)
    overshoot_pct = (
        0.0
        if np.isclose(steady_state, 0.0)
        else max(0.0, (peak - steady_state) / max(abs(steady_state), 1e-9) * 100.0)
    )

    band = 0.02 * max(abs(steady_state), 1e-9)  # 2% band
    settling_time = np.inf
    for idx in range(len(y)):
        if np.abs(y[idx] - steady_state) <= band and np.all(
            np.abs(y[idx:] - steady_state) <= band
        ):
            settling_time = t[idx]
            break

    return {
        "steady_state": steady_state,
        "percent_overshoot": overshoot_pct,
        "settling_time_2pct": settling_time,
        "peak_value": peak,
    }

