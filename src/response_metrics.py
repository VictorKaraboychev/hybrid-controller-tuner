"""
Shared utilities for analyzing hybrid step responses.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def compute_step_metrics(t: np.ndarray, y: np.ndarray) -> Dict[str, float]:
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

    # Settling time: first time after which response stays within 2% band
    # for the remainder of simulation (or at least 0.1s if near the end)
    if len(t) > 1:
        dt = t[1] - t[0]
        min_settled_samples = max(1, int(0.1 / dt))  # At least 0.1s
    else:
        min_settled_samples = 1

    for idx in range(len(y)):
        remaining_samples = len(y) - idx

        # Need at least min_settled_samples remaining to consider settling
        if remaining_samples < min_settled_samples:
            continue

        # Check if response stays within band for the remainder
        if np.all(np.abs(y[idx:] - steady_state) <= band):
            settling_time = t[idx]
            break

    return {
        "steady_state": steady_state,
        "percent_overshoot": overshoot_pct,
        "settling_time_2pct": settling_time,
        "peak_value": peak,
    }
