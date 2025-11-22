"""
Shared utilities for analyzing hybrid step responses.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def compute_metrics(results: tuple[np.ndarray, ...]) -> Dict[str, float]:
    """
    Extract steady-state, overshoot, settling-time, peak, and tracking_error metrics.
    
    Parameters
    ----------
    results : tuple of np.ndarray
        Tuple of arrays from simulate_system. Should be (t, r, y, ...)
        where the first 3 arrays are (t, r, y). Additional arrays are ignored.
        - t: Time array
        - r: Reference signal array
        - y: Output response array
    """

    # Extract t, r, y from tuple (first 3 arrays)
    t = results[0]
    r = results[1]
    y = results[2]

    if len(t) == 0 or len(y) == 0:
        raise ValueError("Time and response vectors must be non-empty.")

    if len(r) != len(y):
        raise ValueError(f"Reference signal r must have same length as output y. Got {len(r)} vs {len(y)}")

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
        dt = 0.0
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

    # Compute tracking_error: sum of squared error multiplied by delta_t
    squared_error = (y - r) ** 2
    tracking_error = np.sum(squared_error) * dt

    metrics = {
        "steady_state": steady_state,
        "percent_overshoot": overshoot_pct,
        "settling_time_2pct": settling_time,
        "peak_value": peak,
        "tracking_error": tracking_error,
    }
    
    return metrics
