"""
Discrete Controller Tuning for Hybrid Closed-Loop Systems
=========================================================

This module searches for a discrete-time controller `D[z]` with arbitrary
polynomials (not restricted to PID form) that satisfies step-response
specifications for a given continuous plant with sampler/ZOH interface.

It leverages the hybrid simulation utilities defined in `simulate_hybrid_system`
to evaluate candidate controllers directly on the mixed continuous/discrete loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from scipy.optimize import differential_evolution

from .response_metrics import compute_step_metrics
from .simulate_hybrid_system import simulate_hybrid_step_response


@dataclass(frozen=True)
class PerformanceSpecs:
    """
    Target performance requirements for the closed-loop step response.

    Attributes
    ----------
    max_overshoot_pct : float
        Maximum allowed percent overshoot relative to final value.
    settling_time_2pct : float
        Required 2% settling time (seconds). Settling time is defined as the
        first time after which the response stays within ±2% of the steady-state.
    """

    max_overshoot_pct: float
    settling_time_2pct: float


def params_to_discrete_tf(
    params: np.ndarray, num_order: int, den_order: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map optimization parameters to a proper discrete-time transfer function.

    Denominator leading coefficient is fixed at 1 to remove scale ambiguity.
    """

    num_coeffs = num_order + 1
    if len(params) != num_coeffs + den_order:
        raise ValueError(
            f"Expected {num_coeffs + den_order} params, received {len(params)}"
        )

    num = np.asarray(params[:num_coeffs], dtype=float)
    den_rest = np.asarray(params[num_coeffs:], dtype=float)
    den = np.concatenate(([1.0], den_rest))

    return num, den


def _make_default_bounds(
    total_params: int, magnitude: float = 2.0
) -> Sequence[Tuple[float, float]]:
    """
    Helper to construct symmetric parameter bounds.
    """

    return [(-magnitude, magnitude)] * total_params


def tune_discrete_controller(
    plant_tf: Tuple[Sequence[float], Sequence[float]],
    sampling_time: float,
    specs: PerformanceSpecs,
    num_order: int = 2,
    den_order: int = 2,
    t_end: float = 5.0,
    step_amplitude: float = 1.0,
    bounds: Sequence[Tuple[float, float]] | None = None,
    popsize: int = 25,
    maxiter: int = 60,
    random_state: int | None = None,
    verbose: bool = True,
    u_min: float | None = None,
    u_max: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Search for discrete controller coefficients meeting the provided specs.

    Returns the best (num, den, metrics) tuple discovered by the optimizer.
    """

    num_params = num_order + 1
    total_params = num_params + den_order
    if bounds is None:
        bounds = _make_default_bounds(total_params)

    if len(bounds) != total_params:
        raise ValueError("Bounds must be provided for each parameter.")

    best_record: dict[str, np.ndarray | dict | float | None] = {
        "params": None,
        "metrics": None,
        "cost": np.inf,
    }

    def objective(param_vec: np.ndarray) -> float:
        try:
            controller_tf = params_to_discrete_tf(param_vec, num_order, den_order)
            t, y, *_ = simulate_hybrid_step_response(
                controller_tf,
                plant_tf,
                sampling_time,
                t_end=t_end,
                step_amplitude=step_amplitude,
                u_min=u_min,
                u_max=u_max,
            )
            metrics = compute_step_metrics(t, y, reference=step_amplitude)
        except Exception:
            # Penalize unstable or failed simulations
            return 1e6

        overshoot_penalty = max(
            0.0, metrics["percent_overshoot"] - specs.max_overshoot_pct
        )

        # Stronger penalty for settling time violations
        settling_penalty = 0.0
        if not np.isfinite(metrics["settling_time_2pct"]):
            # If doesn't settle, use a large penalty based on simulation time
            # This encourages the optimizer to find solutions that actually settle
            settling_penalty = t_end + 10.0 * specs.settling_time_2pct
        else:
            # Use squared penalty to make violations more expensive
            violation = max(
                0.0, metrics["settling_time_2pct"] - specs.settling_time_2pct
            )
            settling_penalty = violation**2

        # Encourage steady-state accuracy
        steady_state_error = abs(metrics["steady_state"] - step_amplitude)

        # Weighted sum cost
        cost = (
            5.0 * overshoot_penalty + 2.0 * settling_penalty + 3.0 * steady_state_error
        )

        if cost < best_record["cost"]:
            best_record["cost"] = cost
            best_record["params"] = param_vec.copy()
            best_record["metrics"] = metrics

        return cost

    result = differential_evolution(
        objective,
        bounds,
        popsize=popsize,
        maxiter=maxiter,
        seed=random_state,
        polish=True,
        updating="deferred",
        disp=verbose,
    )

    if best_record["params"] is None:
        raise RuntimeError(
            "Failed to find a stable controller within the provided bounds/specs. "
            "Consider widening bounds, relaxing specs, or adjusting controller order."
        )

    best_num, best_den = params_to_discrete_tf(
        best_record["params"], num_order, den_order
    )
    metrics = best_record["metrics"]

    if verbose:
        print("Tuning complete:")
        print(f"  Optimizer reported cost: {result.fun:.4f}")
        print(f"  Best stable cost: {best_record['cost']:.4f}")
        print(f"  Controller numerator: {best_num}")
        print(f"  Controller denominator: {best_den}")
        print("  Metrics:")
        for key, value in metrics.items():
            print(f"    {key}: {value}")

    return best_num, best_den, metrics


if __name__ == "__main__":
    """
    Example usage for tuning a 2nd-order discrete controller.
    """

    example_specs = PerformanceSpecs(max_overshoot_pct=10.0, settling_time_2pct=1.5)

    plant_num = [-2.936]
    plant_den = [0.031, 1.0, 0.0]
    sampling_time = 0.015

    num, den, metrics = tune_discrete_controller(
        plant_tf=(plant_num, plant_den),
        sampling_time=sampling_time,
        specs=example_specs,
        num_order=2,
        den_order=2,
        t_end=5.0,
        step_amplitude=1.0,
        popsize=10,
        maxiter=30,
        random_state=42,
        verbose=True,
    )

    print("\nBest found controller D[z]:")
    print(f"  Numerator coefficients: {num}")
    print(f"  Denominator coefficients: {den}")
    print("\nClosed-loop metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
