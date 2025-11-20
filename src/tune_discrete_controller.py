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
    max_control_signal : float, optional
        Maximum allowed absolute value of control signal. If provided, violations
        of this limit will be penalized. If None, no control signal limit is enforced.
        Default: None (disabled).
    """

    max_overshoot_pct: float
    settling_time_2pct: float
    max_control_signal: float | None = None


@dataclass(frozen=True)
class CostWeights:
    """
    Weights for different components of the cost function.

    Attributes
    ----------
    overshoot_weight : float
        Weight for overshoot penalty. Default: 1.0
    settling_time_weight : float
        Weight for settling time penalty. Default: 2.0
    steady_state_error_weight : float
        Weight for steady-state error. Default: 3.0
    control_signal_limit_weight : float
        Weight for control signal limit violation penalty. Default: 1.0
    """

    overshoot_weight: float = 1.0
    settling_time_weight: float = 2.0
    steady_state_error_weight: float = 3.0
    control_signal_limit_weight: float = 1.0


def params_to_discrete_tf(
    params: np.ndarray, num_order: int, den_order: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map optimization parameters to a strictly proper discrete-time transfer function.

    A strictly proper transfer function has degree(numerator) < degree(denominator).
    This ensures the controller is causal and implementable.

    Parameters:
    -----------
    params : np.ndarray
        Optimization parameters: [num_coeffs..., den_coeffs...]
    num_order : int
        Numerator order (degree). Resulting numerator has num_order + 1 coefficients.
    den_order : int
        Denominator order (degree). Resulting denominator has den_order + 1 coefficients.
        The leading denominator coefficient is fixed at 1.0.

    Returns:
    --------
    num : np.ndarray
        Numerator coefficients in descending powers of z
    den : np.ndarray
        Denominator coefficients in descending powers of z (leading coefficient is 1.0)

    Raises:
    -------
    ValueError
        If num_order >= den_order (not strictly proper)
        If incorrect number of parameters provided
    """
    # Enforce strict properness: degree(num) < degree(den)
    if num_order >= den_order:
        raise ValueError(
            f"Controller must be strictly proper: num_order ({num_order}) must be < den_order ({den_order}). "
            f"Current: degree(num) = {num_order}, degree(den) = {den_order}"
        )

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


def _make_objective_function(
    plant_tf: Tuple[Sequence[float], Sequence[float]],
    sampling_time: float,
    specs: PerformanceSpecs,
    num_order: int,
    den_order: int,
    t_end: float,
    step_amplitude: float,
    best_record: dict,
    cost_weights: CostWeights | None = None,
) -> callable:
    """
    Create an objective function for controller optimization.

    Parameters
    ----------
    plant_tf : tuple
        Plant transfer function (num, den) in continuous-time
    sampling_time : float
        Sampling time in seconds
    specs : PerformanceSpecs
        Performance requirements
    num_order : int
        Numerator order
    den_order : int
        Denominator order
    t_end : float
        Simulation end time
    step_amplitude : float
        Step input amplitude
    best_record : dict
        Dictionary to store best parameters, metrics, and cost
    cost_weights : CostWeights, optional
        Weights for cost function components. If None, uses default weights.

    Returns
    -------
    callable
        Objective function that takes parameter vector and returns cost
    """
    if cost_weights is None:
        cost_weights = CostWeights()

    def objective(param_vec: np.ndarray) -> float:
        try:
            controller_tf = params_to_discrete_tf(param_vec, num_order, den_order)
            t, y, u, _ = simulate_hybrid_step_response(
                controller_tf,
                plant_tf,
                sampling_time,
                t_end=t_end,
                step_amplitude=step_amplitude,
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

        # Control signal limit penalty (only penalize if limit is exceeded)
        control_signal_penalty = 0.0
        if specs.max_control_signal is not None:
            max_control_magnitude = np.max(np.abs(u))
            
            violation = max(0.0, max_control_magnitude - specs.max_control_signal)
            control_signal_penalty = violation**2  # Squared penalty for violations

        # Weighted sum cost
        constraints = (
            cost_weights.overshoot_weight * overshoot_penalty
            + cost_weights.settling_time_weight * settling_penalty
            + cost_weights.steady_state_error_weight * steady_state_error
            + cost_weights.control_signal_limit_weight * control_signal_penalty
        )
        
        objectives = (
            cost_weights.settling_time_weight * metrics["settling_time_2pct"]
        )
        
        # Solve the constraints before the objectives become important
        cost = 100.0 * constraints + objectives

        if cost < best_record["cost"]:
            best_record["cost"] = cost
            best_record["params"] = param_vec.copy()
            best_record["metrics"] = metrics

        return cost

    return objective


def _compute_cost_from_metrics(
    metrics: dict,
    specs: PerformanceSpecs,
    step_amplitude: float,
    t_end: float,
    max_control_magnitude: float | None = None,
    cost_weights: CostWeights | None = None,
) -> float:
    """
    Compute cost from performance metrics.

    This is useful when you already have metrics and control signal data,
    without needing to run the full objective function.

    Parameters
    ----------
    metrics : dict
        Performance metrics from compute_step_metrics
    specs : PerformanceSpecs
        Performance requirements
    step_amplitude : float
        Step input amplitude
    t_end : float
        Simulation end time
    max_control_magnitude : float, optional
        Maximum absolute value of control signal. If None, control signal cost is 0.
    cost_weights : CostWeights, optional
        Weights for cost function components. If None, uses default weights.

    Returns
    -------
    float
        Computed cost value
    """
    if cost_weights is None:
        cost_weights = CostWeights()

    overshoot_penalty = max(
        0.0, metrics["percent_overshoot"] - specs.max_overshoot_pct
    )

    # Stronger penalty for settling time violations
    settling_penalty = 0.0
    if not np.isfinite(metrics["settling_time_2pct"]):
        # If doesn't settle, use a large penalty based on simulation time
        settling_penalty = t_end + 10.0 * specs.settling_time_2pct
    else:
        # Use squared penalty to make violations more expensive
        violation = max(
            0.0, metrics["settling_time_2pct"] - specs.settling_time_2pct
        )
        settling_penalty = violation**2

    # Encourage steady-state accuracy
    steady_state_error = abs(metrics["steady_state"] - step_amplitude)

    # Control signal limit penalty (only penalize if limit is exceeded)
    control_signal_penalty = 0.0
    if specs.max_control_signal is not None and max_control_magnitude is not None:
        if max_control_magnitude > specs.max_control_signal:
            # Penalize violation of control signal limit
            violation = max_control_magnitude - specs.max_control_signal
            control_signal_penalty = violation**2  # Squared penalty for violations

    # Weighted sum cost
    cost = (
        cost_weights.overshoot_weight * overshoot_penalty
        + cost_weights.settling_time_weight * settling_penalty
        + cost_weights.steady_state_error_weight * steady_state_error
        + cost_weights.control_signal_limit_weight * control_signal_penalty
    )

    return cost


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
    cost_weights: CostWeights | None = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Search for discrete controller coefficients meeting the provided specs.

    The controller is constrained to be strictly proper: degree(numerator) < degree(denominator).
    This ensures the controller is causal and physically realizable.

    Parameters:
    -----------
    plant_tf : tuple
        Plant transfer function (num, den) in continuous-time
    sampling_time : float
        Sampling time in seconds
    specs : PerformanceSpecs
        Performance requirements (overshoot, settling time, control signal limit)
    num_order : int, optional
        Numerator order (degree). The numerator will have num_order + 1 coefficients.
        Default: 2 (2nd order numerator = 3 coefficients)
    den_order : int, optional
        Denominator order (degree). The denominator will have den_order + 1 coefficients
        (leading coefficient fixed at 1.0). Must be > num_order for strict properness.
        Default: 2 (2nd order denominator = 3 coefficients)
    t_end : float, optional
        Simulation end time in seconds
    step_amplitude : float, optional
        Step input amplitude
    bounds : sequence of tuples, optional
        Parameter bounds for optimization [(min, max), ...]
    popsize : int, optional
        Population size for differential evolution
    maxiter : int, optional
        Maximum iterations for differential evolution
    random_state : int, optional
        Random seed for reproducibility
    verbose : bool, optional
        Print optimization progress
    cost_weights : CostWeights, optional
        Weights for cost function components. If None, uses default weights.

    Returns:
    --------
    num : np.ndarray
        Best numerator coefficients found
    den : np.ndarray
        Best denominator coefficients found
    metrics : dict
        Performance metrics of the best controller

    Raises:
    ------
    ValueError
        If num_order >= den_order (controller would not be strictly proper)
    """
    # Validate strict properness requirement
    if num_order >= den_order:
        raise ValueError(
            f"Controller must be strictly proper: num_order ({num_order}) must be < den_order ({den_order}). "
            f"This ensures degree(numerator) < degree(denominator)."
        )

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

    objective = _make_objective_function(
        plant_tf=plant_tf,
        sampling_time=sampling_time,
        specs=specs,
        num_order=num_order,
        den_order=den_order,
        t_end=t_end,
        step_amplitude=step_amplitude,
        best_record=best_record,
        cost_weights=cost_weights,
    )

    result = differential_evolution(
        objective,
        bounds,
        popsize=popsize,
        maxiter=maxiter,
        seed=random_state,
        polish=True,
        updating="deferred",
        disp=verbose,
        tol=0.001,
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
        print(f"  Controller numerator: {','.join(f'{x:.6f}' for x in best_num)}")
        print(f"  Controller denominator: {','.join(f'{x:.6f}' for x in best_den)}")
        print("  Metrics:")
        for key, value in metrics.items():
            print(f"    {key}: {value:.4f}")

    return best_num, best_den, metrics


def tune_discrete_controller_with_order_search(
    plant_tf: Tuple[Sequence[float], Sequence[float]],
    sampling_time: float,
    specs: PerformanceSpecs,
    num_order_range: Tuple[int, int] = (0, 3),
    den_order_range: Tuple[int, int] = (1, 4),
    t_end: float = 5.0,
    step_amplitude: float = 1.0,
    bounds: Sequence[Tuple[float, float]] | None = None,
    popsize: int = 25,
    maxiter: int = 60,
    random_state: int | None = None,
    verbose: bool = True,
    cost_weights: CostWeights | None = None,
) -> Tuple[np.ndarray, np.ndarray, dict, int, int]:
    """
    Search for the best controller by trying different combinations of poles and zeros.

    This function tries different combinations of numerator and denominator orders,
    optimizes parameters for each combination, and returns the best overall result.

    Parameters:
    -----------
    plant_tf : tuple
        Plant transfer function (num, den) in continuous-time
    sampling_time : float
        Sampling time in seconds
    specs : PerformanceSpecs
        Performance requirements (overshoot, settling time)
    num_order_range : tuple of int, optional
        (min, max) range for numerator order to search. Default: (0, 3)
    den_order_range : tuple of int, optional
        (min, max) range for denominator order to search. Default: (1, 4)
    t_end : float, optional
        Simulation end time in seconds
    step_amplitude : float, optional
        Step input amplitude
    bounds : sequence of tuples, optional
        Parameter bounds for optimization. If None, uses default bounds based on order.
    popsize : int, optional
        Population size for differential evolution
    maxiter : int, optional
        Maximum iterations for differential evolution
    random_state : int, optional
        Random seed for reproducibility
    verbose : bool, optional
        Print optimization progress
    cost_weights : CostWeights, optional
        Weights for cost function components. If None, uses default weights.

    Returns:
    --------
    num : np.ndarray
        Best numerator coefficients found
    den : np.ndarray
        Best denominator coefficients found
    metrics : dict
        Performance metrics of the best controller
    best_num_order : int
        Numerator order of the best controller
    best_den_order : int
        Denominator order of the best controller
    """
    num_min, num_max = num_order_range
    den_min, den_max = den_order_range

    # Generate all valid combinations (num_order < den_order for strict properness)
    combinations = []
    for num_order in range(num_min, num_max + 1):
        for den_order in range(max(den_min, num_order + 1), den_max + 1):
            combinations.append((num_order, den_order))

    if len(combinations) == 0:
        raise ValueError(
            f"No valid combinations found. Ensure den_order_range allows values > num_order_range. "
            f"num_order_range={num_order_range}, den_order_range={den_order_range}"
        )

    if verbose:
        print(f"Searching over {len(combinations)} controller structures:")
        for num_order, den_order in combinations:
            print(
                f"  num_order={num_order}, den_order={den_order} (degree(num)={num_order}, degree(den)={den_order})"
            )
        print()

    best_overall = {
        "num": None,
        "den": None,
        "metrics": None,
        "cost": np.inf,
        "num_order": None,
        "den_order": None,
    }

    for idx, (num_order, den_order) in enumerate(combinations, 1):
        if verbose:
            print(f"\n{'='*60}")
            print(
                f"Trying combination {idx}/{len(combinations)}: num_order={num_order}, den_order={den_order}"
            )
            print(f"{'='*60}")

        try:
            # Use default bounds for this combination if not provided
            # If bounds are provided, they must match the number of parameters for this combination
            num_params = num_order + 1
            total_params = num_params + den_order
            if bounds is None:
                combo_bounds = _make_default_bounds(total_params)
            else:
                if len(bounds) != total_params:
                    if verbose:
                        print(
                            f"  ⚠ Skipping: provided bounds ({len(bounds)} params) don't match "
                            f"this combination ({total_params} params)"
                        )
                    continue
                combo_bounds = bounds

            num, den, metrics = tune_discrete_controller(
                plant_tf=plant_tf,
                sampling_time=sampling_time,
                specs=specs,
                num_order=num_order,
                den_order=den_order,
                t_end=t_end,
                step_amplitude=step_amplitude,
                bounds=combo_bounds,
                popsize=popsize,
                maxiter=maxiter,
                random_state=random_state,
                verbose=False,  # Suppress individual tuning output
                cost_weights=cost_weights,
            )

            # Compute cost for comparison (using same cost function as optimization)
            # Need to simulate again to get control signal for cost calculation
            try:
                t_test, y_test, u_test, _ = simulate_hybrid_step_response(
                    (num, den),
                    plant_tf,
                    sampling_time,
                    t_end=t_end,
                    step_amplitude=step_amplitude,
                )
                max_control_magnitude = np.max(np.abs(u_test))
            except Exception:
                max_control_magnitude = np.inf

            cost = _compute_cost_from_metrics(
                metrics=metrics,
                specs=specs,
                step_amplitude=step_amplitude,
                t_end=t_end,
                max_control_magnitude=max_control_magnitude,
                cost_weights=cost_weights,
            )

            if verbose:
                control_info = ""
                if specs.max_control_signal is not None:
                    control_info = f", max|u|={max_control_magnitude:.4f}"
                print(
                    f"  Result: cost={cost:.4f}, settling_time={metrics['settling_time_2pct']:.4f}s, "
                    f"overshoot={metrics['percent_overshoot']:.2f}%{control_info}"
                )

            # Update best if this is better
            if cost < best_overall["cost"]:
                best_overall["num"] = num
                best_overall["den"] = den
                best_overall["metrics"] = metrics
                best_overall["cost"] = cost
                best_overall["num_order"] = num_order
                best_overall["den_order"] = den_order

                if verbose:
                    print(f"  ✓ New best! (cost improved to {cost:.4f})")

        except Exception as e:
            if verbose:
                print(f"  ✗ Failed: {e}")
            continue

    if best_overall["num"] is None:
        raise RuntimeError(
            "Failed to find any valid controller across all tested combinations. "
            "Consider: widening bounds, relaxing specs, or expanding order ranges."
        )

    if verbose:
        print(f"\n{'='*60}")
        print("BEST OVERALL CONTROLLER:")
        print(f"{'='*60}")
        print(
            f"  Structure: num_order={best_overall['num_order']}, den_order={best_overall['den_order']}"
        )
        print(f"  Cost: {best_overall['cost']:.4f}")
        print(f"  Numerator: {','.join(f'{x:.6f}' for x in best_overall['num'])}")
        print(f"  Denominator: {','.join(f'{x:.6f}' for x in best_overall['den'])}")
        print("  Metrics:")
        for key, value in best_overall["metrics"].items():
            print(f"    {key}: {value}")

    return (
        best_overall["num"],
        best_overall["den"],
        best_overall["metrics"],
        best_overall["num_order"],
        best_overall["den_order"],
    )
