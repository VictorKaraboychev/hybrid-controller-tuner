"""
Discrete Controller Tuning for Hybrid Closed-Loop Systems
=========================================================

This module searches for a discrete-time controller `D[z]` with arbitrary
polynomials (not restricted to PID form) that satisfies step-response
specifications for a user-defined plant with sampler/ZOH interface.

It leverages the hybrid simulation utilities defined in `simulate_user_system`
to evaluate candidate controllers directly on the mixed continuous/discrete loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple, Union

import numpy as np
from scipy.optimize import differential_evolution

from .response_metrics import compute_step_metrics
from .simulate_system import simulate_system


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


@dataclass(frozen=True)
class SystemParameters:
    """
    System and simulation parameters.

    Attributes
    ----------
    num_parameters : int
        Number of optimization parameters. The System class determines
        how these parameters are interpreted.
    t_end : float, optional
        Simulation end time (seconds). Default: 5.0
    step_amplitude : float, optional
        Step input amplitude. Default: 1.0
    dt : float, optional
        Time step for continuous plant simulation (seconds). Default: 0.001
    """

    num_parameters: int
    t_end: float = 5.0
    step_amplitude: float = 1.0
    dt: float = 0.001


@dataclass(frozen=True)
class OptimizationParameters:
    """
    Optimization algorithm parameters.

    Attributes
    ----------
    population : int, optional
        Population size for differential evolution. Default: 25
        Recommended: 10 * number_of_parameters for better diversity
    max_iterations : int, optional
        Maximum iterations for optimization. Default: 60
    de_tol : float, optional
        Convergence tolerance (0.0 to disable early stopping). Default: 0.001
    de_atol : float, optional
        Absolute convergence tolerance. Default: 1e-8
    bounds : Sequence[Tuple[float, float]] | None, optional
        Parameter bounds for optimization [(min, max), ...] for each parameter.
        If None, uses default bounds of [-2.0, 2.0] for all parameters. Default: None
    random_state : int | None, optional
        Random seed for reproducibility (None for random). Default: None
    verbose : bool, optional
        Print optimization progress. Default: True
    workers : int, optional
        Number of parallel workers for objective function evaluation.
        -1 uses all available CPUs, 1 disables parallelization. Default: -1
    mutation : Union[Tuple[float, float], float], optional
        Mutation constant. If tuple, uses dithering (random value between bounds).
        Lower values (0.5-1.0) are more conservative, better for fine-tuning.
        Higher values (1.0-2.0) provide more exploration. Default: (0.75, 1.5)
    recombination : float, optional
        Recombination constant (crossover probability). Range: 0-1.
        Lower values (0.5-0.6) preserve stable regions better.
        Higher values (0.8-0.9) provide more exploration. Default: 0.7
    strategy : str, optional
        Mutation strategy. Options:
        - 'best1bin': Uses best solution (default, good general-purpose)
        - 'rand1bin': Uses random solution (more exploration)
        - 'best2bin': Uses two difference vectors (good for constrained spaces)
        - 'rand2bin': Most exploration (good for difficult landscapes)
        Default: 'best1bin'
    """

    population: int = 25
    max_iterations: int = 60
    de_tol: float = 0.001
    de_atol: float = 1e-8
    bounds: Sequence[Tuple[float, float]] | None = None
    random_state: int | None = None
    verbose: bool = True
    workers: int = -1
    mutation: Union[Tuple[float, float], float] = (0.75, 1.5)
    recombination: float = 0.7
    strategy: str = "best1bin"


class _ObjectiveFunction:
    """
    Objective function class for controller optimization.

    This class-based approach allows the function to be pickled for multiprocessing.
    """

    def __init__(
        self,
        system_file: str,
        specs: PerformanceSpecs,
        num_parameters: int,
        t_end: float,
        step_amplitude: float,
        cost_weights: CostWeights | None = None,
        dt: float = 0.001,
    ):
        self.system_file = system_file
        self.specs = specs
        self.num_parameters = num_parameters
        self.t_end = t_end
        self.step_amplitude = step_amplitude
        self.cost_weights = cost_weights if cost_weights is not None else CostWeights()
        self.dt = dt

    def __call__(self, param_vec: np.ndarray) -> float:
        try:
            # Simulate system using params directly
            t, y, u, _ = simulate_system(
                params=param_vec,
                system_file=self.system_file,
                t_end=self.t_end,
                step_amplitude=self.step_amplitude,
                dt=self.dt,
            )
            metrics = compute_step_metrics(t, y)
        except (ValueError, Exception) as e:
            # If simulation fails (e.g., unbounded growth), return high penalty
            # Unstable systems will be naturally penalized by poor metrics
            return 1e6

        overshoot_penalty = max(
            0.0, metrics["percent_overshoot"] - self.specs.max_overshoot_pct
        )

        # Penalty for settling time violations
        if not np.isfinite(metrics["settling_time_2pct"]):
            settling_penalty = self.t_end + 10.0 * self.specs.settling_time_2pct
        else:
            settling_penalty = (
                max(0.0, metrics["settling_time_2pct"] - self.specs.settling_time_2pct)
                ** 2
            )

        steady_state_error_penalty = (
            abs(metrics["steady_state"] - self.step_amplitude) ** 2
        )

        # Control signal limit penalty
        if self.specs.max_control_signal is not None:
            control_signal_penalty = (
                max(0.0, np.max(np.abs(u)) - self.specs.max_control_signal) ** 2
            )
        else:
            control_signal_penalty = 0.0

        # Weighted cost: constraints heavily weighted, objectives for fine-tuning
        constraints = (
            self.cost_weights.overshoot_weight * overshoot_penalty
            + self.cost_weights.settling_time_weight * settling_penalty
            + self.cost_weights.steady_state_error_weight * steady_state_error_penalty
            + self.cost_weights.control_signal_limit_weight * control_signal_penalty
        )
        objectives = (
            self.cost_weights.settling_time_weight * metrics["settling_time_2pct"]
        )

        return 100.0 * constraints + objectives


def tune_discrete_controller(
    system_file: str = "system.py",
    specs: PerformanceSpecs | None = None,
    num_parameters: int = 4,
    t_end: float = 5.0,
    step_amplitude: float = 1.0,
    bounds: Sequence[Tuple[float, float]] | None = None,
    popsize: int = 25,
    maxiter: int = 60,
    random_state: int | None = None,
    verbose: bool = True,
    cost_weights: CostWeights | None = None,
    de_tol: float = 0.001,
    de_atol: float = 1e-8,
    dt: float = 0.001,
    workers: int = -1,
    mutation: Union[Tuple[float, float], float] = (0.75, 1.5),
    recombination: float = 0.7,
    strategy: str = "best1bin",
) -> Tuple[np.ndarray, dict]:
    """
    Optimize system parameters to meet the provided performance specs.

    Parameters:
    -----------
    system_file : str, optional
        Path to the system.py file defining the System class (default: "system.py")
    specs : PerformanceSpecs
        Performance requirements (overshoot, settling time, control signal limit)
    num_parameters : int, optional
        Number of optimization parameters. The System class determines how
        these parameters are interpreted. Default: 4
    t_end : float, optional
        Simulation end time in seconds (default: 5.0)
    step_amplitude : float, optional
        Step input amplitude (default: 1.0)
    bounds : sequence of tuples, optional
        Parameter bounds for optimization [(min, max), ...]. If None, uses
        default bounds from optimization_params.
    popsize : int, optional
        Population size for differential evolution (default: 25)
    maxiter : int, optional
        Maximum iterations for differential evolution (default: 60)
    random_state : int, optional
        Random seed for reproducibility
    verbose : bool, optional
        Print optimization progress (default: True)
    cost_weights : CostWeights, optional
        Weights for cost function components. If None, uses default weights.
    de_tol : float, optional
        Convergence tolerance passed to scipy's differential_evolution (default: 0.001)
    de_atol : float, optional
        Absolute convergence tolerance (default: 1e-8)
    dt : float, optional
        Time step for continuous plant simulation (default: 0.001)
    workers : int, optional
        Number of parallel workers for objective function evaluation.
        -1 uses all available CPUs, 1 disables parallelization (default: -1)
    mutation : Union[Tuple[float, float], float], optional
        Mutation constant. If tuple, uses dithering (random value between bounds).
        Lower values (0.5-1.0) are more conservative, better for fine-tuning.
        Higher values (1.0-2.0) provide more exploration. Default: (0.75, 1.5)
    recombination : float, optional
        Recombination constant (crossover probability). Range: 0-1.
        Lower values (0.5-0.6) preserve stable regions better.
        Higher values (0.8-0.9) provide more exploration. Default: 0.7
    strategy : str, optional
        Mutation strategy. Options:
        - 'best1bin': Uses best solution (default, good general-purpose)
        - 'rand1bin': Uses random solution (more exploration)
        - 'best2bin': Uses two difference vectors (good for constrained spaces)
        - 'rand2bin': Most exploration (good for difficult landscapes)
        Default: 'best1bin'

    Returns:
    --------
    params : np.ndarray
        Best parameter vector found
    metrics : dict
        Performance metrics of the best solution
    """
    # Validate required parameters
    if specs is None:
        raise ValueError("specs parameter is required")

    if bounds is None:
        bounds = [(-2.0, 2.0)] * num_parameters

    if len(bounds) != num_parameters:
        raise ValueError(
            f"Bounds must be provided for each parameter. "
            f"Expected {num_parameters} bounds, got {len(bounds)}"
        )

    # Validate bounds format
    for i, bound in enumerate(bounds):
        if not isinstance(bound, (tuple, list)) or len(bound) != 2:
            raise ValueError(
                f"Bound {i} must be a tuple/list of (min, max), got {bound}"
            )
        if bound[0] >= bound[1]:
            raise ValueError(f"Bound {i}: min ({bound[0]}) must be < max ({bound[1]})")

    objective = _ObjectiveFunction(
        system_file=system_file,
        specs=specs,
        num_parameters=num_parameters,
        t_end=t_end,
        step_amplitude=step_amplitude,
        cost_weights=cost_weights,
        dt=dt,
    )

    # Generate initial population with smaller I and D gains if using PID (3 parameters)
    # For PID: params = [Kp, Ki, Kd]
    if num_parameters == 3:
        # Initialize with smaller Ki and Kd for better stability
        init_pop = []
        for _ in range(popsize):
            individual = []
            # Kp: use full bounds
            individual.append(np.random.uniform(bounds[0][0], bounds[0][1]))
            # Ki: use smaller range (typically 0.1x the bound range, centered around 0)
            ki_range = min(abs(bounds[1][0]), abs(bounds[1][1])) * 0.1
            individual.append(np.random.uniform(-ki_range, ki_range))
            # Kd: use smaller range (typically 0.1x the bound range, centered around 0)
            kd_range = min(abs(bounds[2][0]), abs(bounds[2][1])) * 0.1
            individual.append(np.random.uniform(-kd_range, kd_range))
            init_pop.append(np.array(individual))
        init_pop = np.array(init_pop)
    else:
        # For other parameter counts, use uniform random initialization
        init_pop = np.array(
            [
                np.array([np.random.uniform(low, high) for low, high in bounds])
                for _ in range(popsize)
            ]
        )

    if verbose:
        print(f"Generating initial population of {popsize} individuals...")
        if num_parameters == 3:
            print("  Using smaller initial ranges for I and D gains (PID controller)")

    result = differential_evolution(
        objective,
        bounds,
        popsize=popsize,
        maxiter=maxiter,
        seed=random_state,
        init=init_pop,
        polish=False,
        updating="deferred",
        disp=verbose,
        tol=de_tol,
        atol=de_atol,
        mutation=mutation,
        recombination=recombination,
        strategy=strategy,
        workers=workers,  # Enable parallel evaluation
    )

    # Use result.x (best parameters found by optimizer)
    best_params = result.x

    # Re-evaluate to get metrics for the best solution
    try:
        t, y, u, _ = simulate_system(
            params=best_params,
            system_file=system_file,
            t_end=t_end,
            step_amplitude=step_amplitude,
            dt=dt,
        )
        metrics = compute_step_metrics(t, y)
    except (ValueError, Exception) as e:
        # If the best solution is unstable, warn and return empty metrics
        if verbose:
            print(f"Warning: Best solution found by optimizer is unstable: {e}")
            print("  This may indicate the optimization bounds are too wide or")
            print("  the system is difficult to stabilize with the given parameters.")
        metrics = {
            "percent_overshoot": float("inf"),
            "settling_time_2pct": float("inf"),
            "steady_state": float("nan"),
            "rise_time": float("inf"),
        }

    if verbose:
        print("Tuning complete:")
        print(f"  Optimizer reported cost: {result.fun:.4f}")
        print(f"  Best parameters: {best_params}")
        print("  Metrics:")
        for key, value in metrics.items():
            if np.isfinite(value):
                print(f"    {key}: {value:.4f}")
            else:
                print(f"    {key}: {value}")

    return best_params, metrics
