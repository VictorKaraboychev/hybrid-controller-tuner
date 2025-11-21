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
    sampling_time : float
        Sampling time for discrete controller (seconds)
    num_order : int
        Numerator order (degree)
    den_order : int
        Denominator order (degree)
    t_end : float, optional
        Simulation end time (seconds). Default: 5.0
    step_amplitude : float, optional
        Step input amplitude. Default: 1.0
    dt : float, optional
        Time step for continuous plant simulation (seconds). Default: 0.001
    """

    sampling_time: float
    num_order: int
    den_order: int
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
    bound_mag : float, optional
        Magnitude of parameter bounds (symmetric: [-bound_mag, bound_mag]). Default: 2.0
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
    bound_mag: float = 2.0
    random_state: int | None = None
    verbose: bool = True
    workers: int = -1
    mutation: Union[Tuple[float, float], float] = (0.75, 1.5)
    recombination: float = 0.7
    strategy: str = "best1bin"


def params_to_discrete_tf(
    params: np.ndarray, num_order: int, den_order: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map optimization parameters to a discrete-time transfer function.

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
        If den_order < 0
        If incorrect number of parameters provided
    """
    if den_order < 0:
        raise ValueError(
            f"den_order must be >= 0, got {den_order}"
        )

    num_coeffs = num_order + 1
    # All denominator coefficients except the leading one (fixed at 1.0) come from parameters
    expected_params = num_coeffs + den_order

    if len(params) != expected_params:
        raise ValueError(
            f"Expected {expected_params} params, received {len(params)}"
        )

    num = np.asarray(params[:num_coeffs], dtype=float)
    
    # Get all denominator coefficients except the leading one from parameters
    den_rest = np.asarray(params[num_coeffs:], dtype=float)
    # Leading coefficient is fixed at 1.0
    den = np.concatenate(([1.0], den_rest))

    return num, den


def _generate_stable_initial_individual(
    bounds: Sequence[Tuple[float, float]],
    num_order: int,
    den_order: int,
    max_attempts: int = 1000,
) -> np.ndarray:
    """
    Generate a single initial individual with poles strictly inside the open unit disk.
    
    All poles must be strictly inside the unit circle (|z| < 1).
    
    Parameters
    ----------
    bounds : Sequence[Tuple[float, float]]
        Parameter bounds for each parameter
    num_order : int
        Numerator order
    den_order : int
        Denominator order
    max_attempts : int
        Maximum number of attempts to generate a valid individual
        
    Returns
    -------
    np.ndarray
        Valid parameter vector with stable poles
        
    Raises
    ------
    RuntimeError
        If unable to generate a valid individual after max_attempts
    """
    for attempt in range(max_attempts):
        params = np.array([np.random.uniform(low, high) for low, high in bounds])
        
        try:
            # Convert to transfer function
            num, den = params_to_discrete_tf(params, num_order, den_order)
            
            # Find roots of denominator (controller poles)
            controller_poles = np.roots(den)
            
            # Check pole stability - all poles must be strictly inside unit disk (|z| < 1)
            if np.all(np.abs(controller_poles) < 1.0 - 1e-10):
                return params
                
        except Exception:
            # If conversion fails, try again
            continue
    
    # If we get here, couldn't generate a valid individual
    raise RuntimeError(
        f"Failed to generate a stable initial individual after {max_attempts} attempts. "
        f"Consider widening bounds or adjusting controller order."
    )






class _ObjectiveFunction:
    """
    Objective function class for controller optimization.
    
    This class-based approach allows the function to be pickled for multiprocessing.
    """
    
    def __init__(
        self,
        system_file: str,
        sampling_time: float,
        specs: PerformanceSpecs,
        num_order: int,
        den_order: int,
        t_end: float,
        step_amplitude: float,
        cost_weights: CostWeights | None = None,
        dt: float = 0.001,
    ):
        self.system_file = system_file
        self.sampling_time = sampling_time
        self.specs = specs
        self.num_order = num_order
        self.den_order = den_order
        self.t_end = t_end
        self.step_amplitude = step_amplitude
        self.cost_weights = cost_weights if cost_weights is not None else CostWeights()
        self.dt = dt
    
    def __call__(self, param_vec: np.ndarray) -> float:
        try:
            controller_tf = params_to_discrete_tf(param_vec, self.num_order, self.den_order)
            
            # Check controller stability
            _, den = controller_tf
            controller_poles = np.roots(den)
            if np.any(np.abs(controller_poles) > 1.0 + 1e-10):
                return 1e6
            
            # Penalize very large coefficients (numerical issues)
            if len(den) > 1:
                max_coeff = np.max(np.abs(den))
                if max_coeff > 50.0:
                    return 1e6 + max_coeff * 100
            
            t, y, u, _ = simulate_system(
                controller_tf=controller_tf,
                system_file=self.system_file,
                sampling_time=self.sampling_time,
                t_end=self.t_end,
                step_amplitude=self.step_amplitude,
                dt=self.dt,
            )
            metrics = compute_step_metrics(t, y)
        except Exception:
            # Penalize unstable or failed simulations
            return 1e6

        overshoot_penalty = max(0.0, metrics["percent_overshoot"] - self.specs.max_overshoot_pct)

        # Penalty for settling time violations
        if not np.isfinite(metrics["settling_time_2pct"]):
            settling_penalty = self.t_end + 10.0 * self.specs.settling_time_2pct
        else:
            settling_penalty = max(0.0, metrics["settling_time_2pct"] - self.specs.settling_time_2pct) ** 2

        steady_state_error_penalty = abs(metrics["steady_state"] - self.step_amplitude) ** 2

        # Control signal limit penalty
        if self.specs.max_control_signal is not None:
            control_signal_penalty = max(0.0, np.max(np.abs(u)) - self.specs.max_control_signal) ** 2
        else:
            control_signal_penalty = 0.0

        # Weighted cost: constraints heavily weighted, objectives for fine-tuning
        constraints = (
            self.cost_weights.overshoot_weight * overshoot_penalty
            + self.cost_weights.settling_time_weight * settling_penalty
            + self.cost_weights.steady_state_error_weight * steady_state_error_penalty
            + self.cost_weights.control_signal_limit_weight * control_signal_penalty
        )
        objectives = self.cost_weights.settling_time_weight * metrics["settling_time_2pct"]
        
        return 100.0 * constraints + objectives


def tune_discrete_controller(
    system_file: str = "system.py",
    sampling_time: float = 0.1,
    specs: PerformanceSpecs | None = None,
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
    de_tol: float = 0.001,
    de_atol: float = 1e-8,
    dt: float = 0.001,
    workers: int = -1,
    mutation: Union[Tuple[float, float], float] = (0.75, 1.5),
    recombination: float = 0.7,
    strategy: str = "best1bin",
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Search for discrete controller coefficients meeting the provided specs.

    Parameters:
    -----------
    system_file : str, optional
        Path to the system.py file defining the plant (default: "system.py")
    sampling_time : float, optional
        Sampling time in seconds (default: 0.1)
    specs : PerformanceSpecs
        Performance requirements (overshoot, settling time, control signal limit)
    num_order : int, optional
        Numerator order (degree). The numerator will have num_order + 1 coefficients.
        Default: 2 (2nd order numerator = 3 coefficients)
    den_order : int, optional
        Denominator order (degree). The denominator will have den_order + 1 coefficients
        (leading coefficient fixed at 1.0).
        Default: 2 (2nd order denominator = 3 coefficients)
    t_end : float, optional
        Simulation end time in seconds (default: 5.0)
    step_amplitude : float, optional
        Step input amplitude (default: 1.0)
    bounds : sequence of tuples, optional
        Parameter bounds for optimization [(min, max), ...]
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
    num : np.ndarray
        Best numerator coefficients found
    den : np.ndarray
        Best denominator coefficients found
    metrics : dict
        Performance metrics of the best controller
    """
    # Validate required parameters
    if specs is None:
        raise ValueError("specs parameter is required")

    total_params = (num_order + 1) + den_order
    
    if bounds is None:
        bounds = [(-2.0, 2.0)] * total_params

    if len(bounds) != total_params:
        raise ValueError(
            f"Bounds must be provided for each parameter. "
            f"Expected {total_params} bounds, got {len(bounds)}"
        )

    objective = _ObjectiveFunction(
        system_file=system_file,
        sampling_time=sampling_time,
        specs=specs,
        num_order=num_order,
        den_order=den_order,
        t_end=t_end,
        step_amplitude=step_amplitude,
        cost_weights=cost_weights,
        dt=dt,
    )

    # Generate initial population with stable poles
    if verbose:
        print(f"Generating initial population of {popsize} controllers with stable poles...")
    
    init_pop = np.array([
        _generate_stable_initial_individual(bounds, num_order, den_order)
        for _ in range(popsize)
    ])
    
    if verbose:
        print("Initial population generated successfully.")
    
    result = differential_evolution(
        objective,
        bounds,
        popsize=popsize,
        maxiter=maxiter,
        seed=random_state,
        init=init_pop,  # Use our stable initial population
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

    # Use result.x (best parameters found by optimizer) and verify it's stable
    best_params = result.x
    
    # Verify the best solution is stable
    controller_tf = params_to_discrete_tf(best_params, num_order, den_order)
    _, den = controller_tf
    controller_poles = np.roots(den)
    
    if np.any(np.abs(controller_poles) > 1.0 + 1e-10):
        raise RuntimeError(
            "Optimizer found an unstable solution. Consider adjusting bounds or specs."
        )
    
    # Re-evaluate to get metrics for the best solution
    t, y, u, _ = simulate_system(
        controller_tf=controller_tf,
        system_file=system_file,
        sampling_time=sampling_time,
        t_end=t_end,
        step_amplitude=step_amplitude,
        dt=dt,
    )
    metrics = compute_step_metrics(t, y)
    
    best_num, best_den = controller_tf

    if verbose:
        print("Tuning complete:")
        print(f"  Optimizer reported cost: {result.fun:.4f}")
        print(f"  Controller numerator: {','.join(f'{x:.6f}' for x in best_num)}")
        print(f"  Controller denominator: {','.join(f'{x:.6f}' for x in best_den)}")
        print("  Metrics:")
        for key, value in metrics.items():
            print(f"    {key}: {value:.4f}")

    return best_num, best_den, metrics
