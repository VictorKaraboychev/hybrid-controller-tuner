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
        Numerator order (degree) - must be < den_order
    den_order : int
        Denominator order (degree) - must be > num_order
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
    Map optimization parameters to a strictly proper discrete-time transfer function.

    A strictly proper transfer function has degree(numerator) < degree(denominator).
    This ensures the controller is causal and implementable.
    
    Integral action is always enforced by ensuring the controller has a pole at z=1.
    This eliminates steady-state error from constant disturbances. The last denominator
    coefficient is computed as: a_n = -1 - sum(a_1 to a_{n-1}).

    Parameters:
    -----------
    params : np.ndarray
        Optimization parameters: [num_coeffs..., den_coeffs...]
        The last denominator coefficient is computed to enforce a pole at z=1,
        so one fewer parameter is needed.
    num_order : int
        Numerator order (degree). Resulting numerator has num_order + 1 coefficients.
    den_order : int
        Denominator order (degree). Resulting denominator has den_order + 1 coefficients.
        The leading denominator coefficient is fixed at 1.0, and the last is computed
        to enforce integral action.

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
        If den_order < 1 (need at least one denominator coefficient for integral action)
        If incorrect number of parameters provided
    """
    # Enforce strict properness: degree(num) < degree(den)
    if num_order >= den_order:
        raise ValueError(
            f"Controller must be strictly proper: num_order ({num_order}) must be < den_order ({den_order}). "
            f"Current: degree(num) = {num_order}, degree(den) = {den_order}"
        )

    if den_order < 1:
        raise ValueError(
            f"Integral action requires den_order >= 1, got {den_order}"
        )

    num_coeffs = num_order + 1
    # Last denominator coefficient is computed for integral action, so one fewer parameter
    expected_params = num_coeffs + den_order - 1

    if len(params) != expected_params:
        raise ValueError(
            f"Expected {expected_params} params, received {len(params)}"
        )

    num = np.asarray(params[:num_coeffs], dtype=float)
    
    # Get all but the last denominator coefficient from parameters
    den_rest = np.asarray(params[num_coeffs:], dtype=float)
    # Enforce pole at z=1: D(1) = 1 + a1 + a2 + ... + an = 0
    # So: an = -1 - (a1 + a2 + ... + a_{n-1})
    last_coeff = -1.0 - np.sum(den_rest)
    den = np.concatenate(([1.0], den_rest, [last_coeff]))

    return num, den


def _generate_stable_initial_individual(
    bounds: Sequence[Tuple[float, float]],
    num_order: int,
    den_order: int,
    max_attempts: int = 1000,
) -> np.ndarray:
    """
    Generate a single initial individual with poles strictly inside the open unit disk.
    
    The integrator pole at z=1 is allowed (enforced by params_to_discrete_tf).
    All other poles must be strictly inside the unit circle (|z| < 1).
    
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
        # Generate random parameters within bounds
        params = np.array([
            np.random.uniform(low, high) for low, high in bounds
        ])
        
        try:
            # Convert to transfer function
            num, den = params_to_discrete_tf(params, num_order, den_order)
            
            # Find roots of denominator (controller poles)
            controller_poles = np.roots(den)
            
            # Check pole stability
            # Allow pole at z=1 (integrator) with tolerance
            # All other poles must be strictly inside unit disk (|z| < 1)
            integrator_tolerance = 1e-6
            valid = True
            
            for pole in controller_poles:
                pole_magnitude = np.abs(pole)
                # Check if this is the integrator pole (z=1)
                is_integrator = abs(pole_magnitude - 1.0) < integrator_tolerance
                
                if is_integrator:
                    # Integrator pole is allowed
                    continue
                elif pole_magnitude >= 1.0 - 1e-10:
                    # Pole on or outside unit circle (not allowed)
                    valid = False
                    break
            
            if valid:
                return params
                
        except Exception:
            # If conversion fails, try again
            continue
    
    # If we get here, couldn't generate a valid individual
    raise RuntimeError(
        f"Failed to generate a stable initial individual after {max_attempts} attempts. "
        f"Consider widening bounds or adjusting controller order."
    )


def _make_stable_initial_population(
    bounds: Sequence[Tuple[float, float]],
    popsize: int,
    num_order: int,
    den_order: int,
) -> np.ndarray:
    """
    Generate an initial population where all individuals have poles strictly inside the open unit disk.
    
    Parameters
    ----------
    bounds : Sequence[Tuple[float, float]]
        Parameter bounds for each parameter
    popsize : int
        Population size
    num_order : int
        Numerator order
    den_order : int
        Denominator order
        
    Returns
    -------
    np.ndarray
        Initial population matrix of shape (popsize, n_params)
    """
    population = []
    for i in range(popsize):
        individual = _generate_stable_initial_individual(
            bounds, num_order, den_order
        )
        population.append(individual)
    
    return np.array(population)


def _make_default_bounds(
    total_params: int, magnitude: float = 2.0
) -> Sequence[Tuple[float, float]]:
    """
    Helper to construct symmetric parameter bounds.
    """

    return [(-magnitude, magnitude)] * total_params


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
            
            # Check controller stability by examining its poles
            _, den = controller_tf
            # Find roots of denominator (controller poles)
            controller_poles = np.roots(den)
            # Check if any pole is outside the unit circle (unstable)
            if np.any(np.abs(controller_poles) > 1.0 + 1e-10):
                # Penalize unstable controllers
                return 1e6
            
            # Check if the computed last coefficient (for integral action) is reasonable
            # Large coefficients can lead to numerical issues
            if len(den) > 1:
                last_coeff = den[-1]
                # Penalize if the computed coefficient is very large (likely problematic)
                if abs(last_coeff) > 50.0:
                    return 1e6 + abs(last_coeff) * 100
            
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

        overshoot_penalty = max(
            0.0, metrics["percent_overshoot"] - self.specs.max_overshoot_pct
        )

        # Stronger penalty for settling time violations
        settling_penalty = 0.0
        if not np.isfinite(metrics["settling_time_2pct"]):
            # If doesn't settle, use a large penalty based on simulation time
            # This encourages the optimizer to find solutions that actually settle
            settling_penalty = self.t_end + 10.0 * self.specs.settling_time_2pct
        else:
            # Use squared penalty to make violations more expensive
            violation = max(
                0.0, metrics["settling_time_2pct"] - self.specs.settling_time_2pct
            )
            settling_penalty = violation**2

        # Encourage steady-state accuracy
        steady_state_error = abs(metrics["steady_state"] - self.step_amplitude)
        steady_state_error_penalty = steady_state_error**2

        # Control signal limit penalty (only penalize if limit is exceeded)
        control_signal_penalty = 0.0
        if self.specs.max_control_signal is not None:
            max_control_magnitude = np.max(np.abs(u))
            
            violation = max(0.0, max_control_magnitude - self.specs.max_control_signal)
            control_signal_penalty = violation**2  # Squared penalty for violations

        # Weighted sum cost
        constraints = (
            self.cost_weights.overshoot_weight * overshoot_penalty
            + self.cost_weights.settling_time_weight * settling_penalty
            + self.cost_weights.steady_state_error_weight * steady_state_error_penalty
            + self.cost_weights.control_signal_limit_weight * control_signal_penalty
        )
        
        objectives = (
            self.cost_weights.settling_time_weight * metrics["settling_time_2pct"]
        )
        
        # Solve the constraints before the objectives become important
        cost = 100.0 * constraints + objectives

        return cost


def _make_objective_function(
    system_file: str,
    sampling_time: float,
    specs: PerformanceSpecs,
    num_order: int,
    den_order: int,
    t_end: float,
    step_amplitude: float,
    cost_weights: CostWeights | None = None,
    dt: float = 0.001,
) -> _ObjectiveFunction:
    """
    Create an objective function for controller optimization.

    Parameters
    ----------
    system_file : str
        Path to the system.py file defining the plant
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
    cost_weights : CostWeights, optional
        Weights for cost function components. If None, uses default weights.
    dt : float, optional
        Time step for continuous plant simulation (default: 0.001)

    Returns
    -------
    _ObjectiveFunction
        Objective function object that can be pickled for multiprocessing
    """
    return _ObjectiveFunction(
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

    The controller is constrained to be strictly proper: degree(numerator) < degree(denominator).
    This ensures the controller is causal and physically realizable.

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
        (leading coefficient fixed at 1.0). Must be > num_order for strict properness.
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

    Raises:
    ------
    ValueError
        If num_order >= den_order (controller would not be strictly proper)
    """
    # Validate required parameters
    if specs is None:
        raise ValueError("specs parameter is required")
    
    # Validate strict properness requirement
    if num_order >= den_order:
        raise ValueError(
            f"Controller must be strictly proper: num_order ({num_order}) must be < den_order ({den_order}). "
            f"This ensures degree(numerator) < degree(denominator)."
        )

    num_params = num_order + 1
    # Last denominator coefficient is computed for integral action, so one fewer parameter
    total_params = num_params + den_order - 1
    
    if bounds is None:
        bounds = _make_default_bounds(total_params)

    if len(bounds) != total_params:
        raise ValueError(
            f"Bounds must be provided for each parameter. "
            f"Expected {total_params} bounds, got {len(bounds)}"
        )

    objective = _make_objective_function(
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
    
    init_pop = _make_stable_initial_population(
        bounds=bounds,
        popsize=popsize,
        num_order=num_order,
        den_order=den_order,
    )
    
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
    
    # Verify the best solution is stable by checking poles
    try:
        controller_tf = params_to_discrete_tf(best_params, num_order, den_order)
        _, den = controller_tf
        controller_poles = np.roots(den)
        
        # Check if any pole is outside the unit circle (unstable)
        if np.any(np.abs(controller_poles) > 1.0 + 1e-10):
            raise RuntimeError(
                "Optimizer found an unstable solution. This may indicate the optimization "
                "converged to an unstable region. Consider adjusting bounds or specs."
            )
    except Exception as e:
        if isinstance(e, RuntimeError):
            raise
        raise RuntimeError(
            f"Failed to find a stable controller. Error: {e}. "
            "Consider widening bounds, relaxing specs, or adjusting controller order."
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
