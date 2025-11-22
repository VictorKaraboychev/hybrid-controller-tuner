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


@dataclass(frozen=True)
class OptimizationParameters:
    """
    Optimization algorithm parameters.

    Attributes
    ----------
    num_parameters : int
        Number of optimization parameters. The System class determines
        how these parameters are interpreted.
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

    num_parameters: int
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

    def __init__(self, system_class):
        """
        Initialize objective function with system class.

        Parameters
        ----------
        system_class
            System class to instantiate for each evaluation
        """
        self.system_class = system_class

    def __call__(self, param_vec: np.ndarray) -> float:
        """
        Evaluate cost for given parameters.

        Parameters
        ----------
        param_vec : np.ndarray
            Controller parameters

        Returns
        -------
        float
            Cost value
        """
        # Create a system instance with the given parameters
        system = self.system_class(params=param_vec)
        # Evaluate cost using the system's own method
        return system.cost()


def optimize(
    system_class,
    optimization_params: OptimizationParameters,
) -> np.ndarray:
    """
    Optimize system parameters using differential evolution.

    Parameters
    ----------
    system_class
        System class to optimize (must have __init__(params) and cost() methods)
    optimization_params : OptimizationParameters
        Optimization algorithm parameters

    Returns
    -------
    np.ndarray
        Optimized controller parameters
    """
    # Get bounds from optimization_params
    bounds = optimization_params.bounds
    if bounds is None:
        bounds = [(-2.0, 2.0)] * optimization_params.num_parameters

    if len(bounds) != optimization_params.num_parameters:
        raise ValueError(
            f"Bounds must have {optimization_params.num_parameters} entries, got {len(bounds)}"
        )

    # Validate bounds format
    for i, bound in enumerate(bounds):
        if not isinstance(bound, (tuple, list)) or len(bound) != 2:
            raise ValueError(
                f"Bound {i} must be a tuple/list of (min, max), got {bound}"
            )
        if bound[0] >= bound[1]:
            raise ValueError(f"Bound {i}: min ({bound[0]}) must be < max ({bound[1]})")

    # Create objective function
    objective = _ObjectiveFunction(system_class)

    # Run optimization
    result = differential_evolution(
        objective,
        bounds,
        popsize=optimization_params.population,
        maxiter=optimization_params.max_iterations,
        seed=optimization_params.random_state,
        polish=False,
        updating="deferred",
        disp=optimization_params.verbose,
        tol=optimization_params.de_tol,
        atol=optimization_params.de_atol,
        mutation=optimization_params.mutation,
        recombination=optimization_params.recombination,
        strategy=optimization_params.strategy,
        workers=optimization_params.workers,
    )

    if optimization_params.verbose:
        print(f"Optimization complete. Cost: {result.fun:.4f}")

    return result.x
