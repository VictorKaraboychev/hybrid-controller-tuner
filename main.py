"""
Command-line entry point for tuning and simulating the hybrid control loop.
"""

from __future__ import annotations

import numpy as np

from src.response_metrics import compute_metrics
from src.plotting_utils import plot_response
from src.simulate_system import simulate_system
from src.tune_discrete_controller import OptimizationParameters, optimize
from src.signals import Step
from systems.inner import InnerSystem
from systems.outer import OuterSystem
from systems.full import FullSystem

# Optimization parameters
optimization_params = OptimizationParameters(
    num_parameters=4,  # Total number of optimization parameters
    population=25,  # Population size for differential evolution
    max_iterations=400,  # Maximum iterations for optimization
    de_tol=0.001,  # Convergence tolerance (0.0 to disable early stopping)
    bounds=[
        (-25.0, 25.0),  # Kp bounds (outer)
        (-1.0, 1.0),  # Ki bounds (outer)
        (-10.0, 10.0),  # Kd bounds (outer)
        (0.01, 1.0),  # Sampling time bounds (outer)
        # (-100.0, 100.0),  # Kp bounds (inner)
        # (-5.0, 5.0),  # Ki bounds (inner)
        # (-100.0, 100.0),  # Kd bounds (inner)
        # (0.005, 0.05),  # Sampling time bounds (inner)
    ],
    random_state=None,  # Random seed for reproducibility (None for random)
    verbose=True,  # Print optimization progress
    workers=-1,  # Use all available CPUs for parallel evaluation
)

# Output paths
save_path = "output/outer_response.png"  # Path to save response plot


def main():
    System = OuterSystem

    # Optimize the system
    params = optimize(System, optimization_params)

    print("\n=== Optimized Parameters ===")
    print(f"params = [{','.join(f'{x:.6f}' for x in params)}]")

    # Create system with optimized parameters and simulate
    results = simulate_system(System(params=params), Step(amplitude=0.15), 10.0, 0.0001)
    final_metrics = compute_metrics(results)

    print("\n=== Final Metrics ===")
    for key, value in final_metrics.items():
        if np.isfinite(value):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Plot results
    fig, _ = plot_response(results, final_metrics, save_path=str(save_path))
    print(f"\nPlot saved to {save_path}")


if __name__ == "__main__":
    main()
