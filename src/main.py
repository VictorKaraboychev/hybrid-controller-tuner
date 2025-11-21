"""
Command-line entry point for tuning and simulating the hybrid control loop.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence, Tuple

import numpy as np

from src.response_metrics import compute_step_metrics
from src.plotting_utils import plot_hybrid_response
from src.simulate_system import simulate_system, load_system_module
from src.tune_discrete_controller import (
    CostWeights,
    PerformanceSpecs,
    SystemParameters,
    OptimizationParameters,
    tune_discrete_controller,
)


def save_controller_to_json(
    output_path: Path,
    num: np.ndarray,
    den: np.ndarray,
    metrics: dict,
    num_order: int | None = None,
    den_order: int | None = None,
    plant_tf: Tuple[Sequence[float], Sequence[float]] | None = None,
    sampling_time: float | None = None,
) -> None:
    """Save controller results to a JSON file."""
    result = {
        "controller": {
            "numerator": num.tolist() if isinstance(num, np.ndarray) else list(num),
            "denominator": den.tolist() if isinstance(den, np.ndarray) else list(den),
        },
        "structure": {},
        "metrics": {},
    }

    if num_order is not None:
        result["structure"]["num_order"] = num_order
    if den_order is not None:
        result["structure"]["den_order"] = den_order

    # Convert metrics to JSON-serializable format
    for key, value in metrics.items():
        if np.isfinite(value):
            result["metrics"][key] = float(value)
        else:
            result["metrics"][key] = None

    # Add plant and sampling info if provided
    if plant_tf is not None:
        result["plant"] = {
            "numerator": list(plant_tf[0]),
            "denominator": list(plant_tf[1]),
        }
    if sampling_time is not None:
        result["sampling_time"] = sampling_time

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)


def main():
    # Load parameters from system.py
    system_file = "system.py"
    system_module = load_system_module(system_file)
    
    # Extract performance specs (required)
    if not hasattr(system_module, "specs"):
        raise AttributeError(
            f"system.py must define a 'specs' variable of type PerformanceSpecs. "
            f"Found in {system_file}: {dir(system_module)}"
        )
    specs = system_module.specs
    if not isinstance(specs, PerformanceSpecs):
        raise TypeError(f"specs must be a PerformanceSpecs instance, got {type(specs)}")
    
    # Extract cost weights (optional)
    cost_weights = getattr(system_module, "cost_weights", None)
    if cost_weights is not None and not isinstance(cost_weights, CostWeights):
        raise TypeError(f"cost_weights must be a CostWeights instance, got {type(cost_weights)}")
    
    # Extract system parameters (required)
    if not hasattr(system_module, "system_params"):
        raise AttributeError(
            f"system.py must define a 'system_params' variable of type SystemParameters. "
            f"Found in {system_file}: {dir(system_module)}"
        )
    system_params = system_module.system_params
    if not isinstance(system_params, SystemParameters):
        raise TypeError(f"system_params must be a SystemParameters instance, got {type(system_params)}")
    
    # Extract optimization parameters (optional with defaults)
    optimization_params = getattr(system_module, "optimization_params", None)
    if optimization_params is None:
        optimization_params = OptimizationParameters()  # Use defaults
    elif not isinstance(optimization_params, OptimizationParameters):
        raise TypeError(f"optimization_params must be an OptimizationParameters instance, got {type(optimization_params)}")
    
    # Extract output paths (optional)
    output_json_path = getattr(system_module, "output_json", "output/controller.json")
    save_path = Path(getattr(system_module, "save_path", "output/response.png"))

    # All denominator coefficients except the leading one (fixed at 1.0) come from parameters
    total_params = (system_params.num_order + 1) + system_params.den_order
    bounds = [(-optimization_params.bound_mag, optimization_params.bound_mag)] * total_params

    num, den, metrics = tune_discrete_controller(
        system_file=system_file,
        sampling_time=system_params.sampling_time,
        specs=specs,
        num_order=system_params.num_order,
        den_order=system_params.den_order,
        t_end=system_params.t_end,
        step_amplitude=system_params.step_amplitude,
        bounds=bounds,
        popsize=optimization_params.population,
        maxiter=optimization_params.max_iterations,
        random_state=optimization_params.random_state,
        verbose=optimization_params.verbose,
        de_tol=optimization_params.de_tol,
        de_atol=optimization_params.de_atol,
        cost_weights=cost_weights,
        dt=system_params.dt,
        workers=optimization_params.workers,
        mutation=optimization_params.mutation,
        recombination=optimization_params.recombination,
        strategy=optimization_params.strategy,
    )

    print("\n=== Tuned Controller ===")
    print(f"num = [{','.join(f'{x:.6f}' for x in num)}];")
    print(f"den = [{','.join(f'{x:.6f}' for x in den)}];")
    print("Metrics from tuning run:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Save to JSON
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_controller_to_json(
        output_path,
        num,
        den,
        metrics,
        num_order=system_params.num_order,
        den_order=system_params.den_order,
        sampling_time=system_params.sampling_time,
    )
    print(f"\nController saved to: {output_path}")

    t, y, u, e = simulate_system(
        controller_tf=(num, den),
        system_file=system_file,
        sampling_time=system_params.sampling_time,
        t_end=system_params.t_end,
        step_amplitude=system_params.step_amplitude,
        dt=system_params.dt,
    )
    final_metrics = compute_step_metrics(t, y)

    fig, _ = plot_hybrid_response(
        t,
        y,
        u,
        e,
        final_metrics,
        step_amplitude=system_params.step_amplitude,
        save_path=str(save_path),
        controller_tf=(num, den),
    )
    print(f"\nPlot saved to {save_path}")
