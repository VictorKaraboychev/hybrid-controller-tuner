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
from src.simulate_hybrid_system import simulate_hybrid_step_response
from src.tune_discrete_controller import (
    CostWeights,
    PerformanceSpecs,
    tune_discrete_controller,
    tune_discrete_controller_with_order_search,
)


def load_specs_from_json(json_path: Path) -> dict[str, Any]:
    """Load tuning specifications from a JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


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
    # Always read from specs.json
    specs_path = Path("specs.json")
    if not specs_path.exists():
        raise FileNotFoundError(
            f"Specs file not found: {specs_path}. Please create specs.json with tuning specifications."
        )

    json_data = load_specs_from_json(specs_path)

    # Extract plant transfer function
    plant_tf = (
        json_data["plant"]["numerator"],
        json_data["plant"]["denominator"],
    )

    # Extract performance specs
    specs = PerformanceSpecs(
        max_overshoot_pct=json_data["specs"]["max_overshoot_pct"],
        settling_time_2pct=json_data["specs"]["settling_time_2pct"],
        max_control_signal=json_data["specs"].get("max_control_signal", None),
    )

    # Extract cost weights (optional)
    cost_weights = None
    if "cost_weights" in json_data:
        weights_data = json_data["cost_weights"]
        cost_weights = CostWeights(
            overshoot_weight=weights_data.get("overshoot_weight", 1.0),
            settling_time_weight=weights_data.get("settling_time_weight", 2.0),
            steady_state_error_weight=weights_data.get("steady_state_error_weight", 3.0),
            control_signal_limit_weight=weights_data.get("control_signal_limit_weight", 1.0),
        )

    # Extract tuning parameters
    sampling_time = json_data["sampling_time"]
    t_end = json_data.get("t_end", 5.0)
    step_amplitude = json_data.get("step_amplitude", 1.0)
    popsize = json_data.get("popsize", 25)
    maxiter = json_data.get("maxiter", 60)
    random_state = json_data.get("random_state", None)
    bound_mag = json_data.get("bound_mag", 2.0)
    search_orders = json_data.get("search_orders", False)
    num_order = json_data.get("num_order", 1)
    den_order = json_data.get("den_order", 2)
    num_order_min = json_data.get("num_order_min", 1)
    num_order_max = json_data.get("num_order_max", 5)
    den_order_min = json_data.get("den_order_min", 2)
    den_order_max = json_data.get("den_order_max", 6)
    verbose = not json_data.get("quiet", False)
    output_json_path = json_data.get("output_json", "data/controller.json")
    save_path = Path(json_data.get("save_path", "hybrid_system_response.png"))
    show_plot = json_data.get("show", False)

    if search_orders:
        # Search over different order combinations
        num, den, metrics, best_num_order, best_den_order = (
            tune_discrete_controller_with_order_search(
                plant_tf=plant_tf,
                sampling_time=sampling_time,
                specs=specs,
                num_order_range=(num_order_min, num_order_max),
                den_order_range=(den_order_min, den_order_max),
                t_end=t_end,
                step_amplitude=step_amplitude,
                bounds=None,  # Use default bounds for each combination
                popsize=popsize,
                maxiter=maxiter,
                random_state=random_state,
                verbose=verbose,
                cost_weights=cost_weights,
            )
        )

        print("\n=== Best Controller Found ===")
        print(f"Structure: num_order={best_num_order}, den_order={best_den_order}")
        print(f"Numerator: {','.join(f'{x:.6f}' for x in num)}")
        print(f"Denominator: {','.join(f'{x:.6f}' for x in den)}")
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
            num_order=best_num_order,
            den_order=best_den_order,
            plant_tf=plant_tf,
            sampling_time=sampling_time,
        )
        print(f"\nController saved to: {output_path}")
    else:
        # Use fixed orders
        # Validate strict properness requirement
        if num_order >= den_order:
            import sys

            print(
                f"ERROR: Controller must be strictly proper: num_order ({num_order}) must be < den_order ({den_order}).",
                file=sys.stderr,
            )
            print(
                "This ensures degree(numerator) < degree(denominator) for a causal, implementable controller.",
                file=sys.stderr,
            )
            sys.exit(1)

        total_params = (num_order + 1) + den_order
        bounds = [(-bound_mag, bound_mag)] * total_params

        num, den, metrics = tune_discrete_controller(
            plant_tf=plant_tf,
            sampling_time=sampling_time,
            specs=specs,
            num_order=num_order,
            den_order=den_order,
            t_end=t_end,
            step_amplitude=step_amplitude,
            bounds=bounds,
            popsize=popsize,
            maxiter=maxiter,
            random_state=random_state,
            verbose=verbose,
            cost_weights=cost_weights,
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
            num_order=num_order,
            den_order=den_order,
            plant_tf=plant_tf,
            sampling_time=sampling_time,
        )
        print(f"\nController saved to: {output_path}")

    t, y, u, e = simulate_hybrid_step_response(
        controller_tf=(num, den),
        plant_tf=plant_tf,
        sampling_time=sampling_time,
        t_end=t_end,
        step_amplitude=step_amplitude,
    )
    final_metrics = compute_step_metrics(t, y, reference=step_amplitude)

    fig, _ = plot_hybrid_response(
        t,
        y,
        u,
        e,
        final_metrics,
        step_amplitude=step_amplitude,
        save_path=str(save_path),
    )
    print(f"\nPlot saved to {save_path}")
    if show_plot:
        fig.show()
