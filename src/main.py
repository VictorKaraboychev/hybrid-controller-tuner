"""
Command-line entry point for tuning and simulating the hybrid control loop.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

from src.response_metrics import compute_step_metrics
from src.plotting_utils import plot_hybrid_response
from src.simulate_hybrid_system import simulate_hybrid_step_response
from src.tune_discrete_controller import (
    PerformanceSpecs,
    tune_discrete_controller,
    tune_discrete_controller_with_order_search,
)


def _parse_coeffs(text: str) -> Sequence[float]:
    return [float(token.strip()) for token in text.split(",") if token.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune a discrete controller and visualize the hybrid closed-loop response."
    )
    parser.add_argument(
        "--plant-num",
        type=str,
        default="-2.936",
        help="Comma-separated plant numerator coefficients (continuous-time).",
    )
    parser.add_argument(
        "--plant-den",
        type=str,
        default="0.031,1,0",
        help="Comma-separated plant denominator coefficients (continuous-time).",
    )
    parser.add_argument(
        "--sampling-time",
        type=float,
        default=0.015,
        help="Controller sampling time (s).",
    )
    parser.add_argument(
        "--max-overshoot",
        type=float,
        default=10.0,
        help="Maximum allowed percent overshoot for tuning.",
    )
    parser.add_argument(
        "--settling-time",
        type=float,
        default=1.5,
        help="Maximum 2%% settling time (s) for tuning.",
    )
    parser.add_argument(
        "--control-signal-weight",
        type=float,
        default=0.0,
        help="Weight for minimizing control signal magnitude in cost function. Higher values prioritize smaller control signals. Default: 0.0 (disabled).",
    )
    parser.add_argument(
        "--num-order",
        type=int,
        default=1,
        help="Discrete controller numerator order (degree). Must be < den-order for strict properness. Ignored if --search-orders is used.",
    )
    parser.add_argument(
        "--den-order",
        type=int,
        default=2,
        help="Discrete controller denominator order (degree). Must be > num-order for strict properness. Ignored if --search-orders is used.",
    )
    parser.add_argument(
        "--search-orders",
        action="store_true",
        help="Search over different combinations of numerator and denominator orders to find the best structure.",
    )
    parser.add_argument(
        "--num-order-min",
        type=int,
        default=1,
        help="Minimum numerator order when using --search-orders. Default: 0",
    )
    parser.add_argument(
        "--num-order-max",
        type=int,
        default=5,
        help="Maximum numerator order when using --search-orders. Default: 3",
    )
    parser.add_argument(
        "--den-order-min",
        type=int,
        default=2,
        help="Minimum denominator order when using --search-orders. Default: 1",
    )
    parser.add_argument(
        "--den-order-max",
        type=int,
        default=6,
        help="Maximum denominator order when using --search-orders. Default: 4",
    )
    parser.add_argument(
        "--t-end", type=float, default=5.0, help="Simulation horizon (s)."
    )
    parser.add_argument("--step", type=float, default=1.0, help="Step amplitude.")
    parser.add_argument(
        "--popsize",
        type=int,
        default=10,
        help="Population size for differential evolution.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=30,
        help="Maximum iterations for differential evolution.",
    )
    parser.add_argument(
        "--bound-mag",
        type=float,
        default=2.0,
        help="Magnitude for symmetric coefficient bounds [-M, M].",
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("hybrid_system_response.png"),
        help="Path to save the annotated plot.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window after tuning.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress optimizer progress output.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()

    plant_tf = (_parse_coeffs(args.plant_num), _parse_coeffs(args.plant_den))
    specs = PerformanceSpecs(
        max_overshoot_pct=args.max_overshoot,
        settling_time_2pct=args.settling_time,
        control_signal_weight=args.control_signal_weight,
    )

    if args.search_orders:
        # Search over different order combinations
        num, den, metrics, best_num_order, best_den_order = (
            tune_discrete_controller_with_order_search(
                plant_tf=plant_tf,
                sampling_time=args.sampling_time,
                specs=specs,
                num_order_range=(args.num_order_min, args.num_order_max),
                den_order_range=(args.den_order_min, args.den_order_max),
                t_end=args.t_end,
                step_amplitude=args.step,
                bounds=None,  # Use default bounds for each combination
                popsize=args.popsize,
                maxiter=args.maxiter,
                random_state=args.random_state,
                verbose=not args.quiet,
            )
        )

        print("\n=== Best Controller Found ===")
        print(f"Structure: num_order={best_num_order}, den_order={best_den_order}")
        print(f"Numerator: {','.join(f'{x:.6f}' for x in num)}")
        print(f"Denominator: {','.join(f'{x:.6f}' for x in den)}")
        print("Metrics from tuning run:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    else:
        # Use fixed orders
        # Validate strict properness requirement
        if args.num_order >= args.den_order:
            import sys

            print(
                f"ERROR: Controller must be strictly proper: --num-order ({args.num_order}) must be < --den-order ({args.den_order}).",
                file=sys.stderr,
            )
            print(
                "This ensures degree(numerator) < degree(denominator) for a causal, implementable controller.",
                file=sys.stderr,
            )
            sys.exit(1)

        total_params = (args.num_order + 1) + args.den_order
        bounds = [(-args.bound_mag, args.bound_mag)] * total_params

        num, den, metrics = tune_discrete_controller(
            plant_tf=plant_tf,
            sampling_time=args.sampling_time,
            specs=specs,
            num_order=args.num_order,
            den_order=args.den_order,
            t_end=args.t_end,
            step_amplitude=args.step,
            bounds=bounds,
            popsize=args.popsize,
            maxiter=args.maxiter,
            random_state=args.random_state,
            verbose=not args.quiet,
        )

        print("\n=== Tuned Controller ===")
        print(f"Numerator: {','.join(f'{x:.6f}' for x in num)}")
        print(f"Denominator: {','.join(f'{x:.6f}' for x in den)}")
        print("Metrics from tuning run:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

    t, y, u, e = simulate_hybrid_step_response(
        controller_tf=(num, den),
        plant_tf=plant_tf,
        sampling_time=args.sampling_time,
        t_end=args.t_end,
        step_amplitude=args.step,
    )
    final_metrics = compute_step_metrics(t, y, reference=args.step)

    fig, _ = plot_hybrid_response(
        t,
        y,
        u,
        e,
        final_metrics,
        step_amplitude=args.step,
        save_path=str(args.save_path),
    )
    print(f"\nPlot saved to {args.save_path}")
    if args.show:
        fig.show()


if __name__ == "__main__":
    main()
