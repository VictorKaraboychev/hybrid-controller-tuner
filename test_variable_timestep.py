"""
Test script to compare fixed vs variable timestep performance.
"""

import numpy as np
import time
from pathlib import Path

from src.simulate_system import simulate_system
from src.response_metrics import compute_metrics
from src.signals import Step
from systems.outer import OuterSystem
from systems.inner import InnerSystem
from systems.full import FullSystem


def test_timestep_performance(
    system_class,
    system_params,
    r_func,
    t_end,
    dt,
    adaptive_tolerance=0.1,
    max_dt=0.1,
    name="System",
):
    """
    Test both fixed and variable timestep modes and compare performance.

    Parameters
    ----------
    system_class : class
        System class to test
    system_params : np.ndarray
        Parameters for the system
    r_func : Signal
        Reference signal function
    t_end : float
        Simulation end time
    dt : float
        Initial timestep
    adaptive_tolerance : float
        Adaptive tolerance for variable mode
    max_dt : float
        Maximum timestep for variable mode
    name : str
        Name of the system for display
    """
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    print(f"Simulation time: {t_end}s, Initial dt: {dt}s")
    print(f"Adaptive tolerance: {adaptive_tolerance}, Max dt: {max_dt}s")

    # Test fixed timestep
    system_fixed = system_class(params=system_params)
    start = time.time()
    results_fixed = simulate_system(system_fixed, r_func, t_end, dt=dt, dt_mode="fixed")
    time_fixed = time.time() - start
    n_fixed = len(results_fixed[0])
    metrics_fixed = compute_metrics(results_fixed)

    # Test variable timestep
    system_var = system_class(params=system_params)
    start = time.time()
    results_var = simulate_system(
        system_var,
        r_func,
        t_end,
        dt=dt,
        dt_mode="variable",
        adaptive_tolerance=adaptive_tolerance,
        max_dt=max_dt,
    )
    time_var = time.time() - start
    n_var = len(results_var[0])
    metrics_var = compute_metrics(results_var)

    # Compare results
    print(f"\n{'Metric':<30} {'Fixed':<15} {'Variable':<15} {'Difference':<15}")
    print(f"{'-'*75}")
    print(
        f"{'Time (ms)':<30} {time_fixed*1000:<15.2f} {time_var*1000:<15.2f} "
        f"{(time_var-time_fixed)*1000:<15.2f}"
    )
    print(f"{'Number of points':<30} {n_fixed:<15} {n_var:<15} " f"{n_var-n_fixed:<15}")
    print(
        f"{'Points/sec':<30} {n_fixed/time_fixed:<15.0f} {n_var/time_var:<15.0f} "
        f"{(n_var/time_var - n_fixed/time_fixed):<15.0f}"
    )

    # Compare metrics
    print(f"\n{'Metric':<30} {'Fixed':<15} {'Variable':<15} {'Difference':<15}")
    print(f"{'-'*75}")
    for key in metrics_fixed.keys():
        val_fixed = metrics_fixed[key]
        val_var = metrics_var[key]
        if np.isfinite(val_fixed) and np.isfinite(val_var):
            diff = abs(val_var - val_fixed)
            print(f"{key:<30} {val_fixed:<15.6f} {val_var:<15.6f} {diff:<15.6f}")
        else:
            print(f"{key:<30} {val_fixed:<15} {val_var:<15} {'N/A':<15}")

    # Calculate speedup
    speedup = time_fixed / time_var if time_var > 0 else 0
    point_reduction = (1 - n_var / n_fixed) * 100 if n_fixed > 0 else 0

    print(f"\n{'Summary':<30}")
    print(f"{'-'*75}")
    print(f"{'Speedup (fixed/variable)':<30} {speedup:<15.2f}x")
    print(f"{'Point reduction':<30} {point_reduction:<15.1f}%")
    print(f"{'Time saved':<30} {(time_fixed-time_var)*1000:<15.2f} ms")

    return {
        "name": name,
        "time_fixed": time_fixed,
        "time_var": time_var,
        "n_fixed": n_fixed,
        "n_var": n_var,
        "speedup": speedup,
        "point_reduction": point_reduction,
        "metrics_fixed": metrics_fixed,
        "metrics_var": metrics_var,
    }


def main():
    """Run comprehensive timestep performance tests."""
    print("=" * 60)
    print("Variable Timestep Performance Test")
    print("=" * 60)

    results = []

    # Test 1: OuterSystem - short simulation
    print("\n" + "=" * 60)
    print("TEST 1: OuterSystem - Short Simulation")
    print("=" * 60)
    result1 = test_timestep_performance(
        OuterSystem,
        np.array([-39.286194, 0.020513, -23.418556, 0.132535]),
        Step(amplitude=0.15),
        t_end=1.0,
        dt=0.001,
        adaptive_tolerance=0.05,
        max_dt=0.01,
        name="OuterSystem (1s)",
    )
    results.append(result1)

    # Test 2: OuterSystem - long simulation
    print("\n" + "=" * 60)
    print("TEST 2: OuterSystem - Long Simulation")
    print("=" * 60)
    result2 = test_timestep_performance(
        OuterSystem,
        np.array([-39.286194, 0.020513, -23.418556, 0.132535]),
        Step(amplitude=0.15),
        t_end=10.0,
        dt=0.001,
        adaptive_tolerance=0.1,
        max_dt=0.1,
        name="OuterSystem (10s)",
    )
    results.append(result2)

    # Test 3: InnerSystem
    print("\n" + "=" * 60)
    print("TEST 3: InnerSystem")
    print("=" * 60)
    result3 = test_timestep_performance(
        InnerSystem,
        np.array([-44.461137, -0.426855, -0.668664, 0.010916]),
        Step(amplitude=1.4),
        t_end=5.0,
        dt=0.001,
        adaptive_tolerance=0.1,
        max_dt=0.05,
        name="InnerSystem (5s)",
    )
    results.append(result3)

    # Test 4: FullSystem
    print("\n" + "=" * 60)
    print("TEST 4: FullSystem")
    print("=" * 60)
    result4 = test_timestep_performance(
        FullSystem,
        np.array(
            [
                -1.920872,
                0.009013,
                -3.117162,
                0.394065,
                -7.892344,
                -0.998973,
                -0.177016,
                0.014990,
            ]
        ),
        Step(amplitude=0.15),
        t_end=5.0,
        dt=0.002,
        adaptive_tolerance=0.1,
        max_dt=0.05,
        name="FullSystem (5s)",
    )
    results.append(result4)

    # Summary
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    print(
        f"{'Test':<30} {'Speedup':<15} {'Point Reduction':<15} {'Time Saved (ms)':<15}"
    )
    print(f"{'-'*75}")

    total_time_fixed = 0
    total_time_var = 0

    for r in results:
        total_time_fixed += r["time_fixed"]
        total_time_var += r["time_var"]
        print(
            f"{r['name']:<30} {r['speedup']:<15.2f}x {r['point_reduction']:<15.1f}% "
            f"{(r['time_fixed']-r['time_var'])*1000:<15.2f}"
        )

    overall_speedup = total_time_fixed / total_time_var if total_time_var > 0 else 0
    total_time_saved = (total_time_fixed - total_time_var) * 1000

    print(f"{'-'*75}")
    print(
        f"{'TOTAL':<30} {overall_speedup:<15.2f}x {'N/A':<15} {total_time_saved:<15.2f}"
    )
    print(
        f"\nTotal time - Fixed: {total_time_fixed*1000:.2f}ms, "
        f"Variable: {total_time_var*1000:.2f}ms"
    )
    print(f"Overall speedup: {overall_speedup:.2f}x")

    # Save results to file
    output_file = "output/timestep_test_results.txt"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        f.write("Variable Timestep Performance Test Results\n")
        f.write("=" * 60 + "\n\n")
        for r in results:
            f.write(f"{r['name']}:\n")
            f.write(f"  Speedup: {r['speedup']:.2f}x\n")
            f.write(f"  Point reduction: {r['point_reduction']:.1f}%\n")
            f.write(f"  Time saved: {(r['time_fixed']-r['time_var'])*1000:.2f}ms\n")
            f.write(f"  Fixed: {r['n_fixed']} points in {r['time_fixed']*1000:.2f}ms\n")
            f.write(
                f"  Variable: {r['n_var']} points in {r['time_var']*1000:.2f}ms\n\n"
            )
        f.write(f"\nOverall speedup: {overall_speedup:.2f}x\n")
        f.write(f"Total time saved: {total_time_saved:.2f}ms\n")

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
