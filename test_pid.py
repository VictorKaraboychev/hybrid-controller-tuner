"""
Test script for PID block with Kp=0.1, Ki=0, Kd=0
"""

import numpy as np
from pathlib import Path

from src.simulate_system import simulate_system
from src.response_metrics import compute_metrics
from src.signals import Step

from src.plotting_utils import plot_response
import matplotlib.pyplot as plt

from systems.full import FullSystem

# PID parameters: [Kp, Ki, Kd]
params = np.array([-1.920872,0.009013,-3.117162,0.394065,-7.892344,-0.998973,-0.177016,0.014990])

print("Testing PID controller with:")
print(f"  Kp = {params[0]}")
print(f"  Ki = {params[1]}")
print(f"  Kd = {params[2]}")
print(f"  Sampling time = {params[3]}")
print()

# Simulation parameters
t_end = 10.0
step_amplitude = 0.15

system = FullSystem(params=params)

# Run simulation
print("Running simulation...")
r_func = Step(amplitude=0.15)
results = simulate_system(system, r_func, t_end)

# Compute metrics
metrics_outer = compute_metrics(results)
# For inner system, need to create modified results with u_outer as reference
# results is (t, r, y_outer, e_outer, u_outer, y_inner, e_inner, u_inner)
# For inner metrics, we want (t, u_outer, y_inner)
inner_results = (results[0], results[4], results[5])  # (t, u_outer, y_inner)
metrics_inner = compute_metrics(inner_results)

print("\n=== Final Metrics Outer System ===")
for key, value in metrics_outer.items():
    if np.isfinite(value):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")
print()

print("\n=== Final Metrics Inner System ===")
for key, value in metrics_inner.items():
    if np.isfinite(value):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")
print()

# Plot results
print("Generating plot...")
save_path = "output/pid_test_response.png"
Path(save_path).parent.mkdir(parents=True, exist_ok=True)

fig, axes = plot_response(
    results,
    metrics_outer,
    save_path=save_path,
)

# Add title with parameters
params_str = ', '.join(f'{x:.6f}' for x in params)
fig.suptitle(f"[{params_str}]", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {save_path}")
print("\nDisplaying plot...")
plt.show()
