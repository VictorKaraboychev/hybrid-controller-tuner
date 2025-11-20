"""
User-defined control system.

This file defines the control system using the system building blocks.
The `system` function is called by the tuner with the controller, reference `r`,
and time `t`, and returns error `e`, control signal `u`, and new output `y`.

The function signature is:
    def system(controller, r, t):
        ...
        return e, u, y

Where:
    controller: DiscreteTF controller block
    r: Reference/input signal at current time step
    t: Current time (seconds)
    e: Error signal (typically r - p.y)
    u: Control signal
    y: New output signal

Note: Use p.y (or any transfer function's .y attribute) to access the last output.
"""

import numpy as np
from src.system_blocks import ContinuousTF, DiscreteTF, Saturation
from src.tune_discrete_controller import (
    PerformanceSpecs,
    CostWeights,
    SystemParameters,
    OptimizationParameters,
)

SIMULATION_DT = 0.001

# ============================================================================
# Define your plant blocks here
# ============================================================================

p = ContinuousTF(
    num=[-2.936],
    den=[0.031, 1.0, 0.0],
    dt=SIMULATION_DT  # Integration time step (should match simulation dt)
)

s = Saturation(min_val=-6.0, max_val=6.0)


# ============================================================================
# System function
# ============================================================================

def system(d_controller, r, t):
    """
    Control system function.
    
    This function is called by the tuner at each discrete time step with the controller,
    reference signal, and current time. It computes the error, control signal, and new output.
    
    Parameters
    ----------
    controller : DiscreteTF
        Discrete-time controller block
    r : float
        Reference/input signal at current time step
    t : float
        Current time (seconds)
        
    Returns
    -------
    e : float
        Error signal
    u : float
        Control signal
    y : float
        New output signal
    """
    e = r - p.y
    u = d_controller.step(t, e)
    u = s.step(u)
    y = p.step(u)
    
    return e, u, y


def reset():
    """
    Reset all blocks to their initial state.
    
    This should be called at the start of a new simulation.
    """
    p.reset()
    s.reset()


# ============================================================================
# Tuning Parameters and Specifications
# ============================================================================

# Performance specifications for controller tuning (required)
specs = PerformanceSpecs(
    max_overshoot_pct=5.0,      # Maximum allowed percent overshoot
    settling_time_2pct=0.25,     # Required 2% settling time (seconds)
    max_control_signal=6.0,      # Maximum allowed control signal (None for no limit)
)

# Cost function weights (optional - uses defaults if None)
cost_weights = CostWeights(
    overshoot_weight=1.0,
    settling_time_weight=2.0,
    steady_state_error_weight=3.0,
    control_signal_limit_weight=2.0,
)

# System parameters (required)
system_params = SystemParameters(
    sampling_time=0.015,        # Sampling time for discrete controller (seconds)
    num_order=3,                # Numerator order (degree) - must be < den_order
    den_order=4,                # Denominator order (degree) - must be > num_order
    t_end=0.5,                  # Simulation end time (seconds)
    step_amplitude=1.4,         # Step input amplitude
    dt=SIMULATION_DT,           # Time step for continuous plant simulation (seconds)
)

# Optimization parameters (optional - uses defaults if None)
optimization_params = OptimizationParameters(
    population=100,             # Population size for differential evolution
    max_iterations=10000,       # Maximum iterations for optimization
    de_tol=0.1,                 # Convergence tolerance (0.0 to disable early stopping)
    bound_mag=2.0,              # Magnitude of parameter bounds (symmetric: [-bound_mag, bound_mag])
    random_state=None,          # Random seed for reproducibility (None for random)
    verbose=True,               # Print optimization progress
    workers=-1,                 # Use all available CPUs for parallel evaluation
    mutation=(0.5, 1.0),        # Conservative mutation for fine-tuning near stable regions
    recombination=0.6,          # Lower recombination to preserve stable regions better
    strategy='best2bin',        # Best strategy for constrained stability spaces
)

# Output paths (optional)
output_json = "output/inner_controller.json"  # Path to save controller JSON
save_path = "output/inner_response.png"        # Path to save response plot

