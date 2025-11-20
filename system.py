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
    e: Error signal (typically r - p2.y for final output)
    u: Control signal
    y: New output signal

Note: Use transfer function .y attributes (e.g., p1.y, p2.y) to access last outputs.
"""

import numpy as np
from src.system_blocks import ContinuousTF, DiscreteTF, Saturation
from src.tune_discrete_controller import (
    PerformanceSpecs,
    CostWeights,
    SystemParameters,
    OptimizationParameters,
)

SIMULATION_DT = 0.015

# ============================================================================
# Define your plant blocks here
# ============================================================================

s1 = Saturation(min_val=-0.7, max_val=0.7)

d2 = DiscreteTF(
    num=[-1.6437408920219747, 1.477660476362062, 0.08838959630344201, 0.07772974353397721],
    den=[1.0, -1.6565270508810168, 0.36370233042832245, 0.8084480013102704, -0.515623280857576],
    sampling_time=0.015
)

s2 = Saturation(min_val=-6.0, max_val=6.0)

p2 = ContinuousTF(
    num=[-2.936],
    den=[0.031, 1.0, 0.0],
    dt=SIMULATION_DT
)

p1 = ContinuousTF(
    num=[-0.258873],
    den=[1.0, 0.0, 0.0],
    dt=SIMULATION_DT
)

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
    
    # Outer Error
    e1 = r - p1.y
    
    # Outer Controller
    u1 = d_controller.step(t, e1)
    # u1 = s1.step(u1)
    
    u1 += p1.y;
    
    # # Inner Error
    # e2 = u1 - p2.y
    
    # # Inner Controller
    # u2 = d2.step(t, e2)
    # u2 = s2.step(u2)
    
    # # Inner Plant
    # y2 = p2.step(u2)
    
    # Outer Plant
    y1 = p1.step(u1)
    
    return e1, u1, y1


def reset():
    """
    Reset all blocks to their initial state.
    
    This should be called at the start of a new simulation.
    """
    p1.reset()
    s1.reset()
    d2.reset()
    s2.reset()
    p2.reset()


# ============================================================================
# Tuning Parameters and Specifications
# ============================================================================

# Performance specifications for controller tuning (required)
specs = PerformanceSpecs(
    max_overshoot_pct=45.0,      # Maximum allowed percent overshoot
    settling_time_2pct=7.0,     # Required 2% settling time (seconds)
    max_control_signal=0.7,      # Maximum allowed control signal (None for no limit)
)

# Cost function weights (optional - uses defaults if None)
cost_weights = CostWeights(
    overshoot_weight=1.0,
    settling_time_weight=4.0,
    steady_state_error_weight=3.0,
    control_signal_limit_weight=2.0,
)

# System parameters (required)
system_params = SystemParameters(
    sampling_time=0.5,          # Sampling time for discrete controller (seconds)
    num_order=3,                # Numerator order (degree) - must be < den_order
    den_order=4,                # Denominator order (degree) - must be > num_order
    t_end=15.0,                 # Simulation end time (seconds)
    step_amplitude=0.15,        # Step input amplitude
    dt=SIMULATION_DT,           # Time step for continuous plant simulation (seconds)
)

# Optimization parameters (optional - uses defaults if None)
optimization_params = OptimizationParameters(
    population=100,             # Population size for differential evolution
    max_iterations=4000,       # Maximum iterations for optimization
    de_tol=0.1,                 # Convergence tolerance (0.0 to disable early stopping)
    bound_mag=2.0,              # Magnitude of parameter bounds (symmetric: [-bound_mag, bound_mag])
    random_state=None,          # Random seed for reproducibility (None for random)
    verbose=True,               # Print optimization progress
    workers=-1,                 # Use all available CPUs for parallel evaluation
    # mutation=(0.5, 1.9),        # Conservative mutation for fine-tuning near stable regions
    # recombination=0.6,          # Lower recombination to preserve stable regions better
    # strategy='best2bin',        # Best strategy for constrained stability spaces
)

# Output paths (optional)
output_json = "output/outer_controller.json"  # Path to save controller JSON
save_path = "output/outer_response.png"        # Path to save response plot
