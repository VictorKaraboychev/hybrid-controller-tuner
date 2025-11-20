# System Builder Documentation

This document explains how to use the system building blocks to define custom control systems.

## Overview

The system builder provides reusable blocks that can be combined to create control systems:

- **ContinuousTF**: Continuous-time transfer function (s-domain)
- **DiscreteTF**: Discrete-time transfer function (z-domain)
- **Saturation**: Signal saturation/limiting block

## Defining Your System

Create a `system.py` file in the project root with the following structure:

```python
from src.system_blocks import ContinuousTF, DiscreteTF, Saturation
import numpy as np

# Define your plant blocks
plant = ContinuousTF(num=[...], den=[...], dt=0.001)
input_saturation = Saturation(min_val=-10.0, max_val=10.0)

def system(controller, r, t):
    """
    Control system function called by the tuner at each discrete time step.

    The controller is passed to this function, and it computes the error,
    control signal, and plant output. The time parameter t allows for
    time-varying disturbances.

    Parameters:
    -----------
    controller : DiscreteTF
        Discrete-time controller block
    r : float
        Reference/input signal at current time step
    t : float
        Current time (seconds)

    Returns:
    --------
    e : float
        Error signal
    u : float
        Control signal
    y : float
        New output signal
    """
    # Compute error: e = r - plant.y
    # Use plant.y to access the last output value (or any transfer function's .y attribute)
    e = r - plant.y

    # Compute control signal using the controller
    # The controller handles sampling internally - it only updates at discrete times
    # and holds its output between samples (ZOH)
    u_unsat = controller.step(t, e)

    # Apply saturation (optional)
    u = input_saturation.step(u_unsat)

    # Example: Add time-varying disturbance
    # disturbance = 0.1 * np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz sine wave
    # u = u + disturbance

    # Apply control signal to plant
    y = plant.step(u)

    return e, u, y

def reset():
    """Reset all blocks (optional but recommended)."""
    plant.reset()
    input_saturation.reset()

# ============================================================================
# Tuning Parameters and Specifications
# ============================================================================

from src.tune_discrete_controller import (
    PerformanceSpecs,
    CostWeights,
    SystemParameters,
    OptimizationParameters,
)

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
    num_order=1,                # Numerator order (degree) - must be < den_order
    den_order=3,                # Denominator order (degree) - must be > num_order
    t_end=0.5,                  # Simulation end time (seconds, default: 5.0)
    step_amplitude=1.4,         # Step input amplitude (default: 1.0)
    dt=0.001,                    # Time step for continuous plant simulation (default: 0.001)
)

# Optimization parameters (optional - uses defaults if None)
optimization_params = OptimizationParameters(
    population=40,              # Population size (default: 25)
    max_iterations=250,         # Maximum iterations (default: 60)
    de_tol=0.0,                 # Convergence tolerance (default: 0.001, 0.0 to disable)
    de_atol=1e-8,               # Absolute convergence tolerance (default: 1e-8)
    bound_mag=2.0,              # Parameter bound magnitude (default: 2.0)
    random_state=None,          # Random seed (default: None)
    verbose=True,               # Print progress (default: True)
    workers=-1,                 # Parallel workers (-1 = all CPUs, default: -1)
    mutation=(0.5, 1.0),        # Mutation constant (default: (0.75, 1.5))
    recombination=0.6,         # Recombination constant (default: 0.7)
    strategy='best2bin',        # Mutation strategy (default: 'best1bin')
)

# Output paths (optional)
output_json = "output/controller.json"  # Path to save controller JSON
save_path = "output/response.png"        # Path to save response plot
```

## Building Blocks

### ContinuousTF

Represents a continuous-time transfer function in the s-domain.

**Parameters:**

- `num`: Numerator coefficients in descending powers of s
  - Example: `[1, 2]` represents `s + 2`
- `den`: Denominator coefficients in descending powers of s
  - Example: `[1, 3, 2]` represents `s^2 + 3*s + 2`
- `dt`: Integration time step (seconds, default: 0.001)

**Example:**

```python
# P(s) = 1 / (s^2 + 2*s + 1)
plant = ContinuousTF(
    num=[1.0],
    den=[1.0, 2.0, 1.0],
    dt=0.001
)
```

**Methods:**

- `step(u)`: Step the system with input `u`, returns output `y`
- `reset()`: Reset internal state to zero

**Attributes:**

- `.y`: Last output value (accessible after calling `step()`)

### DiscreteTF

Represents a discrete-time transfer function in the z-domain.

**Parameters:**

- `num`: Numerator coefficients in descending powers of z
  - Example: `[1, 0.5]` represents `z + 0.5`
- `den`: Denominator coefficients in descending powers of z
  - Example: `[1, -0.5, 0.2]` represents `z^2 - 0.5*z + 0.2`
- `sampling_time`: Sampling time Ts (seconds)

**Example:**

```python
# D[z] = (z + 0.5) / (z^2 - 0.5*z + 0.2)
controller = DiscreteTF(
    num=[1.0, 0.5],
    den=[1.0, -0.5, 0.2],
    sampling_time=0.1
)
```

**Methods:**

- `step(t, u)`: Step the system with input `u` at time `t`, returns output `y`
  - The time parameter `t` is used to determine when to sample (updates occur at discrete sample times)
  - Between samples, the output is held constant (ZOH behavior)
- `reset()`: Reset internal history to zero

**Attributes:**

- `.y`: Last output value (accessible after calling `step()`)

**Note:** The zero-order hold (ZOH) behavior is inherent in discrete-time systems - the output is held constant between samples.

### Saturation

Limits a signal between minimum and maximum values.

**Parameters:**

- `min_val`: Minimum output value (None for no lower limit)
- `max_val`: Maximum output value (None for no upper limit)

**Example:**

```python
# Limit signal between -10 and 10
saturation = Saturation(min_val=-10.0, max_val=10.0)

# Only upper limit
saturation = Saturation(max_val=10.0)

# Only lower limit
saturation = Saturation(min_val=-5.0)
```

**Methods:**

- `step(u)`: Apply saturation to input `u`, returns saturated output
- `reset()`: No-op (included for consistency)

## Accessing Last Output Values

All transfer function blocks (`ContinuousTF` and `DiscreteTF`) have a `.y` attribute that stores their last output value. This allows you to access previous outputs without needing to pass them as parameters.

**Example:**

```python
def system(controller, r, t):
    # Access the plant's last output
    e = r - plant.y

    # You can also access outputs from other transfer functions
    # e.g., in a nested control system:
    # e1 = r - p1.y  # Outer loop error
    # e2 = u1 - p2.y  # Inner loop error

    u = controller.step(t, e)
    y = plant.step(u)
    return e, u, y
```

This eliminates the need for a `y_prev` parameter in the system function signature.

## Examples

### Example 1: Simple Single-Loop Control

See `data/inner_system.py` for a simple single-loop control system example.

### Example 2: Nested Control Loops

See `system.py` or `data/outer_system.py` for examples of nested control systems (outer and inner loops):

```python
def system(d_controller, r, t):
    # Outer Error
    e1 = r - p1.y

    # Outer Controller
    u1 = d_controller.step(t, e1)
    u1 = s1.step(u1)

    # Inner Error
    e2 = u1 - p2.y

    # Inner Controller
    u2 = d2.step(t, e2)
    u2 = s2.step(u2)

    # Inner Plant
    y2 = p2.step(u2)

    # Outer Plant
    y1 = p1.step(y2)

    return e1, u1, y1
```

This example shows how to build cascaded control systems where the outer controller's output becomes the reference for the inner controller.

## Tuning Parameters

All tuning parameters and specifications are defined in `system.py`. Required parameters:

- **`specs`**: `PerformanceSpecs` object with:

  - `max_overshoot_pct`: Maximum allowed percent overshoot
  - `settling_time_2pct`: Required 2% settling time (seconds)
  - `max_control_signal`: Maximum allowed control signal (None for no limit)

- **`system_params`**: `SystemParameters` object with:
  - `sampling_time`: Sampling time for discrete controller (seconds)
  - `num_order`: Numerator order (degree) - must be < den_order
  - `den_order`: Denominator order (degree) - must be > num_order
  - `t_end`: Simulation end time (default: 5.0)
  - `step_amplitude`: Step input amplitude (default: 1.0)
  - `dt`: Time step for continuous plant simulation (default: 0.001)

Optional parameters (with defaults):

- **`cost_weights`**: `CostWeights` object for cost function weights (uses defaults if None)
- **`optimization_params`**: `OptimizationParameters` object with:
  - `population`: Population size (default: 25)
  - `max_iterations`: Maximum iterations (default: 60)
  - `de_tol`: Convergence tolerance (default: 0.001, set to 0.0 to disable)
  - `de_atol`: Absolute convergence tolerance (default: 1e-8)
  - `bound_mag`: Parameter bound magnitude (default: 2.0)
  - `random_state`: Random seed (default: None)
  - `verbose`: Print optimization progress (default: True)
  - `workers`: Number of parallel workers (-1 = all CPUs, default: -1)
  - `mutation`: Mutation constant - tuple for dithering (default: (0.75, 1.5))
    - Lower values (0.5-1.0) are more conservative, better for fine-tuning
    - Higher values (1.0-2.0) provide more exploration
  - `recombination`: Recombination constant/crossover probability (default: 0.7)
    - Lower values (0.5-0.6) preserve stable regions better
    - Higher values (0.8-0.9) provide more exploration
  - `strategy`: Mutation strategy (default: 'best1bin')
    - `'best1bin'`: Uses best solution (good general-purpose)
    - `'rand1bin'`: Uses random solution (more exploration)
    - `'best2bin'`: Uses two difference vectors (good for constrained spaces)
    - `'rand2bin'`: Most exploration (good for difficult landscapes)
- **`output_json`**: Path to save controller JSON (default: "output/controller.json")
- **`save_path`**: Path to save response plot (default: "output/response.png")

## Tips

1. **Time Steps**: Make sure the `dt` parameter in `ContinuousTF` matches the `dt` used in simulation
2. **Sampling Time**: The `sampling_time` in `DiscreteTF` should match your actual sampling period
3. **State Management**: Always call `reset()` at the start of a new simulation
4. **Transfer Function Format**: Coefficients are in descending powers (highest order first)
5. **Normalization**: `DiscreteTF` automatically normalizes the denominator so the leading coefficient is 1.0
6. **Time-Varying Disturbances**: Use the `t` parameter in your `system(controller, r, t)` function to add time-dependent disturbances or other time-varying effects to your plant model
7. **All Parameters in One File**: All tuning parameters, specs, and plant definition are in `system.py` - no JSON file needed!
8. **Accessing Outputs**: Use the `.y` attribute on any transfer function to access its last output value (e.g., `plant.y`, `p1.y`, `p2.y`)
9. **Optimization Parameters**: For finding stable controllers, try:
   - `mutation=(0.5, 1.0)`: More conservative for fine-tuning
   - `recombination=0.6`: Lower to preserve stable regions
   - `strategy='best2bin'`: Good for constrained stability spaces
10. **Nested Systems**: For cascaded control systems, use the output of one plant as input to another, and access intermediate outputs using `.y` attributes
