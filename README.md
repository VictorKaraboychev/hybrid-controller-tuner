# Hybrid Controller Tuner

A Python framework for automatically tuning discrete-time controllers for hybrid control systems (discrete controller with continuous plant). Uses differential evolution optimization to find controller parameters that meet specified performance requirements.

## Features

- **Block-Based Architecture**: Modular system design using reusable blocks (`Block` base class)
- **Hybrid System Simulation**: Simulates discrete-time controllers with continuous-time plants using Zero-Order Hold (ZOH)
- **Adaptive Timestep**: Variable timestep mode for efficient long simulations (up to 9x faster)
- **Automatic Tuning**: Differential evolution optimization with parallel evaluation
- **Flexible Controllers**: Supports PID controllers and arbitrary polynomial transfer functions
- **Signal Generators**: Built-in signal classes (Step, Ramp, Sinusoid, SquareWave, Constant)
- **Nested Control Systems**: Build cascaded control loops (e.g., outer/inner loops)
- **Performance Metrics**: Computes overshoot, settling time, tracking error, and more
- **Visualization**: Generates plots showing output, control signals, and error responses

## Installation

1. Clone this repository:

```bash
git clone "https://github.com/VictorKaraboychev/hybrid-controller-tuner"
cd hybrid-controller-tuner
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Required packages:

- `numpy >= 1.21.0`
- `scipy >= 1.7.0`
- `matplotlib >= 3.4.0`

## Quick Start

### Basic Usage

1. **Define your control system** in `main.py` or create a new system file:

```python
from src.systems import PID, Block, ContinuousTF, Saturation
from src.signals import Step
from src.simulate_system import simulate_system
from src.response_metrics import compute_metrics
from src.tune_discrete_controller import OptimizationParameters, optimize

class MySystem(Block):
    def __init__(self, params: np.ndarray):
        # PID controller: [Kp, Ki, Kd, sampling_time]
        self.pid = PID(
            Kp=params[0],
            Ki=params[1],
            Kd=params[2],
            sampling_time=params[3],
        )
        # Continuous plant
        self.plant = ContinuousTF(num=[-2.936], den=[0.031, 1.0, 0.0])
        # Input saturation
        self.s = Saturation(min_val=-6.0, max_val=6.0)
    
    def reset(self):
        self.pid.reset()
        self.plant.reset()
        self.s.reset()
    
    def __call__(self, t: float, r: float):
        e = r - self.plant.y
        u = self.pid(t, e)
        u = self.s(t, u)
        y = self.plant(t, u)
        return r, y, e, u
    
    def cost(self) -> float:
        results = simulate_system(self, Step(amplitude=1.4), 1.0)
        metrics = compute_metrics(results)
        return metrics["tracking_error"]
```

2. **Configure optimization parameters**:

```python
optimization_params = OptimizationParameters(
    num_parameters=4,  # Number of parameters to optimize
    population=20,  # Population size for differential evolution
    max_iterations=1000,  # Maximum iterations
    de_tol=0.000001,  # Convergence tolerance
    bounds=[
        (-100.0, 100.0),  # Kp bounds
        (-5.0, 5.0),      # Ki bounds
        (-100.0, 100.0),  # Kd bounds
        (0.01, 1.0),      # Sampling time bounds
    ],
    verbose=True,
    workers=-1,  # Use all CPUs
)
```

3. **Run the optimizer**:

```python
params = optimize(MySystem, optimization_params)
```

4. **Simulate and visualize**:

```python
from src.plotting_utils import plot_response

system = MySystem(params=params)
results = simulate_system(system, Step(amplitude=1.4), 1.0)
metrics = compute_metrics(results)
plot_response(results, metrics, save_path="output/response.png")
```

## System Architecture

### Block-Based Design

All system components extend the `Block` base class:

```python
class Block(ABC):
    @abstractmethod
    def __call__(self, t: float, r: float) -> float:
        """Step the block with time t and input r, return output."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset internal state."""
        pass
```

### Available Blocks

- **`ContinuousTF`**: Continuous-time transfer functions (s-domain)
  - Automatically computes `dt` from time difference
  - Uses Euler integration for state-space representation

- **`DiscreteTF`**: Discrete-time transfer functions (z-domain)
  - Handles zero-order hold (ZOH) automatically
  - Updates only at discrete sample times

- **`PID`**: PID controller with direct difference equations
  - Parameters: `Kp`, `Ki`, `Kd`, `sampling_time`
  - No derivative filter (simplified implementation)

- **`Saturation`**: Signal limiting block
  - Parameters: `min_val`, `max_val`

### Signal Generators

Signal classes extend the `Signal` base class:

```python
from src.signals import Step, Ramp, Sinusoid, SquareWave, Constant

# Step signal
r_func = Step(amplitude=0.15)

# Ramp signal
r_func = Ramp(slope=0.1, offset=0.0)

# Sinusoid
r_func = Sinusoid(amplitude=1.0, frequency=1.0, phase=0.0)

# Square wave
r_func = SquareWave(amplitude=1.0, frequency=1.0, duty_cycle=0.5)

# Constant
r_func = Constant(value=0.5)
```

## Simulation

### Fixed Timestep Mode

```python
results = simulate_system(
    system, 
    Step(amplitude=0.15), 
    t_end=10.0, 
    dt=0.001,
    dt_mode='fixed'  # Default
)
```

### Variable Timestep Mode

For long simulations where signals settle, variable timestep can be up to 9x faster:

```python
results = simulate_system(
    system,
    Step(amplitude=0.15),
    t_end=10.0,
    dt=0.001,
    dt_mode='variable',
    adaptive_tolerance=1.0,  # Higher = more aggressive about increasing dt
    max_dt=0.1,              # Maximum timestep
    min_dt=1e-6              # Minimum timestep
)
```

**When to use variable timestep:**
- Long simulations (t_end > 5 seconds)
- Systems that settle into steady state
- When accuracy in transients is less critical

**When to use fixed timestep:**
- Short simulations
- Need precise transient capture
- Systems with continuous rapid changes

### Simulation Results

`simulate_system` returns a tuple of numpy arrays:

```python
results = simulate_system(system, r_func, t_end, dt)
# results = (t, signal_1, signal_2, ..., signal_n)

# For a system returning (r, y, e, u):
t, r, y, e, u = results
```

The tuple format is consistent: `(t, r, y, e, ...other_signals)` where:
- `t`: Time array
- `r`: Reference signal array
- `y`: Output response array
- `e`: Error signal array
- Additional signals follow (control signals, etc.)

## System Definition

### Creating a System Class

Your system class must:

1. Extend `Block`
2. Implement `__init__(self, params: np.ndarray)` - takes optimization parameters
3. Implement `__call__(self, t: float, r: float)` - returns tuple of signals
4. Implement `reset(self)` - resets all block states
5. Implement `cost(self) -> float` - returns cost for optimization

Example:

```python
class MySystem(Block):
    def __init__(self, params: np.ndarray):
        # Initialize blocks from params
        self.controller = PID(
            Kp=params[0],
            Ki=params[1],
            Kd=params[2],
            sampling_time=params[3],
        )
        self.plant = ContinuousTF(num=[1.0], den=[1.0, 1.0, 0.0])
        self.saturation = Saturation(min_val=-10.0, max_val=10.0)
    
    def reset(self):
        self.controller.reset()
        self.plant.reset()
        self.saturation.reset()
    
    def __call__(self, t: float, r: float):
        # Control loop
        e = r - self.plant.y
        u = self.controller(t, e)
        u = self.saturation(t, u)
        y = self.plant(t, u)
        # Return signals in order: (r, y, e, ...other_signals)
        return r, y, e, u
    
    def cost(self) -> float:
        # Simulate and compute cost
        results = simulate_system(self, Step(amplitude=1.0), 5.0)
        metrics = compute_metrics(results)
        # Return cost (e.g., tracking error)
        return metrics["tracking_error"]
```

### Nested Systems

Build cascaded control loops:

```python
class CascadedSystem(Block):
    def __init__(self, params: np.ndarray):
        # Outer loop controller
        self.outer_pid = PID(Kp=params[0], Ki=params[1], Kd=params[2], 
                            sampling_time=params[3])
        self.outer_plant = ContinuousTF(...)
        
        # Inner loop controller
        self.inner_pid = PID(Kp=params[4], Ki=params[5], Kd=params[6],
                            sampling_time=params[7])
        self.inner_plant = ContinuousTF(...)
    
    def __call__(self, t: float, r: float):
        # Outer loop
        e_outer = r - self.outer_plant.y
        u_outer = self.outer_pid(t, e_outer)
        
        # Inner loop (outer output is inner reference)
        e_inner = u_outer - self.inner_plant.y
        u_inner = self.inner_pid(t, e_inner)
        y_inner = self.inner_plant(t, u_inner)
        
        # Outer plant uses inner output
        y_outer = self.outer_plant(t, y_inner)
        
        return r, y_outer, e_outer, u_outer, y_inner, e_inner, u_inner
```

## Optimization

### Optimization Parameters

```python
from src.tune_discrete_controller import OptimizationParameters

optimization_params = OptimizationParameters(
    num_parameters=4,           # Number of parameters to optimize
    population=20,              # Population size (larger = better but slower)
    max_iterations=1000,        # Maximum iterations
    de_tol=0.000001,           # Convergence tolerance (0.0 to disable)
    bounds=[                    # Parameter bounds (one per parameter)
        (-100.0, 100.0),
        (-5.0, 5.0),
        (-100.0, 100.0),
        (0.01, 1.0),
    ],
    random_state=42,            # Random seed (None for random)
    verbose=True,               # Print progress
    workers=-1,                 # Number of parallel workers (-1 = all CPUs)
)
```

### Running Optimization

```python
from src.tune_discrete_controller import optimize

# Optimize
params = optimize(MySystem, optimization_params)

print(f"Optimized parameters: {params}")
```

### Cost Function

The `cost()` method in your system class defines what to optimize. Common approaches:

```python
def cost(self) -> float:
    results = simulate_system(self, Step(amplitude=1.0), 5.0)
    metrics = compute_metrics(results)
    
    # Option 1: Minimize tracking error
    return metrics["tracking_error"]
    
    # Option 2: Weighted combination
    return (
        metrics["tracking_error"] +
        10.0 * max(0, metrics["percent_overshoot"] - 5.0) +
        5.0 * max(0, metrics["settling_time_2pct"] - 1.0)
    )
```

## Performance Metrics

Available metrics from `compute_metrics()`:

- `steady_state`: Final steady-state value
- `percent_overshoot`: Percentage overshoot
- `settling_time_2pct`: 2% settling time (seconds)
- `peak_value`: Peak output value
- `tracking_error`: Sum of squared errors × dt

```python
from src.response_metrics import compute_metrics

results = simulate_system(system, r_func, t_end, dt)
metrics = compute_metrics(results)

print(f"Steady state: {metrics['steady_state']}")
print(f"Overshoot: {metrics['percent_overshoot']}%")
print(f"Settling time: {metrics['settling_time_2pct']}s")
print(f"Tracking error: {metrics['tracking_error']}")
```

## Visualization

```python
from src.plotting_utils import plot_response

results = simulate_system(system, r_func, t_end, dt)
metrics = compute_metrics(results)

fig, axes = plot_response(
    results,           # Tuple of arrays from simulate_system
    metrics,           # Dictionary from compute_metrics
    save_path="output/response.png"
)
```

The plot shows:
1. **Output Response**: With reference, steady-state, 2% settling band, peak annotation
2. **Error Signal**: Error over time
3. **Control Signals**: Any additional signals beyond (t, r, y, e)

## Examples

See the `systems/` directory for example implementations:

- `systems/inner.py`: Inner loop system
- `systems/outer.py`: Outer loop system  
- `systems/full.py`: Cascaded inner/outer system

Run examples:

```bash
# Test PID controller
python test_pid.py

# Test variable timestep performance
python test_variable_timestep.py

# Run optimization
python main.py
```

## Tips

- **Variable Timestep**: Use for long simulations (t_end > 5s) to get 5-9x speedup
- **Optimization**: Increase `population` and `max_iterations` for better results
- **Sampling Time**: Choose 10-20x faster than plant's dominant time constant
- **Bounds**: Set realistic bounds based on your system's expected parameter ranges
- **Parallel Evaluation**: Use `workers=-1` to utilize all CPU cores
- **Cost Function**: Design your cost function to match your priorities (tracking vs. overshoot vs. settling time)

## Troubleshooting

**Optimization doesn't converge:**
- Increase `population` and/or `max_iterations`
- Adjust parameter `bounds` to more appropriate ranges
- Check that your cost function is well-behaved

**System appears unstable:**
- Check that controller parameters are within reasonable bounds
- Verify plant transfer function is correct
- Consider using a higher-order controller

**Variable timestep too slow:**
- Increase `adaptive_tolerance` (e.g., 1.0 or higher)
- Increase `max_dt` to allow larger timesteps
- Use fixed timestep for short simulations

**Simulation accuracy issues:**
- Use fixed timestep mode for precise results
- Decrease `dt` for better resolution
- Check that `max_dt` in variable mode isn't too large

## License

See LICENSE file for details.
