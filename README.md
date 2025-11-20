# Hybrid Controller Tuner

A Python tool for automatically tuning discrete-time controllers for hybrid control systems (discrete controller with continuous plant). The tuner uses differential evolution optimization to find controller parameters that meet specified performance requirements such as overshoot and settling time.

## Features

- **Hybrid System Simulation**: Simulates discrete-time controllers with continuous-time plants using Zero-Order Hold (ZOH) discretization
- **Automatic Tuning**: Uses differential evolution to optimize controller parameters
- **Flexible Controller Structure**: Supports arbitrary polynomial controllers (not limited to PID)
- **Order Search**: Optional automatic search over different controller orders
- **Performance Metrics**: Computes overshoot, settling time, and other step response metrics
- **Visualization**: Generates plots showing output, control signal, and error responses

## Installation

1. Clone this repository:

```bash
git clone <repository-url>
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

## Usage

### Basic Usage

1. **Configure your plant and specifications** in `data/specs.json` (see Configuration section below)

2. **Run the tuner**:

```bash
python main.py
```

The tuner will:

- Read configuration from `data/specs.json`
- Optimize controller parameters to meet your specifications
- Save the tuned controller to a JSON file (default: `data/controller.json`)
- Generate a response plot (default: `hybrid_system_response.png`)

### Configuration File

The tuner reads all configuration from `data/specs.json`. Here's the structure:

```json
{
  "plant": {
    "numerator": [-2.936],
    "denominator": [0.031, 1.0, 0.0]
  },
  "sampling_time": 0.015,
  "specs": {
    "max_overshoot_pct": 5.0,
    "settling_time_2pct": 0.05,
    "control_signal_weight": 0.0
  },
  "t_end": 1.0,
  "step_amplitude": 1.0,
  "popsize": 20,
  "maxiter": 100,
  "num_order": 1,
  "den_order": 2,
  "bound_mag": 2.0,
  "search_orders": false,
  "num_order_min": 1,
  "num_order_max": 5,
  "den_order_min": 2,
  "den_order_max": 6,
  "random_state": null,
  "quiet": false,
  "show": false,
  "save_path": "hybrid_system_response.png",
  "output_json": "data/controller.json"
}
```

#### Configuration Parameters

**Plant Definition:**

- `plant.numerator`: List of numerator coefficients for the continuous plant transfer function P(s) in descending powers of s
- `plant.denominator`: List of denominator coefficients for P(s) in descending powers of s

**System Parameters:**

- `sampling_time`: Sampling time Ts in seconds (discrete controller update rate)
- `t_end`: Simulation end time in seconds
- `step_amplitude`: Amplitude of step reference input (default: 1.0)

**Performance Specifications:**

- `specs.max_overshoot_pct`: Maximum allowed percent overshoot (e.g., 5.0 for 5%)
- `specs.settling_time_2pct`: Required 2% settling time in seconds
- `specs.control_signal_weight`: Weight for minimizing control effort (0.0 = disabled, higher = more emphasis on small control signals)

**Optimization Parameters:**

- `popsize`: Population size for differential evolution (default: 25)
- `maxiter`: Maximum number of iterations (default: 60)
- `bound_mag`: Magnitude bound for parameter search range (default: 2.0)
- `random_state`: Random seed for reproducibility (null = random)

**Controller Structure:**

- `num_order`: Numerator order (degree) for fixed-order tuning. Controller numerator has `num_order + 1` coefficients
- `den_order`: Denominator order (degree) for fixed-order tuning. Controller denominator has `den_order + 1` coefficients (leading coefficient is always 1.0)
- **Important**: Controller must be strictly proper: `num_order < den_order`

**Order Search (Optional):**

- `search_orders`: If `true`, automatically searches over different controller orders
- `num_order_min`, `num_order_max`: Range of numerator orders to search
- `den_order_min`, `den_order_max`: Range of denominator orders to search

**Output Options:**

- `quiet`: If `true`, suppress optimization progress output
- `show`: If `true`, display the plot interactively (requires GUI)
- `save_path`: Path to save the response plot
- `output_json`: Path to save the tuned controller JSON file

### Example Configurations

#### Example 1: Simple First-Order Plant

```json
{
  "plant": {
    "numerator": [-0.258873],
    "denominator": [1.0, 0.0, 0.0]
  },
  "sampling_time": 0.5,
  "specs": {
    "max_overshoot_pct": 45.0,
    "settling_time_2pct": 4.0,
    "control_signal_weight": 0.0
  },
  "t_end": 30.0,
  "popsize": 20,
  "maxiter": 300,
  "num_order": 1,
  "den_order": 2,
  "output_json": "data/outer_controller.json",
  "save_path": "data/outer_response.png"
}
```

#### Example 2: Second-Order Plant with Strict Requirements

```json
{
  "plant": {
    "numerator": [-2.936],
    "denominator": [0.031, 1.0, 0.0]
  },
  "sampling_time": 0.015,
  "specs": {
    "max_overshoot_pct": 5.0,
    "settling_time_2pct": 0.05,
    "control_signal_weight": 0.0
  },
  "t_end": 1.0,
  "popsize": 20,
  "maxiter": 100,
  "num_order": 1,
  "den_order": 2,
  "output_json": "data/inner_controller.json",
  "save_path": "data/inner_response.png"
}
```

## Output

### Controller JSON File

The tuned controller is saved as a JSON file with the following structure:

```json
{
  "controller": {
    "numerator": [-1.890024, 1.890024],
    "denominator": [1.0, -0.789375, 0.268998]
  },
  "structure": {
    "num_order": 1,
    "den_order": 2
  },
  "metrics": {
    "steady_state": 1.0,
    "percent_overshoot": 0.013,
    "settling_time_2pct": 4.0,
    "peak_value": 1.0001
  },
  "plant": {
    "numerator": [-0.258873],
    "denominator": [1.0, 0.0, 0.0]
  },
  "sampling_time": 0.5
}
```

The controller transfer function D[z] is defined by:

- `controller.numerator`: Numerator coefficients in descending powers of z
- `controller.denominator`: Denominator coefficients in descending powers of z

### Response Plot

The generated plot shows three subplots:

1. **Output Response**: Step response with reference, steady-state, 2% settling band, peak annotation, and settling time marker
2. **Control Signal**: Discrete control signal u[k] shown as Zero-Order Hold (ZOH)
3. **Error Signal**: Error e(t) = r(t) - y(t)

## How It Works

1. **Plant Discretization**: The continuous plant P(s) is discretized using Zero-Order Hold (ZOH) to obtain P[z]

2. **Controller Structure**: The discrete controller D[z] is parameterized as a rational transfer function with specified numerator and denominator orders

3. **Optimization**: Differential evolution searches for controller parameters that minimize a cost function based on:

   - Overshoot violation penalty
   - Settling time violation penalty
   - Optional control signal magnitude penalty

4. **Simulation**: Each candidate controller is evaluated by simulating the hybrid closed-loop system:

   - Discrete controller: e[k] → u[k] = D[z]e[k]
   - Zero-Order Hold: u[k] → u(t)
   - Continuous plant: u(t) → y(t) = P(s)u(t)
   - Error: e(t) = r(t) - y(t), sampled to e[k]

5. **Validation**: The final controller is validated and metrics are computed

## Tips

- **Strict Properness**: Ensure `num_order < den_order` for a causal, implementable controller
- **Sampling Time**: Choose an appropriate sampling time (typically 10-20x faster than the plant's dominant time constant)
- **Optimization**: Increase `popsize` and `maxiter` for better results, but at the cost of longer computation time
- **Order Search**: Use `search_orders: true` to automatically find the best controller structure, but this significantly increases computation time
- **Tight Specs**: Very tight specifications (low overshoot, fast settling) may require higher-order controllers or may be infeasible

## Troubleshooting

**"System appears unstable"**: The optimized controller may be unstable. Try:

- Increasing `bound_mag` to allow larger parameter values
- Using a higher-order controller (increase `den_order`)
- Relaxing performance specifications

**Optimization doesn't converge**: Try:

- Increasing `maxiter` and/or `popsize`
- Adjusting `bound_mag` to a more appropriate range
- Using order search to find a better controller structure

**Controller not meeting specs**: The specifications may be too tight. Consider:

- Relaxing `max_overshoot_pct` or `settling_time_2pct`
- Using a higher-order controller
- Increasing optimization iterations

## License

See LICENSE file for details.
