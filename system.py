"""
User-defined control system.

This file defines the control system using the system building blocks.
The System class takes optimization parameters and creates the optimized block,
then provides a step method to run the system simulation.
"""

import numpy as np
from src.system_blocks import ContinuousTF, DiscreteTF, Saturation, PID
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

# These will be created as instance variables in the System class

# ============================================================================
# System class
# ============================================================================


class System:
    """
    Control system class that can be optimized.

    Takes optimization parameters in __init__ to create the optimized block,
    and provides a step method to run the system simulation at each time step.
    """

    def __init__(self, params: np.ndarray):
        """
        Initialize the system with optimization parameters.

        Parameters
        ----------
        params : np.ndarray
            Array of optimization parameters. The System class determines how
            these parameters are interpreted. For discrete controller optimization,
            these are [num_coeffs..., den_coeffs...] where the number of coefficients
            is determined by system_params.num_order and system_params.den_order.
        """

        # Create plant blocks (these are fixed, not optimized)
        self.s1 = Saturation(min_val=-0.7, max_val=0.7)

        self.d2 = DiscreteTF(
            num=[
                -1.6437408920219747,
                1.477660476362062,
                0.08838959630344201,
                0.07772974353397721,
            ],
            den=[
                1.0,
                -1.6565270508810168,
                0.36370233042832245,
                0.8084480013102704,
                -0.515623280857576,
            ],
            sampling_time=0.015,
        )

        self.s2 = Saturation(min_val=-6.0, max_val=6.0)

        self.p2 = ContinuousTF(num=[-2.936], den=[0.031, 1.0, 0.0], dt=SIMULATION_DT)

        self.p1 = ContinuousTF(num=[-0.258873], den=[1.0, 0.0, 0.0], dt=SIMULATION_DT)

        # Create the optimized controller block using from_params
        # For 4 parameters: num_order=1, den_order=2 gives (1+1) + 2 = 4 parameters
        # self.d_controller = DiscreteTF.from_params(
        #     params=params,
        #     num_order=1,
        #     den_order=2,
        #     sampling_time=0.5,
        # )

        self.pid_controller = PID.from_params(
            params=params,
            sampling_time=0.01,
        )
    
    def reset(self):
        """Reset all blocks to their initial state."""
        self.p1.reset()
        self.p2.reset()
        self.d2.reset()
        self.s1.reset()
        self.s2.reset()
        self.pid_controller.reset()

    def step(self, r: float, t: float):
        """
        Step the control system with reference signal and time.

        Parameters
        ----------
        r : float
            Reference/input signal at current time step.
        t : float
            Current time (seconds).

        Returns
        -------
        e : float
            Error signal.
        u : float
            Control signal.
        y : float
            New output signal.
        """
        # Outer Error
        e1 = r - self.p1.y

        # Outer Controller
        u1 = self.pid_controller.step(t, e1)
        # u1 = self.s1.step(u1)

        # # Inner Error
        # e2 = u1 - self.p2.y

        # # Inner Controller
        # u2 = self.d2.step(t, e2)
        # u2 = self.s2.step(u2)

        # # Inner Plant
        # y2 = self.p2.step(u2)

        # Outer Plant
        y1 = self.p1.step(u1)

        return e1, u1, y1


# ============================================================================
# Tuning Parameters and Specifications
# ============================================================================

# Performance specifications for controller tuning (required)
specs = PerformanceSpecs(
    max_overshoot_pct=45.0,  # Maximum allowed percent overshoot
    settling_time_2pct=7.0,  # Required 2% settling time (seconds)
    max_control_signal=0.7,  # Maximum allowed control signal (None for no limit)
)

# Cost function weights (optional - uses defaults if None)
cost_weights = CostWeights(
    overshoot_weight=1.0,
    settling_time_weight=1.0,
    steady_state_error_weight=100.0,
    control_signal_limit_weight=2.0,
)

system_params = SystemParameters(
    num_parameters=3,  # Total number of optimization parameters
    t_end=15.0,  # Simulation end time (seconds)
    step_amplitude=0.15,  # Step input amplitude
    dt=SIMULATION_DT,  # Time step for continuous plant simulation (seconds)
)

# Optimization parameters (optional - uses defaults if None)
optimization_params = OptimizationParameters(
    population=10,  # Population size for differential evolution
    max_iterations=1000,  # Maximum iterations for optimization
    de_tol=0.0,  # Convergence tolerance (0.0 to disable early stopping)
    bounds=[
        (-10.0, 10.0),  # Kp bounds
        (-1.0, 1.0),  # Ki bounds (much smaller for stability)
        (-10.0, 10.0),  # Kd bounds (much smaller for stability)
    ],
    random_state=None,  # Random seed for reproducibility (None for random)
    verbose=True,  # Print optimization progress
    workers=-1,  # Use all available CPUs for parallel evaluation
    # mutation=(0.5, 1.9),        # Conservative mutation for fine-tuning near stable regions
    # recombination=0.6,          # Lower recombination to preserve stable regions better
    # strategy='best2bin',        # Best strategy for constrained stability spaces
)

# Output paths (optional)
output_json = "output/outer_controller.json"  # Path to save controller JSON
save_path = "output/outer_response.png"  # Path to save response plot
