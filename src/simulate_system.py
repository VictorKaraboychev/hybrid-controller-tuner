"""
Simulator for user-defined plant systems with discrete controllers.

This module provides utilities to simulate control systems where the plant is defined
in system.py using the system building blocks. The system is defined by the user's
System class which takes optimization parameters and provides a step method.
"""

import numpy as np
from pathlib import Path
import importlib.util
import sys


def load_system_module(system_file="system.py"):
    """
    Load the user's system.py module.

    Parameters
    ----------
    system_file : str or Path
        Path to the system.py file

    Returns
    -------
    module
        The loaded system module
    """
    system_path = Path(system_file)
    if not system_path.exists():
        raise FileNotFoundError(f"System file not found: {system_path}")

    spec = importlib.util.spec_from_file_location("user_system", system_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load system module from {system_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["user_system"] = module
    spec.loader.exec_module(module)

    # Check that System class exists
    if not hasattr(module, "System"):
        raise AttributeError(
            f"System module must define a 'System' class with __init__(params) and step(r, t) methods. "
            f"Found in {system_path}: {dir(module)}"
        )

    return module


def simulate_system(
    params: np.ndarray,
    system_file="system.py",
    t_end=10.0,
    step_amplitude=1.0,
    dt=0.001,
):
    """
    Simulate a hybrid control system with a user-defined System class.

    The system is defined by the user's System class in system.py, which takes
    optimization parameters and provides a step method that runs through the system.

    Parameters
    ----------
    params : np.ndarray
        Array of optimization parameters used to create the System instance.
    system_file : str or Path
        Path to the system.py file defining the System class
    t_end : float
        End time for simulation (seconds)
    step_amplitude : float
        Amplitude of step input
    dt : float
        Time step for continuous-time plant simulation (seconds)

    Returns
    -------
    t : array
        Time vector in seconds
    y : array
        Output response
    u : array
        Control signal (discrete samples)
    e : array
        Error signal e(t) = r(t) - y(t)
    """
    # Load user's system module
    system_module = load_system_module(system_file)
    SystemClass = system_module.System

    # Create System instance with params
    # The System class determines how to interpret the params
    system = SystemClass(params=params)

    # Create fine time vector for continuous simulation
    n_fine = int(t_end / dt) + 1
    t = np.arange(n_fine) * dt

    # Initialize arrays
    y = np.zeros(n_fine)
    u = np.zeros(n_fine)  # Continuous control signal
    e = np.zeros(n_fine)  # Continuous error signal
    r = np.ones(n_fine) * step_amplitude  # Step reference signal

    # Simulate step-by-step at continuous rate
    for i in range(n_fine):
        t_i = t[i]
        r_i = r[i]

        # Call system's step method
        e_i, u_i, y_i = system.step(r_i, t_i)

        # Store values
        e[i] = e_i
        u[i] = u_i
        y[i] = y_i

    # Create discrete arrays for compatibility (sample at a reasonable rate)
    # Use dt as the sampling rate for output
    Ts = dt
    n_samples = int(t_end / Ts) + 1
    t_disc = np.arange(n_samples) * Ts
    u_disc = np.interp(t_disc, t, u)
    e_disc = np.interp(t_disc, t, e)
    y_disc = np.interp(t_disc, t, y)

    return t_disc, y_disc, u_disc, e_disc
