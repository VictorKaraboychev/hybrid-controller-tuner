"""
Simulator for user-defined plant systems with discrete controllers.

This module provides utilities to simulate control systems where the plant is defined
in system.py using the system building blocks. The controller is a discrete-time
transfer function, and the plant is defined by the user's system function.
"""

import numpy as np
from pathlib import Path
import importlib.util
import sys
from src.system_blocks import DiscreteTF


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
    
    # Check that system function exists
    if not hasattr(module, "system"):
        raise AttributeError(
            f"System module must define a 'system(controller, r, t)' function. "
            f"Use plant.y (or any transfer function's .y attribute) to access the last output. "
            f"Found in {system_path}: {dir(module)}"
        )
    
    return module


def simulate_system(
    controller_tf,
    system_file="system.py",
    sampling_time=0.1,
    t_end=10.0,
    step_amplitude=1.0,
    dt=0.001,
):
    """
    Simulate a hybrid control system with a user-defined system.
    
    The controller is a discrete-time transfer function, and the system is defined
    by the user's system.py file. The controller is passed to the user's system
    function, which takes reference r and time t, and returns error e, control signal u,
    and new output y.
    
    The system function should use plant.y (or any transfer function's .y attribute) to
    access the last output value. Both ContinuousTF and DiscreteTF blocks have a .y
    attribute that stores their last output value.
    
    Parameters
    ----------
    controller_tf : tuple
        Discrete-time controller transfer function as (num, den) where num and den
        are arrays of coefficients in descending order of powers of z.
    system_file : str or Path
        Path to the system.py file defining the system
    sampling_time : float
        Sampling time Ts in seconds
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
    system_func = system_module.system
    
    # Reset function if available
    if hasattr(system_module, "reset"):
        system_module.reset()
    
    Ts = sampling_time
    D_num, D_den = controller_tf
    
    # Create controller using DiscreteTF block
    controller = DiscreteTF(
        num=D_num,
        den=D_den,
        sampling_time=Ts
    )
    
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
        
        # Call user's system function at continuous rate
        # The controller handles sampling internally via step_with_sampling()
        # The plant is stepped continuously
        # System function should use plant.y to access the last output
        e_i, u_i, y_i = system_func(controller, r_i, t_i)
        
        # Store values
        e[i] = e_i
        u[i] = u_i
        y[i] = y_i
        
        # Check for instability
        if i > 10 and (np.abs(y_i) > 1e8 or np.abs(u_i) > 1e8):
            raise ValueError("System appears unstable - response growing unbounded")
    
    # Create discrete arrays for compatibility (sample at controller rate)
    n_samples = int(t_end / Ts) + 1
    t_disc = np.arange(n_samples) * Ts
    u_disc = np.interp(t_disc, t, u)
    e_disc = np.interp(t_disc, t, e)
    y_disc = np.interp(t_disc, t, y)
    
    return t_disc, y_disc, u_disc, e_disc

