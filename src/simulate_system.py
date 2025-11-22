import numpy as np

def simulate_system(system, r_func, t_end: float, dt: float | None = None):
    """
    Simulate a control system with a System object.

    The system object provides a __call__ method that runs through the system
    with a reference signal provided by r_func.

    Parameters
    ----------
    system
        System instance with __call__(t, r) method that returns any number of signals
    r_func : callable
        Function that takes time t and returns the reference signal r(t)
    t_end : float
        Simulation end time (seconds)
    dt : float, optional
        Time step (seconds). If None, uses t_end / 1000
    
    Returns
    -------
    tuple of np.ndarray
        Tuple of arrays (t, signal_1, signal_2, ..., signal_n) where each array
        contains the values for that signal at each time step. The number and order
        of signals matches what system(t, r) returns.
    """
    # Create fine time vector for continuous simulation
    if dt is None:
        dt = t_end / 1000
    
    n = int(t_end / dt) + 1
    t = np.arange(n) * dt
    
    # Determine number of signals by calling system once
    r_0 = r_func(0.0)
    result_0 = system(0.0, r_0)
    if not isinstance(result_0, tuple):
        result_0 = (result_0,)
    num_signals = len(result_0)
    
    # Pre-allocate arrays for all signals
    signals = [np.zeros(n) for _ in range(num_signals)]
    
    # Reset system state
    system.reset()

    # Simulate step-by-step at continuous rate
    for i in range(n):
        t_i = t[i]

        # Get reference signal from function
        r = r_func(t_i)
        
        # Call system with time and reference signal
        result = system(t_i, r)
        
        # Handle both single value and tuple returns
        if not isinstance(result, tuple):
            result = (result,)
        
        # Store each signal value directly in pre-allocated array
        for j, value in enumerate(result):
            signals[j][i] = value

    # Return tuple: (t, signal_1, signal_2, ..., signal_n)
    return (t, *signals)
