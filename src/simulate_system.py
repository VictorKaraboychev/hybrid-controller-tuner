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
    list of tuples
        List of tuples where each tuple contains (t, signal_1, signal_2, ..., signal_n)
        for each time step. The number and order of signals matches what system(t, r) returns.
    """
    # Create fine time vector for continuous simulation
    if dt is None:
        dt = t_end / 1000
    
    n = int(t_end / dt) + 1
    t = np.arange(n) * dt
    
    results = []

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
        
        # Create tuple with time and all signal values
        results.append((t_i, *result))

    return results
