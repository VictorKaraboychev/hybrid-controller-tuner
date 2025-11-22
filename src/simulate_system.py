import numpy as np

def simulate_system(
    system, 
    r_func, 
    t_end: float, 
    dt: float | None = None,
    dt_mode: str = "fixed",
    min_dt: float = 1e-5,
    max_dt: float = 0.1,
    adaptive_tolerance: float = 0.01,
):
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
        Time step (seconds). If None, uses t_end / 1000 for fixed mode or 0.001 for variable mode
    dt_mode : str, optional
        Timestep mode: "fixed" for constant timestep, "variable" for adaptive timestep.
        Default is "fixed".
    min_dt : float, optional
        Minimum timestep for variable mode (seconds). Default is 1e-6.
    max_dt : float, optional
        Maximum timestep for variable mode (seconds). Default is 0.1.
    adaptive_tolerance : float, optional
        Tolerance for adaptive timestep control. Higher values allow larger timesteps
        when signals are changing slowly. Default is 0.01.
    
    Returns
    -------
    tuple of np.ndarray
        Tuple of arrays (t, signal_1, signal_2, ..., signal_n) where each array
        contains the values for that signal at each time step. The number and order
        of signals matches what system(t, r) returns.
    """
    # Determine number of signals by calling system once
    system.reset()
    r_0 = r_func(0.0)
    result_0 = system(0.0, r_0)
    if not isinstance(result_0, tuple):
        result_0 = (result_0,)
    num_signals = len(result_0)
    
    # Set default dt if not provided
    if dt is None:
        dt = t_end / 1000 if dt_mode == "fixed" else 0.001
    
    if dt_mode == "fixed":
        # Fixed timestep mode - pre-allocate arrays
        n = int(t_end / dt) + 1
        t = np.arange(n) * dt
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
    
    else:
        # Variable/adaptive timestep mode
        # Use dynamic lists since we don't know the final size
        t_list = []
        signals_list = [[] for _ in range(num_signals)]
        
        # Reset system state
        system.reset()
        
        # Initialize adaptive timestep parameters
        current_dt = dt
        t_current = 0.0
        prev_signals = None
        prev_dt = dt
        stable_steps = 0  # Count consecutive stable steps
        
        # Growth/decay factors for timestep adjustment
        growth_factor = 1.2  # Increase dt by 20% when signals are slow
        decay_factor = 0.8   # Decrease dt by 20% when signals are fast
        stable_threshold = 10  # Number of stable steps before increasing dt
        
        while t_current < t_end:
            # Ensure we don't overshoot t_end
            if t_current + current_dt > t_end:
                current_dt = t_end - t_current
                if current_dt < min_dt:
                    break
            
            # Get reference signal from function
            r = r_func(t_current)
            
            # Call system with time and reference signal
            result = system(t_current, r)
            
            # Handle both single value and tuple returns
            if not isinstance(result, tuple):
                result = (result,)
            
            # Store current time and signals
            t_list.append(t_current)
            for j, value in enumerate(result):
                signals_list[j].append(value)
            
            # Update timestep based on signal changes (if we have previous values)
            if prev_signals is not None:
                # Compute maximum absolute change across all signals
                max_absolute_change = 0.0
                
                for j in range(num_signals):
                    signal_prev = prev_signals[j]
                    signal_curr = result[j]
                    absolute_change = abs(signal_curr - signal_prev)
                    max_absolute_change = max(max_absolute_change, absolute_change)
                
                # Use absolute change threshold (not normalized) for simplicity
                # This works better when signals have different scales
                change_threshold = adaptive_tolerance * current_dt
                
                # Adjust timestep based on change
                # Only increase dt when signals are very stable (after multiple stable steps)
                # Only decrease dt when signals are changing rapidly
                if max_absolute_change < change_threshold * 0.1:
                    # Signals are very stable - count stable steps
                    stable_steps += 1
                    if stable_steps >= stable_threshold:
                        # Been stable for a while - increase dt
                        new_dt = current_dt * growth_factor
                        current_dt = min(new_dt, max_dt)
                        stable_steps = 0  # Reset counter
                elif max_absolute_change > change_threshold * 10:
                    # Signals are changing rapidly - decrease dt immediately
                    new_dt = current_dt * decay_factor
                    current_dt = max(new_dt, min_dt)
                    stable_steps = 0  # Reset counter
                else:
                    # Moderate change - keep current dt, reset counter
                    stable_steps = 0  # Reset counter
            
            # Store current signals and timestep for next iteration
            prev_signals = tuple(result)
            prev_dt = current_dt
            
            # Advance time
            t_current += current_dt
        
        # Convert lists to numpy arrays
        t = np.array(t_list)
        signals = [np.array(sig_list) for sig_list in signals_list]
        
        # Return tuple: (t, signal_1, signal_2, ..., signal_n)
        return (t, *signals)
