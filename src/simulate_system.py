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
        
        # Growth/decay factors for timestep adjustment - be very aggressive
        growth_factor = 2.5  # Increase dt by 150% when signals are slow (very aggressive)
        decay_factor = 0.95  # Decrease dt by only 5% when signals are fast (minimal decrease)
        stable_threshold = 3  # Number of stable steps before increasing dt (very low)
        
        # Only check adaptation every N steps to reduce overhead
        adaptation_check_interval = 3  # Check more frequently but with less overhead
        step_count = 0
        
        # Running average of signal changes for smoother adaptation
        change_history = []
        history_size = 5  # Smaller history for faster adaptation
        
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
            
            # Update timestep based on signal changes (only check periodically)
            step_count += 1
            if prev_signals is not None and step_count >= adaptation_check_interval:
                step_count = 0  # Reset counter
                
                # Compute maximum relative change across all signals
                # Use relative change (normalized by signal magnitude) to handle different scales
                max_relative_change = 0.0
                
                for j in range(num_signals):
                    signal_prev = prev_signals[j]
                    signal_curr = result[j]
                    
                    # Compute relative change (fractional change per unit time)
                    signal_magnitude = max(abs(signal_prev), abs(signal_curr), 1e-10)
                    absolute_change = abs(signal_curr - signal_prev)
                    # Normalize by magnitude and time to get rate of change
                    relative_change = (absolute_change / signal_magnitude) / prev_dt
                    
                    max_relative_change = max(max_relative_change, relative_change)
                
                # Store in history for smoother adaptation
                change_history.append(max_relative_change)
                if len(change_history) > history_size:
                    change_history.pop(0)
                
                # Use average of recent changes for more stable adaptation
                avg_change = np.mean(change_history) if len(change_history) >= history_size else 0.0
                
                # Only adapt if we have enough history (but start adapting sooner)
                if len(change_history) >= max(3, history_size // 2):
                    # Very lenient thresholds - be aggressive about increasing dt
                    # Only decrease dt for truly extreme changes
                    if avg_change < adaptive_tolerance * 0.5:
                        # Stable - count stable steps aggressively
                        stable_steps += adaptation_check_interval * 2  # Count faster
                        if stable_steps >= stable_threshold:
                            # Been stable - increase dt aggressively
                            new_dt = current_dt * growth_factor
                            current_dt = min(new_dt, max_dt)
                            stable_steps = 0  # Reset counter
                    elif avg_change > adaptive_tolerance * 100.0:
                        # Extremely rapid change - decrease dt minimally
                        new_dt = current_dt * decay_factor
                        current_dt = max(new_dt, min_dt)
                        stable_steps = 0  # Reset counter
                    else:
                        # Moderate change - be lenient, allow gradual increase
                        if avg_change < adaptive_tolerance * 2:
                            # Moderately stable - count towards increase
                            stable_steps += 1
                            if stable_steps >= stable_threshold * 2:
                                # Been moderately stable - increase dt
                                new_dt = current_dt * 1.5
                                current_dt = min(new_dt, max_dt)
                                stable_steps = 0
                        # For other cases, just maintain - don't decrease
            
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
