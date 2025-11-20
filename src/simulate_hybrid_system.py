"""
Hybrid System Simulator

This module simulates a hybrid control system with:
- Discrete-time controller D[z]
- Continuous-time plant P(s)
- Sampler and Zero-Order Hold (ZOH)

System structure:
  e(t) = r(t) - y(t)  [continuous error]
  e[k] = e(k*Ts)      [sampled error]
  u[k] = D[z]e[k]     [discrete controller output]
  u(t) = ZOH(u[k])    [zero-order hold]
  y(t) = P(s)u(t)     [continuous plant output]
"""

import numpy as np
from scipy import signal

def simulate_hybrid_step_response(
    controller_tf,
    plant_tf,
    sampling_time,
    t_end=10.0,
    step_amplitude=1.0,
    u_min=None,
    u_max=None,
):
    """
    Simulate step response of a hybrid control system.

    Parameters:
    -----------
    controller_tf : tuple
        Discrete-time controller transfer function as (num, den) where num and den
        are arrays of coefficients in descending order of powers of z.
        Example: D[z] = (b0*z^2 + b1*z + b2) / (a0*z^2 + a1*z + a2)
    plant_tf : tuple
        Continuous-time plant transfer function as (num, den) where num and den
        are arrays of coefficients in descending order of powers of s.
        Example: P(s) = (c0*s + c1) / (d0*s^2 + d1*s + d2)
    sampling_time : float
        Sampling time Ts in seconds
    t_end : float, optional
        End time for simulation in seconds (default: 10.0)
    step_amplitude : float, optional
        Amplitude of step input (default: 1.0)
    u_min : float, optional
        Minimum control signal constraint (saturation lower limit). If None, no lower limit.
    u_max : float, optional
        Maximum control signal constraint (saturation upper limit). If None, no upper limit.

    Returns:
    --------
    t : array
        Time vector in seconds
    y : array
        Output response
    u : array
        Control signal (after ZOH and saturation)
    e : array
        Error signal e(t) = r(t) - y(t)
    """
    Ts = sampling_time

    # Extract controller and plant transfer functions
    D_num, D_den = controller_tf
    P_num, P_den = plant_tf

    # Discretize plant P(s) using Zero-Order Hold (ZOH)
    # P[z] = ZOH{P(s)}
    try:
        # Convert continuous plant to state-space first, then discretize
        plant_ss = signal.tf2ss(P_num, P_den)

        # Discretize state-space using ZOH
        # cont2discrete can take either a StateSpace object or (A, B, C, D) tuple
        # Try passing the StateSpace object directly first
        try:
            disc_system = signal.cont2discrete(plant_ss, Ts, method="zoh")
            # If it returns a StateSpace object
            if hasattr(disc_system, "A"):
                A_d = disc_system.A
                B_d = disc_system.B
                C_d = disc_system.C
                D_d = disc_system.D
            else:
                # It's a tuple (A, B, C, D, dt)
                A_d, B_d, C_d, D_d = disc_system[:4]
        except (TypeError, AttributeError):
            # Fallback: pass matrices as tuple
            disc_system = signal.cont2discrete(
                (plant_ss.A, plant_ss.B, plant_ss.C, plant_ss.D), Ts, method="zoh"
            )
            # Extract from tuple (A, B, C, D, dt)
            A_d, B_d, C_d, D_d = disc_system[:4]

        # Convert back to transfer function
        P_disc_num, P_disc_den = signal.ss2tf(A_d, B_d, C_d, D_d)

        # Handle 2D arrays if needed
        if P_disc_num.ndim > 1:
            P_disc_num = P_disc_num[0]
        if P_disc_den.ndim > 1:
            P_disc_den = P_disc_den[0]

        # Remove very small coefficients that might cause numerical issues
        P_disc_num = np.array(P_disc_num)
        P_disc_den = np.array(P_disc_den)
        P_disc_num[np.abs(P_disc_num) < 1e-12] = 0
        P_disc_den[np.abs(P_disc_den) < 1e-12] = 0

    except Exception as e:
        raise ValueError(f"Failed to discretize plant: {e}")

    # Form closed-loop in z-domain: Y[z]/R[z] = D[z]P[z] / (1 + D[z]P[z])
    # Open-loop: G_ol[z] = D[z] * P[z]
    open_loop_num = np.convolve(D_num, P_disc_num)
    open_loop_den = np.convolve(D_den, P_disc_den)

    # Closed-loop: G_cl[z] = G_ol[z] / (1 + G_ol[z])
    # = (num/den) / (1 + num/den) = (num/den) / ((den + num)/den) = num / (den + num)
    # Pad polynomials to same length for addition
    max_len = max(len(open_loop_den), len(open_loop_num))
    den_padded = np.pad(open_loop_den, (max_len - len(open_loop_den), 0), "constant")
    num_padded = np.pad(open_loop_num, (max_len - len(open_loop_num), 0), "constant")

    # Closed-loop denominator: den + num
    closed_loop_den = den_padded + num_padded
    closed_loop_num = open_loop_num

    # Remove leading zeros
    closed_loop_den = np.trim_zeros(closed_loop_den, "f")
    closed_loop_num = np.trim_zeros(closed_loop_num, "f")
    if len(closed_loop_den) == 0:
        closed_loop_den = [1]
    if len(closed_loop_num) == 0:
        closed_loop_num = [0]

    # Normalize denominator to avoid numerical issues
    if closed_loop_den[0] != 1.0 and closed_loop_den[0] != 0:
        closed_loop_num = closed_loop_num / closed_loop_den[0]
        closed_loop_den = closed_loop_den / closed_loop_den[0]

    # Create discrete time vector
    n_samples = int(t_end / Ts) + 1
    t = np.arange(n_samples) * Ts

    # Simulate discrete step response
    try:
        # Try using scipy's dstep
        _, y_disc = signal.dstep((closed_loop_num, closed_loop_den, Ts), n=n_samples)
        y = y_disc[0].flatten() if y_disc[0].ndim > 1 else y_disc[0]
    except:
        # Fallback: manual simulation using difference equation
        y = _simulate_discrete_step(closed_loop_num, closed_loop_den, n_samples)

    # Scale by step amplitude
    y = y * step_amplitude

    # Ensure arrays have same length
    if len(y) < len(t):
        y = np.pad(
            y,
            (0, len(t) - len(y)),
            "constant",
            constant_values=y[-1] if len(y) > 0 else 0,
        )
    elif len(y) > len(t):
        y = y[: len(t)]

    # Always use step-by-step simulation to properly compute u[k] in the feedback loop
    # This ensures the control signal is computed correctly and affects the plant output
    t, y, u, e = _simulate_step_by_step(
        controller_tf, plant_tf, sampling_time, t_end, step_amplitude, u_min, u_max
    )

    # Check for instability
    if len(y) > 50:
        last_portion = y[-min(50, len(y) // 4) :]
        if np.any(np.abs(last_portion) > 1e8):
            raise ValueError("System appears unstable - response growing unbounded")
        if len(last_portion) > 20:
            growth_rate = np.abs(last_portion[-1] / (last_portion[0] + 1e-10))
            if growth_rate > 1000 and np.abs(last_portion[-1]) > 1e6:
                raise ValueError(
                    "System appears unstable - exponential growth detected"
                )

    return t, y, u, e


def _simulate_discrete_step(num, den, n_samples):
    """
    Manually simulate discrete-time step response using difference equation.

    Parameters:
    -----------
    num : array
        Numerator coefficients (descending powers of z)
    den : array
        Denominator coefficients (descending powers of z)
    n_samples : int
        Number of samples to simulate

    Returns:
    --------
    y : array
        Output response
    """
    # Normalize denominator (make leading coefficient 1)
    if den[0] != 1.0 and den[0] != 0:
        num = num / den[0]
        den = den / den[0]

    # Reverse for difference equation (y[k] = ...)
    num_rev = np.flip(num)
    den_rev = np.flip(den)[1:]  # Skip a0 (should be 1)

    # Initialize
    y = np.zeros(n_samples)
    u = np.ones(n_samples)  # Step input

    # Simulate using difference equation
    # y[k] = sum(bi*u[k-i]) - sum(ai*y[k-i])
    for k in range(n_samples):
        # Output from numerator
        y_num = 0.0
        for i in range(len(num_rev)):
            if k - i >= 0:
                y_num += num_rev[i] * u[k - i]

        # Feedback from denominator
        y_den = 0.0
        for i in range(len(den_rev)):
            if k - i - 1 >= 0:
                y_den += den_rev[i] * y[k - i - 1]

        y[k] = y_num - y_den

        # Early termination if growing unbounded
        if k > 10 and abs(y[k]) > 1e6:
            y[k:] = np.nan
            break

    return y


def _simulate_step_by_step(
    controller_tf, plant_tf, sampling_time, t_end, step_amplitude, u_min=None, u_max=None
):
    """
    Simulate hybrid system step-by-step to properly compute control signal in feedback loop.
    This ensures u[k] is computed and applied correctly, with optional saturation constraints.

    Parameters:
    -----------
    controller_tf : tuple
        Discrete-time controller transfer function (num, den)
    plant_tf : tuple
        Continuous-time plant transfer function (num, den)
    sampling_time : float
        Sampling time Ts
    t_end : float
        End time for simulation
    step_amplitude : float
        Step input amplitude
    u_min : float or None
        Minimum control signal limit
    u_max : float or None
        Maximum control signal limit

    Returns:
    --------
    t : array
        Time vector
    y : array
        Output response
    u : array
        Control signal (saturated)
    e : array
        Error signal
    """
    Ts = sampling_time
    D_num, D_den = controller_tf
    P_num, P_den = plant_tf

    # Discretize plant using ZOH
    try:
        plant_ss = signal.tf2ss(P_num, P_den)
        try:
            disc_system = signal.cont2discrete(plant_ss, Ts, method="zoh")
            if hasattr(disc_system, "A"):
                A_d = disc_system.A
                B_d = disc_system.B
                C_d = disc_system.C
                D_d = disc_system.D
            else:
                A_d, B_d, C_d, D_d = disc_system[:4]
        except (TypeError, AttributeError):
            disc_system = signal.cont2discrete(
                (plant_ss.A, plant_ss.B, plant_ss.C, plant_ss.D), Ts, method="zoh"
            )
            A_d, B_d, C_d, D_d = disc_system[:4]
    except Exception as e:
        raise ValueError(f"Failed to discretize plant: {e}")

    # Create discrete time vector
    n_samples = int(t_end / Ts) + 1
    t = np.arange(n_samples) * Ts

    # Initialize arrays
    y = np.zeros(n_samples)
    u = np.zeros(n_samples)
    e = np.zeros(n_samples)
    r = np.ones(n_samples) * step_amplitude

    # Initialize controller state (for difference equation)
    # Normalize controller denominator
    D_num = np.array(D_num, dtype=float)
    D_den = np.array(D_den, dtype=float)
    if D_den[0] != 1.0 and D_den[0] != 0:
        D_num = D_num / D_den[0]
        D_den = D_den / D_den[0]

    # For difference equation: D[z] = num(z)/den(z)
    # If num = [b0, b1, b2, ...] and den = [a0=1, a1, a2, ...] (descending powers)
    # Then: u[k] = b0*e[k] + b1*e[k-1] + ... - a1*u[k-1] - a2*u[k-2] - ...
    # We need to store past values: e[k], e[k-1], ... and u[k-1], u[k-2], ...
    # No need to flip - use coefficients directly with proper indexing
    
    # Controller state: past inputs (errors) and outputs (control signals)
    # For numerator of order n: need e[k], e[k-1], ..., e[k-n]
    # For denominator of order m: need u[k-1], u[k-2], ..., u[k-m] (skip u[k] since a0=1)
    num_order = len(D_num) - 1
    den_order = len(D_den) - 1
    controller_input_history = np.zeros(num_order + 1)  # e[k], e[k-1], ..., e[k-n]
    controller_output_history = np.zeros(den_order)  # u[k-1], u[k-2], ..., u[k-m]

    # Plant state (state-space)
    x_plant = np.zeros((A_d.shape[0],))

    # Simulate step-by-step
    # Standard discrete-time control loop:
    # At time k: measure y[k], compute e[k], compute u[k], apply u[k] to get x[k+1]
    # For the first iteration: y[0] = C*x[0] = 0, e[0] = r[0], u[0] = controller(e[0])
    
    for k in range(n_samples):
        # At time k: we have state x[k] from previous iteration (or initial state x[0]=0)
        # Calculate output: y[k] = C*x[k] + D*u[k-1]
        # For k=0: u[-1] = 0, so y[0] = C*x[0] = 0
        # For k>0: use u[k-1] from previous iteration
        u_prev = u[k-1] if k > 0 else 0.0
        y_plant = C_d @ x_plant
        if D_d.size > 0 and np.any(D_d != 0):
            if D_d.ndim > 1:
                y_plant = y_plant + D_d @ np.array([u_prev])
            else:
                y_plant = y_plant + D_d * u_prev
        y[k] = y_plant.flatten()[0] if y_plant.size > 0 else 0.0

        # Calculate error: e[k] = r[k] - y[k]
        e[k] = r[k] - y[k]

        # Update controller input history (shift and add new error)
        # Shift: e[k-1] -> e[k-2], e[k-2] -> e[k-3], etc.
        controller_input_history = np.roll(controller_input_history, 1)
        controller_input_history[0] = e[k]  # Most recent error at index 0

        # Compute controller output: u[k] = D[z]e[k]
        # u[k] = b0*e[k] + b1*e[k-1] + ... - a1*u[k-1] - a2*u[k-2] - ...
        # D_num = [b0, b1, b2, ...] (descending powers)
        # controller_input_history = [e[k], e[k-1], e[k-2], ...]
        u_num = np.sum(D_num * controller_input_history[: len(D_num)])
        
        # D_den = [1, a1, a2, ...] (skip leading 1, use a1, a2, ...)
        # controller_output_history = [u[k-1], u[k-2], ...]
        if len(D_den) > 1:
            u_den = np.sum(D_den[1:] * controller_output_history[: len(D_den) - 1])
        else:
            u_den = 0.0
        u_unsat = u_num - u_den

        # Apply saturation
        if u_min is not None:
            u_unsat = max(u_min, u_unsat)
        if u_max is not None:
            u_unsat = min(u_max, u_unsat)
        u[k] = u_unsat

        # Update controller output history
        controller_output_history = np.roll(controller_output_history, 1)
        controller_output_history[0] = u[k]

        # Update state for next iteration: x[k+1] = A*x[k] + B*u[k]
        # Handle B_d shape (could be 2D or 1D)
        if B_d.ndim > 1:
            u_input = B_d @ np.array([u[k]])
        else:
            u_input = B_d * u[k]
        x_plant = A_d @ x_plant + u_input

        # Check for instability
        if k > 10 and (np.abs(y[k]) > 1e8 or np.abs(u[k]) > 1e8):
            raise ValueError("System appears unstable - response growing unbounded")

    return t, y, u, e
