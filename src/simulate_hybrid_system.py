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

from .response_metrics import compute_step_metrics
from .plotting_utils import plot_hybrid_response


def pid_to_discrete_tf(kp, ki, kd, Ts, method='tustin'):
    """
    Convert continuous PID gains to a discrete controller D[z].

    Parameters:
    -----------
    kp : float
        Proportional gain
    ki : float
        Integral gain
    kd : float
        Derivative gain
    Ts : float
        Sampling time
    method : str
        Discretization method. Currently supports 'tustin'.

    Returns:
    --------
    tuple : (num, den) discrete-time transfer function coefficients
    """
    if method.lower() != 'tustin':
        raise ValueError("Only Tustin method is currently supported for PID discretization.")

    # Continuous PID: C(s) = kp + ki/s + kd*s
    # Using Tustin transformation: s = (2/Ts) * (z-1)/(z+1)
    # C(z) = kp + ki*Ts*(z+1)/(2*(z-1)) + kd*(2/Ts)*(z-1)/(z+1)
    # Put over common denominator (z-1)(z+1) = z^2 - 1

    b0 = kp + ki*Ts/2 + 2*kd/Ts
    b1 = ki*Ts - 4*kd/Ts
    b2 = -kp + ki*Ts/2 + 2*kd/Ts

    # Denominator: z^2 - 1 = z^2 + 0*z - 1
    a0 = 1.0
    a1 = 0.0
    a2 = -1.0

    num = np.array([b0, b1, b2], dtype=float)
    den = np.array([a0, a1, a2], dtype=float)

    return (num, den)


def simulate_hybrid_step_response(controller_tf, plant_tf, sampling_time, t_end=10.0, step_amplitude=1.0):
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
        
    Returns:
    --------
    t : array
        Time vector in seconds
    y : array
        Output response
    u : array
        Control signal (after ZOH)
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
            disc_system = signal.cont2discrete(plant_ss, Ts, method='zoh')
            # If it returns a StateSpace object
            if hasattr(disc_system, 'A'):
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
                (plant_ss.A, plant_ss.B, plant_ss.C, plant_ss.D),
                Ts,
                method='zoh'
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
    den_padded = np.pad(open_loop_den, (max_len - len(open_loop_den), 0), 'constant')
    num_padded = np.pad(open_loop_num, (max_len - len(open_loop_num), 0), 'constant')
    
    # Closed-loop denominator: den + num
    closed_loop_den = den_padded + num_padded
    closed_loop_num = open_loop_num
    
    # Remove leading zeros
    closed_loop_den = np.trim_zeros(closed_loop_den, 'f')
    closed_loop_num = np.trim_zeros(closed_loop_num, 'f')
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
        y = np.pad(y, (0, len(t) - len(y)), 'constant', constant_values=y[-1] if len(y) > 0 else 0)
    elif len(y) > len(t):
        y = y[:len(t)]
    
    # Calculate error signal e(t) = r(t) - y(t)
    r = np.ones_like(t) * step_amplitude  # Step reference
    e = r - y
    
    # Calculate control signal u[k] = D[z]e[k]
    # We need to simulate the controller with the error signal
    u = _simulate_controller_output(D_num, D_den, e, Ts)
    
    # Check for instability
    if len(y) > 50:
        last_portion = y[-min(50, len(y)//4):]
        if np.any(np.abs(last_portion) > 1e8):
            raise ValueError("System appears unstable - response growing unbounded")
        if len(last_portion) > 20:
            growth_rate = np.abs(last_portion[-1] / (last_portion[0] + 1e-10))
            if growth_rate > 1000 and np.abs(last_portion[-1]) > 1e6:
                raise ValueError("System appears unstable - exponential growth detected")
    
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


def _simulate_controller_output(controller_num, controller_den, error_signal, Ts):
    """
    Simulate controller output u[k] = D[z]e[k] given error signal.
    
    Parameters:
    -----------
    controller_num : array
        Controller numerator coefficients
    controller_den : array
        Controller denominator coefficients
    error_signal : array
        Error signal e[k] (sampled)
    Ts : float
        Sampling time
        
    Returns:
    --------
    u : array
        Controller output u[k]
    """
    # Normalize denominator
    if controller_den[0] != 1.0 and controller_den[0] != 0:
        controller_num = controller_num / controller_den[0]
        controller_den = controller_den / controller_den[0]
    
    # Reverse for difference equation
    num_rev = np.flip(controller_num)
    den_rev = np.flip(controller_den)[1:]
    
    # Initialize
    u = np.zeros(len(error_signal))
    
    # Simulate using difference equation
    for k in range(len(error_signal)):
        # Output from numerator
        u_num = 0.0
        for i in range(len(num_rev)):
            if k - i >= 0:
                u_num += num_rev[i] * error_signal[k - i]
        
        # Feedback from denominator
        u_den = 0.0
        for i in range(len(den_rev)):
            if k - i - 1 >= 0:
                u_den += den_rev[i] * u[k - i - 1]
        
        u[k] = u_num - u_den
    
    return u


if __name__ == "__main__":
    """
    Example usage
    """
    # Example: PID controller (kP, kI, kD)
    kP = -8.0
    kI = 0.0
    kD = 0.0
    sampling_time = 0.015
    # controller_tf = pid_to_discrete_tf(kP, kI, kD, sampling_time)
    
    controller_num = [-1.71826469 -0.52338036 -0.60482782]
    controller_den = [1.0, 0.35443462, -0.20391]
    controller_tf = (controller_num, controller_den)
    
    # Example plant: P(s) = -2.936 / (0.031*s^2 + s)
    plant_num = [-2.936]
    plant_den = [0.031, 1, 0]
    plant_tf = (plant_num, plant_den)
    
    # Simulation parameters
    t_end = 5.0
    step_amplitude = 1.0
    
    # Simulate
    print("Simulating hybrid system...")
    print(f"Controller gains: kP={kP}, kI={kI}, kD={kD}")
    print(f"Plant: P(s) = {plant_num[0]} / ({plant_den[0]}*s^2 + {plant_den[1]}*s)")
    print(f"Sampling time: {sampling_time} s")
    
    try:
        t, y, u, e = simulate_hybrid_step_response(
            controller_tf, plant_tf, sampling_time, t_end, step_amplitude=step_amplitude
        )
        metrics = compute_step_metrics(t, y, reference=step_amplitude)
        
        print("Simulation successful!")
        print(f"  Time samples: {len(t)}")
        print(f"  Steady-state: {metrics['steady_state']:.4f}")
        print(f"  Peak value: {metrics['peak_value']:.4f}")
        print(f"  Percent overshoot: {metrics['percent_overshoot']:.2f}%")
        if np.isfinite(metrics['settling_time_2pct']):
            print(f"  2% settling time: {metrics['settling_time_2pct']:.4f} s")
        else:
            print("  2% settling time: Not settled within simulation horizon")
        
        fig, axes = plot_hybrid_response(
            t, y, u, e, metrics, step_amplitude=step_amplitude, save_path='hybrid_system_response.png'
        )
        print("  Plot saved to 'hybrid_system_response.png'")
        fig.show()
        
    except Exception as e:
        print(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()

