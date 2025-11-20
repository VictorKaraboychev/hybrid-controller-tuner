"""
System Building Blocks

This module provides reusable blocks for building control systems:
- ContinuousTF: Continuous-time transfer function (s-domain)
- DiscreteTF: Discrete-time transfer function (z-domain) with zero-order hold
- Saturation: Signal saturation/limiting block
"""

import numpy as np
from scipy import signal


class ContinuousTF:
    """
    Continuous-time transfer function block.
    
    Represents a transfer function in the s-domain: G(s) = num(s) / den(s)
    where num and den are polynomial coefficients in descending powers of s.
    
    This block maintains internal state-space representation for step-by-step
    simulation. For continuous-time systems, the output is computed by integrating
    the state-space equations.
    
    Parameters
    ----------
    num : array-like
        Numerator coefficients in descending powers of s.
        Example: [1, 2] represents s + 2
    den : array-like
        Denominator coefficients in descending powers of s.
        Example: [1, 3, 2] represents s^2 + 3*s + 2
    dt : float
        Integration time step for continuous-time simulation (seconds)
    """
    
    def __init__(self, num, den, dt=0.001):
        self.num = np.array(num, dtype=float)
        self.den = np.array(den, dtype=float)
        self.dt = dt
        
        # Convert to state-space for step-by-step simulation
        try:
            ss_result = signal.tf2ss(self.num, self.den)
            
            # Handle both tuple (A, B, C, D) and StateSpace object returns
            if isinstance(ss_result, tuple):
                # Newer scipy versions return (A, B, C, D) tuple
                A, B, C, D = ss_result
                self.A = np.array(A)
                self.B = np.array(B)
                self.C = np.array(C)
                self.D = np.array(D)
            else:
                # Older scipy versions return StateSpace object
                self.A = np.array(ss_result.A)
                self.B = np.array(ss_result.B)
                self.C = np.array(ss_result.C)
                self.D = np.array(ss_result.D)
            
            self.state = np.zeros((self.A.shape[0],))
            self.last_input = 0.0
            self.y = 0.0  # Last output value
        except Exception as e:
            raise ValueError(f"Failed to convert transfer function to state-space: {e}")
    
    def get_output(self, u):
        """
        Get current output without stepping the state.
        
        Computes output from current state: y = C*x + D*u
        
        Parameters
        ----------
        u : float
            Current input signal
            
        Returns
        -------
        y : float
            Current output signal
        """
        # Compute output: y = C*x + D*u
        y = self.C @ self.state
        if self.D.size > 0 and np.any(self.D != 0):
            if self.D.ndim > 1:
                y = y + self.D @ np.array([u])
            else:
                y = y + self.D * u
        
        return y.flatten()[0] if y.size > 0 else 0.0
    
    def step(self, u):
        """
        Step the continuous-time transfer function with input u.
        
        Uses numerical integration (Euler method) to update the state.
        For a state-space system: dx/dt = A*x + B*u, y = C*x + D*u
        
        Parameters
        ----------
        u : float
            Input signal at current time step
            
        Returns
        -------
        y : float
            Output signal at current time step (after stepping)
        """
        # Get output before stepping (for the current state)
        y = self.get_output(u)
        
        # Update state using Euler integration
        # x[k+1] = x[k] + dt * (A*x[k] + B*u)
        B_flat = self.B.flatten() if self.B.ndim > 1 else self.B
        dx = self.A @ self.state + B_flat * u
        self.state = self.state + self.dt * dx
        
        self.last_input = u
        self.y = y  # Store last output
        return y
    
    def reset(self):
        """Reset the internal state to zero."""
        self.state = np.zeros((self.A.shape[0],))
        self.last_input = 0.0
        self.y = 0.0


class DiscreteTF:
    """
    Discrete-time transfer function block with zero-order hold.
    
    Represents a transfer function in the z-domain: G(z) = num(z) / den(z)
    where num and den are polynomial coefficients in descending powers of z.
    
    This block automatically applies zero-order hold (ZOH) internally. The ZOH
    is applied by discretizing the transfer function using the sampling time.
    
    Uses scipy's dlti for reliable difference equation implementation.
    
    Parameters
    ----------
    num : array-like
        Numerator coefficients in descending powers of z.
        Example: [1, 0.5] represents z + 0.5
    den : array-like
        Denominator coefficients in descending powers of z.
        Example: [1, -0.5, 0.2] represents z^2 - 0.5*z + 0.2
    sampling_time : float
        Sampling time Ts in seconds. This is used for ZOH discretization
        if the transfer function is originally continuous.
    """
    
    def __init__(self, num, den, sampling_time):
        self.num = np.array(num, dtype=float)
        self.den = np.array(den, dtype=float)
        self.sampling_time = sampling_time
        
        # Normalize denominator (leading coefficient should be 1.0)
        if self.den[0] != 1.0 and self.den[0] != 0:
            self.num = self.num / self.den[0]
            self.den = self.den / self.den[0]
        
        # Convert from descending powers of z to scipy format
        # Scipy's dlti expects coefficients in ascending powers of z^-1
        # When degrees match: dividing by z gives same coefficient order
        # Example: (z + 0.5)/(z - 0.5) = (1 + 0.5*z^-1)/(1 - 0.5*z^-1)
        # So [1, 0.5] in descending z is [1, 0.5] in ascending z^-1 (same order!)
        # For strictly proper systems, scipy handles them correctly without padding
        
        # Pass coefficients directly to scipy (no padding needed)
        # Scipy will handle strictly proper systems correctly
        num_scipy = self.num
        den_scipy = self.den
        
        # Create scipy discrete-time system
        self.sys = signal.dlti(num_scipy, den_scipy, dt=sampling_time)
        
        # Get state-space representation for step-by-step simulation
        # scipy's dlti.to_ss() gives us the state-space form
        ss = self.sys.to_ss()
        self.A = ss.A
        self.B = ss.B
        self.C = ss.C
        self.D = ss.D
        
        # Initialize state for state-space simulation
        self.state = np.zeros((self.A.shape[0],))
        
        # For sampling control
        self.last_sample_time = -np.inf  # Initialize to allow first sample
        self.last_output = 0.0  # ZOH output between samples
        self.y = 0.0  # Last output value (accessible via tf.y)
    
    def step(self, t, u):
        """
        Step the discrete-time transfer function with input u.
        
        This method handles discrete sampling automatically. It only updates the controller
        state at discrete sample times (every sampling_time seconds). Between samples,
        it returns the last output (zero-order hold behavior).
        
        Uses scipy's state-space representation for accurate difference equation computation.
        
        Parameters
        ----------
        t : float
            Current time (seconds)
        u : float
            Input signal at current time
            
        Returns
        -------
        y : float
            Output signal (updated at sample times, held between samples via ZOH)
        """
        # Check if it's time to sample
        if t - self.last_sample_time >= self.sampling_time:
            # Time to sample - update controller state using state-space form
            # For discrete state-space: x[k+1] = A*x[k] + B*u[k], y[k] = C*x[k] + D*u[k]
            # Note: output is computed from current state x[k], then state is updated to x[k+1]
            
            # Compute output from current state (before updating)
            # y[k] = C*x[k] + D*u[k]
            y = (self.C @ self.state).flatten()
            if self.D.size > 0 and np.any(self.D != 0):
                u_vec = np.array([[u]]) if np.isscalar(u) else np.array([u]).reshape(-1, 1)
                y = y + (self.D @ u_vec).flatten()
            y = float(y[0]) if y.size > 0 else 0.0
            
            # Update state for next sample: x[k+1] = A*x[k] + B*u[k]
            if self.B.ndim == 1:
                # B is a 1D array (column vector)
                self.state = self.A @ self.state + self.B * u
            elif self.B.ndim == 2:
                # B is a 2D matrix
                if self.B.shape[1] == 1:
                    # B is a column vector (n x 1)
                    self.state = self.A @ self.state + (self.B * u).flatten()
                else:
                    # B is a matrix (n x m)
                    u_vec = np.array([u]) if np.isscalar(u) else u
                    self.state = self.A @ self.state + (self.B @ u_vec).flatten()
            else:
                # Fallback
                self.state = self.A @ self.state + self.B * u
            
            # Store output and update sample time
            self.last_output = y
            self.y = y  # Store last output (accessible via tf.y)
            self.last_sample_time = t
            return y
        else:
            # Between samples - return last output (ZOH)
            return self.last_output
    
    def reset(self):
        """Reset the internal state to zero."""
        self.state = np.zeros((self.A.shape[0],))
        self.last_sample_time = -np.inf
        self.last_output = 0.0
        self.y = 0.0


class Saturation:
    """
    Saturation/limiting block.
    
    Constrains a signal between minimum and maximum values.
    
    Parameters
    ----------
    min_val : float or None
        Minimum output value. If None, no lower limit is applied.
    max_val : float or None
        Maximum output value. If None, no upper limit is applied.
    """
    
    def __init__(self, min_val=None, max_val=None):
        self.min_val = min_val
        self.max_val = max_val
        
        # Validate that min <= max if both are provided
        if min_val is not None and max_val is not None:
            if min_val > max_val:
                raise ValueError(f"min_val ({min_val}) must be <= max_val ({max_val})")
    
    def step(self, u):
        """
        Apply saturation to input signal.
        
        Parameters
        ----------
        u : float
            Input signal
            
        Returns
        -------
        y : float
            Saturated output signal
        """
        y = u
        
        if self.min_val is not None:
            y = max(self.min_val, y)
        if self.max_val is not None:
            y = min(self.max_val, y)
        
        return y
    
    def reset(self):
        """Reset the block (no-op for saturation, but included for consistency)."""
        pass

