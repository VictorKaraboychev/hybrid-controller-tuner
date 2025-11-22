"""
Signal generators for control system simulation.

This module provides various signal classes that can be used as reference inputs
for control systems. All signals extend the Signal abstract base class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Signal(ABC):
    """
    Abstract base class for signal generators.
    
    All signal classes must implement __call__(t) to return the signal value at time t.
    """
    
    @abstractmethod
    def __call__(self, t: float) -> float:
        """
        Get the signal value at time t.
        
        Parameters
        ----------
        t : float
            Current time (seconds)
        
        Returns
        -------
        float
            Signal value at time t
        """
        pass


class Step(Signal):
    """
    Step signal: constant value after t >= t0.
    
    Parameters
    ----------
    amplitude : float
        Step amplitude (default: 1.0)
    t0 : float
        Step start time (default: 0.0)
    """
    
    def __init__(self, amplitude: float = 1.0, t0: float = 0.0):
        self.amplitude = amplitude
        self.t0 = t0
    
    def __call__(self, t: float) -> float:
        return self.amplitude if t >= self.t0 else 0.0


class Ramp(Signal):
    """
    Ramp signal: linear increase starting at t0.
    
    Parameters
    ----------
    slope : float
        Ramp slope (units per second) (default: 1.0)
    t0 : float
        Ramp start time (default: 0.0)
    """
    
    def __init__(self, slope: float = 1.0, t0: float = 0.0):
        self.slope = slope
        self.t0 = t0
    
    def __call__(self, t: float) -> float:
        if t < self.t0:
            return 0.0
        return self.slope * (t - self.t0)


class Sinusoid(Signal):
    """
    Sinusoidal signal: A * sin(2*pi*f*t + phase).
    
    Parameters
    ----------
    amplitude : float
        Signal amplitude (default: 1.0)
    frequency : float
        Frequency in Hz (default: 1.0)
    phase : float
        Phase offset in radians (default: 0.0)
    """
    
    def __init__(self, amplitude: float = 1.0, frequency: float = 1.0, phase: float = 0.0):
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
    
    def __call__(self, t: float) -> float:
        return self.amplitude * np.sin(2 * np.pi * self.frequency * t + self.phase)


class SquareWave(Signal):
    """
    Square wave signal.
    
    Parameters
    ----------
    amplitude : float
        Signal amplitude (default: 1.0)
    frequency : float
        Frequency in Hz (default: 1.0)
    duty_cycle : float
        Duty cycle (0.0 to 1.0) (default: 0.5)
    phase : float
        Phase offset in seconds (default: 0.0)
    """
    
    def __init__(self, amplitude: float = 1.0, frequency: float = 1.0, 
                 duty_cycle: float = 0.5, phase: float = 0.0):
        self.amplitude = amplitude
        self.frequency = frequency
        self.duty_cycle = duty_cycle
        self.phase = phase
        self.period = 1.0 / frequency
    
    def __call__(self, t: float) -> float:
        t_shifted = t + self.phase
        t_in_period = (t_shifted % self.period) / self.period
        return self.amplitude if t_in_period < self.duty_cycle else -self.amplitude


class Constant(Signal):
    """
    Constant signal: always returns the same value.
    
    Parameters
    ----------
    value : float
        Constant value (default: 0.0)
    """
    
    def __init__(self, value: float = 0.0):
        self.value = value
    
    def __call__(self, t: float) -> float:
        return self.value

