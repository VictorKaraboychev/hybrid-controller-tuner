import numpy as np

from src.systems import PID, Block, ContinuousTF, Saturation
from src.signals import Step
from src.simulate_system import simulate_system
from src.response_metrics import compute_metrics

# Station 14
# K1 = -2.936
# TAU = 0.031

# Station 5
K1 = -2.569
TAU = 0.025

K2 = 0.060911271
K3 = -4.25

class FullSystem(Block):
    def __init__(self, params: np.ndarray):
        self.pid_outer = PID(
            Kp=params[0],
            Ki=params[1],
            Kd=params[2],
            sampling_time=params[3],
        )
        self.plant_outer = ContinuousTF(num=[K2 * K3], den=[1.0, 0.0, 0.0])
        self.s_outer = Saturation(min_val=-0.7, max_val=0.7)
        
        self.pid_inner = PID(
            Kp=params[4],
            Ki=params[5],
            Kd=params[6],
            sampling_time=params[7],
        )
        self.plant_inner = ContinuousTF(num=[K1], den=[TAU, 1.0, 0.0])
        self.s_inner = Saturation(min_val=-6.0, max_val=6.0)
    
    def reset(self):
        self.pid_inner.reset()
        self.plant_inner.reset()
        self.s_inner.reset()
        self.pid_outer.reset()
        self.plant_outer.reset()
        self.s_outer.reset()
        
    def __call__(self, t: float, r: float):
        e_outer = r - self.plant_outer.y
        u_outer = self.pid_outer(t, e_outer)
        u_outer = self.s_outer(t, u_outer)
        
        e_inner = u_outer - self.plant_inner.y
        u_inner = self.pid_inner(t, e_inner)
        u_inner = self.s_inner(t, u_inner)
        y_inner = self.plant_inner(t, u_inner)
        
        y_outer = self.plant_outer(t, y_inner)
        
        return r, y_outer, e_outer, u_outer, y_inner, e_inner, u_inner
      
    def cost(self) -> float:
        results = simulate_system(self, Step(amplitude=0.15), 5.0, dt=0.015)
        
        metrics_outer = compute_metrics(results)
        # For inner system: (t, r, y, e, u) where r=u_outer, y=y_inner, e=e_inner, u=u_inner
        inner_results = (results[0], results[4], results[5], results[6], results[7])  # (t, u_outer, y_inner, e_inner, u_inner)
        metrics_inner = compute_metrics(inner_results)
        
        # Control effort bounds penalties
        control_effort_penalty_outer = max(0.0, metrics_outer["max_control_effort"] - 0.7)
        control_effort_penalty_inner = max(0.0, metrics_inner["max_control_effort"] - 6.0)
        
        # Overshoot bounds penalties
        overshoot_penalty_outer = max(0.0, metrics_outer["percent_overshoot"] - 45.0)
        overshoot_penalty_inner = max(0.0, metrics_inner["percent_overshoot"] - 5.0)
        
        # Settling time bounds penalties
        settling_time_penalty_outer = max(0.0, metrics_outer["settling_time_2pct"] - 5.0)
        settling_time_penalty_inner = max(0.0, metrics_inner["settling_time_2pct"] - 0.25)
        
        steady_state_penalty_outer = max(0.0, abs(metrics_outer["steady_state"] - 0.15))
        tracking_error_penalty_inner = metrics_inner["tracking_error"]
        
        cost = (
					control_effort_penalty_outer +
					control_effort_penalty_inner +
					overshoot_penalty_outer +
					overshoot_penalty_inner +
					settling_time_penalty_outer +
					settling_time_penalty_inner +
					steady_state_penalty_outer
				)
        
        return cost