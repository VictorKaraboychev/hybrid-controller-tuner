import numpy as np

from src.systems import PID, Block, ContinuousTF, Saturation
from src.signals import Step
from src.simulate_system import simulate_system
from src.response_metrics import compute_metrics

K2 = 0.060911271
K3 = -4.25

class OuterSystem(Block):
    def __init__(self, params: np.ndarray):
        self.pid = PID(
            Kp=params[0],
            Ki=params[1],
            Kd=params[2],
            sampling_time=params[3],
        )
        self.plant = ContinuousTF(num=[K2 * K3], den=[1.0, 0.0, 0.0])
        self.s = Saturation(min_val=-0.7, max_val=0.7)
    
    def reset(self):
        self.pid.reset()
        self.plant.reset()
        self.s.reset()
    
    def __call__(self, t: float, r: float):
        e = r - self.plant.y
        u = self.pid(t, e)
        u = self.s(t, u)
        y = self.plant(t, u)
        return r, y, e, u
    
    def cost(self) -> float:
        results = simulate_system(self, Step(amplitude=0.15), 15.0, dt_mode="variable")
        metrics = compute_metrics(results)
        return metrics["tracking_error"]