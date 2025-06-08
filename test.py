from models.planing_hull import PlaningHullModel
from controllers.feedback_linearization import FeedbackLinearizationController
import numpy as np

if __name__ == "__main__":
    model = PlaningHullModel(mass=1000.0, added_mass=200.0)
    controller = FeedbackLinearizationController(kp=10.0, kd=5.0)

    dt = 0.1
    time = np.arange(0, 5, dt)
    state = np.array([0.0, 0.0])
    states = []

    for t in time:
        control_force = controller.control(state, desired=np.array([1.0, 0.0]))
        state = model.step(state, force=float(control_force), dt=dt)
        states.append(state)

    states = np.array(states)
    print(states[-1])

