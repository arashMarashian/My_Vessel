"""Simple planing hull dynamic model.

This module provides a minimal representation of a high-speed planing hull in
pitch and heave. It is intended as a starting point for more advanced
hydrodynamic models.
"""

import numpy as np

class PlaningHullModel:
    """Planing hull model with basic state propagation."""

    def __init__(self, mass: float, added_mass: float = 0.0):
        self.mass = mass
        self.added_mass = added_mass

    @property
    def total_mass(self) -> float:
        return self.mass + self.added_mass

    def step(self, state: np.ndarray, force: float, dt: float) -> np.ndarray:
        """Advance the state vector using a simple Euler step.

        Parameters
        ----------
        state : np.ndarray
            Current [position, velocity].
        force : float
            External force in the vertical direction (N).
        dt : float
            Time step in seconds.
        """
        position, velocity = state
        acceleration = force / self.total_mass
        velocity = velocity + acceleration * dt
        position = position + velocity * dt
        return np.array([position, velocity])


