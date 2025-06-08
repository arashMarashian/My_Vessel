"""Utility plotting functions for simulation results."""

import matplotlib.pyplot as plt


def plot_state_trajectory(time, states, label=None):
    """Plot position and velocity over time."""
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(time, states[:, 0], label=label)
    ax[0].set_ylabel("position")
    ax[1].plot(time, states[:, 1], label=label)
    ax[1].set_ylabel("velocity")
    ax[1].set_xlabel("time [s]")
    if label:
        ax[0].legend()
    return fig, ax

