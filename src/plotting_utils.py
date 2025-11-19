"""
Plotting utilities for visualizing hybrid control system responses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_hybrid_response(
    t: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    e: np.ndarray,
    metrics: dict,
    step_amplitude: float = 1.0,
    save_path: str | None = None,
) -> tuple[Figure, list[Axes]]:
    """
    Plot output, control, and error signals with 2% settling band and peak annotations.

    Parameters
    ----------
    t : np.ndarray
        Time vector in seconds
    y : np.ndarray
        Output response
    u : np.ndarray
        Control signal (after ZOH)
    e : np.ndarray
        Error signal e(t) = r(t) - y(t)
    metrics : dict
        Dictionary containing step response metrics (from compute_step_metrics)
    step_amplitude : float, optional
        Amplitude of step input (default: 1.0)
    save_path : str | None, optional
        Path to save the figure (default: None, don't save)

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    axes : list of matplotlib.axes.Axes
        List of axes objects
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    steady = metrics["steady_state"]
    band = 0.02 * max(abs(steady), 1e-9)
    peak_idx = np.argmax(y)
    settling_time = metrics["settling_time_2pct"]

    axes[0].plot(t, y, "b-", linewidth=2, label="Output y(t)")
    axes[0].axhline(step_amplitude, color="k", linestyle=":", alpha=0.7, label="Reference")
    axes[0].axhline(steady, color="r", linestyle="--", alpha=0.6, label="Steady State")
    axes[0].fill_between(
        t, steady - band, steady + band, color="orange", alpha=0.15, label="±2% Band"
    )
    axes[0].plot(t[peak_idx], y[peak_idx], "mo", label="Peak")
    axes[0].annotate(
        f"Peak {y[peak_idx]:.3f}",
        xy=(t[peak_idx], y[peak_idx]),
        xytext=(t[peak_idx], y[peak_idx] + 0.05 * step_amplitude),
        arrowprops=dict(arrowstyle="->", color="m"),
        fontsize=9,
    )
    if np.isfinite(settling_time):
        axes[0].axvline(settling_time, color="g", linestyle="-.", label="2% Settling Time")
        axes[0].annotate(
            f"Ts = {settling_time:.3f}s",
            xy=(settling_time, steady),
            xytext=(settling_time, steady + 0.1 * step_amplitude),
            arrowprops=dict(arrowstyle="->", color="g"),
            fontsize=9,
        )
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Output")
    axes[0].set_title("Step Response - Output")
    axes[0].legend(loc="best")

    axes[1].plot(t, u, "g-", linewidth=2, label="Control u[k] (after ZOH)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Control Signal")
    axes[1].set_title("Control Signal")
    axes[1].legend()

    axes[2].plot(t, e, "r-", linewidth=2, label="Error e(t) = r(t) - y(t)")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Error")
    axes[2].set_title("Error Signal")
    axes[2].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, axes

