"""
Plotting utilities for visualizing hybrid control system responses.
"""

from __future__ import annotations

from pathlib import Path
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
    controller_tf: tuple[np.ndarray, np.ndarray] | None = None,
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
    controller_tf : tuple of np.ndarray, optional
        Controller transfer function (num, den) for plotting poles and zeros

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    axes : list of matplotlib.axes.Axes
        List of axes objects
    """
    import matplotlib.pyplot as plt

    # Create 4 subplots if controller_tf is provided, otherwise 3
    n_subplots = 4 if controller_tf is not None else 3
    fig, axes = plt.subplots(n_subplots, 1, figsize=(10, 3.5 * n_subplots))

    steady = metrics["steady_state"]
    band = 0.02 * max(abs(steady), 1e-9)
    peak_idx = np.argmax(y)
    settling_time = metrics["settling_time_2pct"]

    axes[0].plot(t, y, "b-", linewidth=2, label="Output y(t)")
    axes[0].axhline(
        step_amplitude, color="k", linestyle=":", alpha=0.7, label="Reference"
    )
    axes[0].axhline(steady, color="r", linestyle="--", alpha=0.6, label="Steady State")
    axes[0].fill_between(
        t, steady - band, steady + band, color="orange", alpha=0.15, label="±2% Band"
    )
    axes[0].plot(t[peak_idx], y[peak_idx], "mo", label="Peak")
    axes[0].annotate(
        f"Peak {y[peak_idx]:.3f}",
        xy=(t[peak_idx], y[peak_idx]),
        xytext=(1.1 * t[peak_idx], 0.9 * y[peak_idx]),
        arrowprops=dict(arrowstyle="->", color="m"),
        fontsize=9,
    )
    if np.isfinite(settling_time):
        axes[0].axvline(
            settling_time, color="g", linestyle="-.", label="2% Settling Time"
        )
        axes[0].annotate(
            f"Ts = {settling_time:.3f}s",
            xy=(settling_time, 0.0),
            xytext=(1.1 * settling_time, 0.1 * step_amplitude),
            arrowprops=dict(arrowstyle="->", color="g"),
            fontsize=9,
        )
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Output")
    axes[0].set_title("Step Response - Output")
    axes[0].legend(loc="best")

    # Plot control signal u(t)
    axes[1].plot(t, u, "g-", linewidth=2, label="Control u(t)")
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

    # Plot poles and zeros on z-plane if controller is provided
    if controller_tf is not None:
        num, den = controller_tf
        num = np.asarray(num)
        den = np.asarray(den)

        # Find zeros (roots of numerator) and poles (roots of denominator)
        zeros = np.roots(num)
        poles = np.roots(den)

        # Plot unit circle
        theta = np.linspace(0, 2 * np.pi, 200)
        unit_circle = np.exp(1j * theta)
        axes[3].plot(
            unit_circle.real,
            unit_circle.imag,
            "k--",
            linewidth=1.5,
            label="Unit Circle",
        )

        # Plot poles (x markers)
        if len(poles) > 0:
            axes[3].plot(
                poles.real,
                poles.imag,
                "rx",
                markersize=10,
                markeredgewidth=2,
                label="Poles",
            )

        # Plot zeros (o markers)
        if len(zeros) > 0:
            axes[3].plot(
                zeros.real,
                zeros.imag,
                "bo",
                markersize=10,
                markeredgewidth=2,
                fillstyle="none",
                label="Zeros",
            )

        axes[3].axhline(0, color="k", linewidth=0.5, alpha=0.3)
        axes[3].axvline(0, color="k", linewidth=0.5, alpha=0.3)
        axes[3].grid(True, alpha=0.3)
        axes[3].set_xlabel("Real")
        axes[3].set_ylabel("Imaginary")
        axes[3].set_title("Controller Poles and Zeros (z-plane)")
        axes[3].legend(loc="center left", bbox_to_anchor=(1, 0.5))
        axes[3].set_aspect("equal")
        # Set reasonable limits to show unit disk clearly
        axes[3].set_xlim(-1.2, 1.2)
        axes[3].set_ylim(-1.2, 1.2)

    plt.tight_layout(rect=[0, 0, 0.95, 1])  # Leave space on right for legend
    if save_path:
        # Create output directory if it doesn't exist
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, axes
