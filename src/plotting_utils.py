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


def plot_response(
    results: list[tuple],
    metrics: dict,
    save_path: str | None = None,
) -> tuple[Figure, list[Axes]]:
    """
    Plot output, control, and error signals with 2% settling band and peak annotations.

    Parameters
    ----------
    results : list of tuples
        List of tuples from simulate_system. Each tuple should be (t, r, y, e, ...)
        where the first 4 elements are (t, r, y, e). Additional elements are plotted
        as control signals.
        - t: Time value
        - r: Reference signal value
        - y: Output response value
        - e: Error signal value
        - *other_signals: Any additional signals to plot (will be labeled as "Control signal {n}")
    metrics : dict
        Dictionary containing step response metrics (from compute_metrics)
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

    # Extract signals from list of tuples
    # First 4 elements are (t, r, y, e), rest are additional control signals
    t = np.array([row[0] for row in results])
    r = np.array([row[1] for row in results])
    y = np.array([row[2] for row in results])
    e = np.array([row[3] for row in results])
    
    # Get additional signals if present
    if len(results) > 0 and len(results[0]) > 4:
        num_other_signals = len(results[0]) - 4
        other_signals = [np.array([row[4 + i] for row in results]) for i in range(num_other_signals)]
    else:
        other_signals = []

    # Create subplots: output, error, and one for each additional signal
    n_subplots = 2 + len(other_signals)  # output, error, plus additional signals
    fig, axes = plt.subplots(n_subplots, 1, figsize=(10, 3.5 * n_subplots))
    
    # Handle single subplot case
    if n_subplots == 1:
        axes = [axes]

    steady = metrics["steady_state"]
    band = 0.02 * max(abs(steady), 1e-9)
    peak_idx = np.argmax(y)
    settling_time = metrics["settling_time_2pct"]

    # Plot output with reference
    axes[0].plot(t, y, "b-", linewidth=2, label="Output y(t)")
    axes[0].plot(t, r, "k:", linewidth=1.5, alpha=0.7, label="Reference r(t)")
    ref_value = r[-1] if len(r) > 0 else steady
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
            xytext=(1.1 * settling_time, 0.1 * ref_value),
            arrowprops=dict(arrowstyle="->", color="g"),
            fontsize=9,
        )
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Output")
    axes[0].set_title("Step Response - Output")
    axes[0].legend(loc="best")

    # Plot error signal
    axes[1].plot(t, e, "r-", linewidth=2, label="Error e(t) = r(t) - y(t)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Error")
    axes[1].set_title("Error Signal")
    axes[1].legend()

    # Plot additional signals as "Control signal {n}"
    for idx, signal in enumerate(other_signals):
        ax_idx = 2 + idx
        axes[ax_idx].plot(t, signal, "g-", linewidth=2, label=f"Control signal {idx + 1}")
        axes[ax_idx].grid(True, alpha=0.3)
        axes[ax_idx].set_xlabel("Time (s)")
        axes[ax_idx].set_ylabel(f"Control signal {idx + 1}")
        axes[ax_idx].set_title(f"Control signal {idx + 1}")
        axes[ax_idx].legend()

    plt.tight_layout()
    if save_path:
        # Create output directory if it doesn't exist
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, axes
