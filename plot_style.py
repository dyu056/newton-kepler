"""Shared plotting style for Newton-Kepler result figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def apply_plot_settings() -> None:
    """Apply a compact serif style suitable for paper figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "axes.axisbelow": True,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "lines.linewidth": 1.8,
            "lines.markersize": 2.4,
            "lines.markeredgewidth": 0.6,
            "legend.fontsize": 9,
            "legend.frameon": True,
            "legend.edgecolor": "black",
            "legend.fancybox": False,
            "legend.framealpha": 0.95,
            "grid.linestyle": "--",
            "grid.alpha": 0.5,
            "grid.linewidth": 0.5,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def get_color_palette() -> dict[str, str]:
    return {
        "royal_blue": "#002060",
        "crimson": "#C00000",
        "emerald": "#008B45",
        "gold": "#B8860B",
        "purple": "#68228B",
        "teal": "#008080",
        "slate": "#2F4F4F",
    }


def get_line_styles() -> list[str]:
    return ["-", "--", "-.", ":"]


def get_functional_colors() -> dict[str, str]:
    colors = get_color_palette()
    return {
        "train": colors["slate"],
        "iid": colors["emerald"],
        "ood": colors["crimson"],
        "primary": colors["royal_blue"],
        "secondary": colors["gold"],
    }


def get_color_cycle() -> list[str]:
    colors = get_color_palette()
    return [
        colors["royal_blue"],
        colors["crimson"],
        colors["emerald"],
        colors["gold"],
        colors["purple"],
        colors["teal"],
        colors["slate"],
    ]


def style_axes(ax) -> None:
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(which="both", direction="in")


def save_figure(fig, path: str | Path) -> None:
    """Save a PNG figure using the shared style."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
