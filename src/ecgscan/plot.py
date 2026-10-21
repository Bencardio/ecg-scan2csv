"""Tracé du signal reconstruit, à l'échelle du papier (25 mm/s, 10 mm/mV), pour le contrôle visuel."""

from __future__ import annotations

from typing import Any

import numpy as np

from .post import GRIDS

ROW_MM = 30.0  # écart entre deux rangées


def plot_ecg(data: np.ndarray, header: list[str], meta: dict[str, Any], width_in: float = 14.0):
    """Une figure façon page ECG : la grille déclarée puis les bandes rythme, sur papier millimétré."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = data[:, 0]
    col = {name: data[:, i] for i, name in enumerate(header)}
    layout = meta["layout"].split("+")[0]
    rows = [list(r) for r in GRIDS[layout]] + [[name] for name in header if name.startswith("rhythm_")]
    x_mm = t * 25.0
    width_mm = float(np.nanmax(x_mm)) + 5 if len(x_mm) else 250.0
    height_mm = ROW_MM * len(rows) + 35  # marges pour les grandes ondes R en haut et S profondes en bas

    fig = plt.figure(figsize=(width_in, width_in * height_mm / width_mm))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-5, width_mm)
    ax.set_ylim(0, height_mm)
    ax.axis("off")
    for x in np.arange(0, width_mm, 1.0):
        major = round(x) % 5 == 0
        ax.axvline(x, color="#f2b8b8" if major else "#fbe3e3", lw=0.6 if major else 0.3, zorder=0)
    for y in np.arange(0, height_mm, 1.0):
        major = round(y) % 5 == 0
        ax.axhline(y, color="#f2b8b8" if major else "#fbe3e3", lw=0.6 if major else 0.3, zorder=0)

    windows = meta.get("lead_windows_s", {})
    over = set(meta.get("qc", {}).get("leads_over_threshold", []))
    for r, leads in enumerate(rows):
        base = height_mm - 10 - ROW_MM * (r + 0.5)
        for name in leads:
            if name not in col:
                continue
            ax.plot(x_mm, base + col[name] * 10.0, color="#111", lw=0.7, zorder=2)
            start = windows.get(name, (0.0, 0.0))[0] if not name.startswith("rhythm_") else 0.0
            pct = meta.get("nan_pct", {}).get(name)
            label = name + (f"  ({pct:.0f} % NaN)" if pct and pct >= 1 else "")
            ax.text(start * 25.0 + 1, base + 9, label, fontsize=8, color="#b00" if name in over else "#333",
                    zorder=3)
    return fig
