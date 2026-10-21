"""Fabrique une page ECG synthétique façon GE 3x4 + bande rythme V1 (25 mm/s, 10 mm/mV), en PNG.

Sert au test d'intégration : vérifie que toute la chaîne tourne, pas la précision (le modèle est entraîné
sur de vrais tracés).
"""

from __future__ import annotations

import numpy as np

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
GRID = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]


def synthetic_ecg(t: np.ndarray, lead_idx: int) -> np.ndarray:
    """ECG grossier en mV : P, QRS (rS en V1), T ; 75 bpm."""
    sig = np.zeros_like(t)
    amp = 1.0 + 0.1 * lead_idx
    rs = lead_idx == 6
    for beat in np.arange(0.3, 10, 0.8):
        sig += 0.12 * amp * np.exp(-((t - beat + 0.16) ** 2) / (2 * 0.02**2))
        sig += (0.3 if rs else 1.2) * amp * np.exp(-((t - beat) ** 2) / (2 * 0.012**2))
        sig -= (1.4 if rs else 0.25) * amp * np.exp(-((t - beat - 0.03) ** 2) / (2 * 0.012**2))
        sig += 0.25 * amp * np.exp(-((t - beat - 0.3) ** 2) / (2 * 0.05**2))
    return sig


def render(path: str, dpi: int = 300, gain_mm_mV: float = 10.0, speed: float = 25.0) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    W, H = 297.0, 210.0  # A4 paysage, mm
    fig = plt.figure(figsize=(W / 25.4, H / 25.4), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    x0, x1, y_top, y_bot = 15.0, 15.0 + 10 + 250.0, 185.0, 20.0
    for x in np.arange(x0, x1 + 0.01, 1.0):
        ax.plot([x, x], [y_bot - 15, y_top + 15], color="#f4b6b6", lw=0.6 if round(x - x0) % 5 == 0 else 0.25)
    for y in np.arange(y_bot - 15, y_top + 15.01, 1.0):
        ax.plot([x0, x1], [y, y], color="#f4b6b6", lw=0.6 if round(y - y_bot) % 5 == 0 else 0.25)
    t = np.arange(0, 10, 0.002)
    truth = {lead: synthetic_ecg(t, i) for i, lead in enumerate(LEADS)}
    rows = []
    for row in GRID:
        r = np.empty_like(t)
        for c, lead in enumerate(row):
            sl = slice(c * len(t) // 4, (c + 1) * len(t) // 4)
            r[sl] = truth[lead][sl]
        rows.append(r)
    rows.append(truth["V1"])
    baselines = np.linspace(y_top - 10, y_bot + 15, 4)
    for r, yb in zip(rows, baselines):
        # impulsion d'étalonnage 1 mV, 0,2 s, puis tracé
        px = [x0, x0 + 1.25, x0 + 1.25, x0 + 1.25 + 5, x0 + 1.25 + 5, x0 + 1.25 + 5 + 0.5]
        py = [yb, yb, yb + gain_mm_mV, yb + gain_mm_mV, yb, yb]
        ax.plot(px, py, color="black", lw=0.8)
        ax.plot(px[-1] + t * speed, yb + r * gain_mm_mV, color="black", lw=0.8)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
