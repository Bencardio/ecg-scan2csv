"""Post-traitement des lignes extraites par Open-ECG-Digitizer vers le format d'échange ecg-vcg.

Entrée : les lignes du digitizer AVANT canonicalisation (une ligne par rangée imprimée, de haut en bas),
en µV calculés avec l'hypothèse 10 mm/mV, échantillonnées à un échantillon par colonne de pixels.

Sortie : un CSV (t_s, I…V6, rhythm_<X>) à 500 Hz en mV, et un dictionnaire de métadonnées pour le JSON.

Ce module ne dépend que de numpy, pour pouvoir être testé sans le modèle.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import __version__ as PIPELINE_VERSION
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS_OUT = 500.0

# Grilles imprimées : une liste par rangée, de haut en bas ; chaque rangée est découpée en colonnes de même durée.
GRIDS: dict[str, list[list[str]]] = {
    "3x4": [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]],
    "6x2": [["I", "V1"], ["II", "V2"], ["III", "V3"], ["aVR", "V4"], ["aVL", "V5"], ["aVF", "V6"]],
    "12x1": [[lead] for lead in LEADS],
}


@dataclass
class Settings:
    layout: str = "3x4"  # 3x4, 6x2 ou 12x1 (la grille)
    rhythm_leads: list[str] = field(default_factory=lambda: ["V1"])  # bandes rythme sous la grille, de haut en bas
    speed_mm_s: float = 25.0
    gain_mm_mV: float | str = "auto"  # "auto" = lu sur l'impulsion d'étalonnage, sinon valeur imposée
    fallback_gain_mm_mV: float = 10.0
    nominal_duration_s: float = 10.0
    duration_tolerance_s: float = 0.3
    nan_threshold_pct: float = 20.0
    fs_out: float = FS_OUT

    @property
    def layout_label(self) -> str:
        if self.layout == "12x1" or not self.rhythm_leads:
            return self.layout
        return f"{self.layout}+{len(self.rhythm_leads)}"


@dataclass
class Pulse:
    start: int  # indice de colonne du front montant
    end: int  # indice de colonne du front descendant
    height_uV: float  # hauteur lue avec l'hypothèse 10 mm/mV


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Intervalles [début, fin) où mask est vrai."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    return list(zip(edges[::2], edges[1::2]))


def find_calibration_pulse(row: np.ndarray, dt: float, from_end: bool = False, search_s: float = 0.6) -> Pulse | None:
    """Cherche l'impulsion rectangulaire d'étalonnage (1 mV) au début (ou à la fin) d'une rangée.

    Critères : plateau plat de 80 à 320 ms, au moins 300 µV au-dessus de la ligne de base lue,
    précédé (ou suivi) d'une ligne de base plate.
    """
    n = len(row)
    w = min(n, int(round(search_s / dt)))
    seg = row[n - w :][::-1] if from_end else row[:w]
    valid = ~np.isnan(seg)
    if valid.sum() < 0.8 * w:
        return None
    # ligne de base robuste : l'impulsion peut commencer dès le premier pixel lu
    base = float(np.nanpercentile(seg, 25))
    elevated = valid & (seg - base > 300.0)
    best: Pulse | None = None
    for a, b in _runs(elevated):
        dur = (b - a) * dt
        if not 0.08 <= dur <= 0.32:
            continue
        core = seg[a + (b - a) // 5 : b - (b - a) // 5]
        if core.size < 3:
            continue
        height = float(np.nanmedian(core) - base)
        # un plateau : faible dispersion par rapport à la hauteur
        if np.nanstd(core) > 0.08 * height:
            continue
        if best is None or height > best.height_uV:
            best = Pulse(a, b, height)
        break  # la première impulsion plausible suffit
    if best is None:
        return None
    if from_end:
        return Pulse(n - best.end, n - best.start, best.height_uV)
    return best


def _resample_nan_aware(t_src: np.ndarray, y_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Interpolation linéaire ; NaN si l'un des deux voisins source est NaN ou hors du support."""
    out = np.full(t_dst.shape, np.nan)
    if y_src.size < 2:
        return out
    valid = ~np.isnan(y_src)
    if valid.sum() < 2:
        return out
    y_filled = np.where(valid, y_src, 0.0)
    y = np.interp(t_dst, t_src, y_filled)
    m = np.interp(t_dst, t_src, valid.astype(float))
    inside = (t_dst >= t_src[0]) & (t_dst <= t_src[-1])
    ok = inside & (m > 0.999)
    out[ok] = y[ok]
    return out


def _flat_extreme_runs(x: np.ndarray, dt: float, min_ms: float = 12.0) -> dict[str, float]:
    """Repère un écrêtage : plateau prolongé exactement au minimum ou au maximum du tracé."""
    v = x[~np.isnan(x)]
    if v.size < 10:
        return {"clipped_low_ms": 0.0, "clipped_high_ms": 0.0}
    span = float(v.max() - v.min()) or 1.0
    res = {}
    for name, target in (("clipped_low_ms", v.min()), ("clipped_high_ms", v.max())):
        near = np.abs(np.nan_to_num(x, nan=np.inf) - target) < 0.01 * span
        longest = max((b - a for a, b in _runs(near)), default=0) * dt * 1000
        res[name] = round(longest, 1) if longest >= min_ms else 0.0
    return res


def process(lines_uV: np.ndarray, px_per_mm_x: float, settings: Settings) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Transforme les lignes du digitizer en tableau (t_s, dérivations…) à fs_out.

    Retourne (data, header, meta). data a une colonne de temps puis une colonne par dérivation.
    """
    if settings.layout not in GRIDS:
        raise ValueError(f"Mise en page inconnue : {settings.layout}")
    grid = GRIDS[settings.layout]
    n_rows_expected = len(grid) + (0 if settings.layout == "12x1" else len(settings.rhythm_leads))
    lines = np.asarray(lines_uV, dtype=float)
    if lines.ndim != 2:
        raise ValueError("lines_uV doit être un tableau 2D (rangées × colonnes de pixels)")
    warnings: list[str] = []
    if lines.shape[0] != n_rows_expected:
        raise ValueError(
            f"{lines.shape[0]} rangées détectées, {n_rows_expected} attendues pour {settings.layout_label}"
        )

    dt = 1.0 / (px_per_mm_x * settings.speed_mm_s)  # secondes par colonne de pixels

    # 1. Impulsions d'étalonnage : début (et éventuellement fin) de chaque rangée.
    starts, ends, heights = [], [], []
    for r in range(lines.shape[0]):
        p = find_calibration_pulse(lines[r], dt)
        if p is not None:
            starts.append(p.end)
            heights.append(p.height_uV)
        q = find_calibration_pulse(lines[r], dt, from_end=True)
        if q is not None:
            ends.append(q.start)
            heights.append(q.height_uV)
    gap = int(round(0.02 / dt))  # petite marge après le front descendant
    i0 = int(np.median(starts)) + gap if starts else 0
    i1 = int(np.median(ends)) - gap if ends else lines.shape[1]
    if not starts:
        warnings.append("impulsion d'étalonnage non trouvée au début des rangées : t = 0 au premier pixel lu")

    if settings.gain_mm_mV == "auto":
        if heights:
            gain = round(10.0 * float(np.median(heights)) / 1000.0, 2)
            gain_source = "calibration_pulse"
            nominal = min((2.5, 5.0, 10.0, 20.0), key=lambda g: abs(g - gain))
            if abs(gain - nominal) / nominal > 0.15:
                warnings.append(f"gain mesuré {gain} mm/mV éloigné des valeurs usuelles ({nominal})")
        else:
            gain = settings.fallback_gain_mm_mV
            gain_source = "assumed"
            warnings.append(f"gain non mesurable : {gain} mm/mV supposé")
    else:
        gain = float(settings.gain_mm_mV)
        gain_source = "user"

    body = lines[:, i0:i1]
    duration = body.shape[1] * dt
    if abs(duration - settings.nominal_duration_s) > settings.duration_tolerance_s:
        warnings.append(
            f"durée lue {duration:.2f} s pour {settings.nominal_duration_s} s attendues : "
            "vérifier la vitesse de défilement ou le cadrage"
        )

    scale = (10.0 / gain) / 1000.0  # µV lus à 10 mm/mV -> mV réels
    t_src_full = np.arange(body.shape[1]) * dt
    n_out = int(round(duration * settings.fs_out))
    t_out = np.arange(n_out) / settings.fs_out

    columns: dict[str, np.ndarray] = {}
    windows: dict[str, tuple[float, float]] = {}

    def put(name: str, row: np.ndarray, a: int, b: int) -> None:
        y = row[a:b] * scale
        if np.any(~np.isnan(y)):
            y = y - np.nanmedian(y)  # retire seulement l'offset continu de la fenêtre
        t_lo, t_hi = t_src_full[a], t_src_full[b - 1]
        mask = (t_out >= t_lo) & (t_out <= t_hi)
        col = columns.setdefault(name, np.full(n_out, np.nan))
        col[mask] = _resample_nan_aware(t_src_full[a:b], y, t_out[mask])
        windows[name] = (round(float(t_lo), 4), round(float(t_hi), 4))

    width = body.shape[1]
    for r, row_leads in enumerate(grid):
        cols = len(row_leads)
        for c, lead in enumerate(row_leads):
            a = c * width // cols
            b = (c + 1) * width // cols
            put(lead, body[r], a, b)
    rhythm_names = []
    if settings.layout != "12x1":
        for k, lead in enumerate(settings.rhythm_leads):
            name = f"rhythm_{lead}"
            rhythm_names.append(name)
            put(name, body[len(grid) + k], 0, width)

    header = ["t_s"] + LEADS + rhythm_names
    data = np.full((n_out, len(header)), np.nan)
    data[:, 0] = t_out
    for j, name in enumerate(header[1:], start=1):
        if name in columns:
            data[:, j] = columns[name]

    # 2. Contrôle qualité.
    nan_pct: dict[str, float] = {}
    for name in header[1:]:
        if name not in windows:
            nan_pct[name] = 100.0
            continue
        lo, hi = windows[name]
        m = (t_out >= lo) & (t_out <= hi)
        nan_pct[name] = round(100.0 * float(np.isnan(columns[name][m]).mean()), 2) if m.any() else 100.0
    over = [k for k, v in nan_pct.items() if v > settings.nan_threshold_pct]

    rhythm_checks = {}
    for name in rhythm_names:
        lead = name.removeprefix("rhythm_")
        lo, hi = windows.get(lead, (0, -1))
        m = (t_out >= lo) & (t_out <= hi)
        a, b = columns[name][m], columns[lead][m] if lead in columns else np.array([])
        ok = ~np.isnan(a) & ~np.isnan(b) if b.size else np.array([], dtype=bool)
        r = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 50 else float("nan")
        clip = _flat_extreme_runs(columns[name], 1.0 / settings.fs_out)
        rhythm_checks[name] = {"corr_with_grid_segment": round(r, 3) if not np.isnan(r) else None, **clip}
        if np.isnan(r) or r < 0.9:
            warnings.append(f"{name} ressemble peu à {lead} de la grille (r = {r:.2f}) : ordre des rangées à vérifier")
        if clip["clipped_low_ms"] or clip["clipped_high_ms"]:
            warnings.append(f"{name} : plateau aux extrêmes, possible écrêtage")

    meta: dict[str, Any] = {
        "source": "scan",
        "layout": settings.layout_label,
        "rhythm_leads": settings.rhythm_leads if settings.layout != "12x1" else [],
        "speed_mm_s": settings.speed_mm_s,
        "gain_mm_mV": gain,
        "gain_source": gain_source,
        "fs_hz": settings.fs_out,
        "duration_s": round(duration, 3),
        "units": "mV",
        "lead_windows_s": windows,
        "nan_pct": nan_pct,
        "qc": {
            "nan_threshold_pct": settings.nan_threshold_pct,
            "leads_over_threshold": over,
            "rhythm": rhythm_checks,
            "calibration_pulses_found": len(heights),
            "warnings": warnings,
        },
        "pipeline": {"name": "ecgscan", "version": PIPELINE_VERSION, "px_per_mm_x": round(px_per_mm_x, 4)},
    }
    return data, header, meta


def write_outputs(data: np.ndarray, header: list[str], meta: dict[str, Any], csv_path: str, json_path: str) -> None:
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in data:
            w.writerow(["NaN" if np.isnan(v) else f"{v:.6g}" for v in row])
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
