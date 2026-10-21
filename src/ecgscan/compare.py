"""Validation : compare des CSV issus de scans à des CSV de référence (même ECG, extrait du PDF vectoriel).

Les CSV sont appariés par nom de fichier. Pour chaque dérivation, on cherche le décalage (± max_lag_ms) qui
maximise la corrélation, puis on rapporte r, RMSE (µV) et décalage.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def read_csv(path: str | Path) -> tuple[list[str], np.ndarray]:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    data = np.array([[float(v) if v not in ("", "NaN", "nan") else np.nan for v in r] for r in rows[1:]])
    return header, data


def best_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[int, float, float, int]:
    best = (0, -np.inf, np.nan, 0)
    for lag in range(-max_lag, max_lag + 1):
        x, y = (a[lag:], b[: len(b) - lag]) if lag >= 0 else (a[:lag], b[-lag:])
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]
        ok = ~np.isnan(x) & ~np.isnan(y)
        if ok.sum() < 100:
            continue
        r = float(np.corrcoef(x[ok], y[ok])[0, 1])
        if r > best[1]:
            rmse = float(np.sqrt(np.mean((x[ok] - y[ok]) ** 2))) * 1000
            best = (lag, r, rmse, int(ok.sum()))
    return best


def compare_dirs(scan_dir: str | Path, ref_dir: str | Path, max_lag_ms: float = 200.0) -> list[list]:
    results: list[list] = []
    for path in sorted(Path(scan_dir).glob("*.csv")):
        ref = Path(ref_dir) / path.name
        if not ref.exists():
            print(f"pas de référence pour {path.name}")
            continue
        hs, ds = read_csv(path)
        hr, dr = read_csv(ref)
        fs = 1.0 / np.median(np.diff(ds[:, 0]))
        max_lag = int(round(max_lag_ms / 1000 * fs))
        for lead in hs[1:]:
            if lead not in hr:
                continue
            lag, r, rmse, n = best_lag(ds[:, hs.index(lead)], dr[:, hr.index(lead)], max_lag)
            results.append([path.name, lead, n, round(lag / fs * 1000, 1), round(r, 4), round(rmse, 1)])
    return results


def write_results(results: list[list], out: str | Path) -> None:
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "lead", "n", "lag_ms", "r", "rmse_uV"])
        w.writerows(results)
