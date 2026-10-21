"""Conversion d'une page scannée : digitizer + post-traitement. Partagé par la ligne de commande et l'application."""

from __future__ import annotations

import os
import traceback
from typing import Any

import numpy as np

from . import __version__, digitizer, post


def output_name(stem: str, page: int, n_pages: int) -> str:
    """Un PDF de plusieurs pages donne <nom>_p1, <nom>_p2… ; une page seule garde le nom du scan."""
    return stem + (f"_p{page}" if n_pages > 1 else "")


def convert_page(
    model: digitizer.Digitizer, image, dpi: float | None, settings: post.Settings, file_name: str, page: int = 1
) -> tuple[np.ndarray, list[str], dict[str, Any], np.ndarray]:
    """Renvoie (données, en-tête, métadonnées, lignes brutes µV) ; lève une exception si la page est inexploitable."""
    if image.height > image.width:
        raise ValueError("page en portrait : l'ECG est probablement tourné, relancer avec une rotation de 90° (ou 270°)")
    got = model(image)
    data, header, meta = post.process(got["lines_uV"], got["px_per_mm_x"], settings)
    meta = {"file": file_name, "page": page, **meta, "scan_dpi": dpi}
    meta["pipeline"].update(
        ecgscan=__version__,
        digitizer_commit=digitizer.COMMIT,
        digitizer_layout_guess=got["layout_guess"],
        resample_size=int(model.config.MODEL.KWARGS.resample_size),
    )
    return data, header, meta, got["lines_uV"]


def convert_scan(
    model: digitizer.Digitizer,
    path,
    settings: post.Settings,
    display_name: str | None = None,
    rotate: int = 0,
    pdf_dpi: int | None = None,
    default_dpi: float | None = None,
    keep_image: bool = False,
) -> list[dict[str, Any]]:
    """Convertit toutes les pages d'un scan ; une erreur sur une page n'arrête pas les suivantes.

    Chaque enregistrement : name (nom de sortie sans extension), file, page, status ("ok" ou "error"), et selon le
    cas error, ou data, header, meta, raw (lignes brutes µV) ; image (PIL) si keep_image.
    """
    file_name = display_name or os.path.basename(str(path))
    stem = os.path.splitext(file_name)[0]
    records: list[dict[str, Any]] = []
    pages = digitizer.iter_pages(path, pdf_dpi, rotate)
    while True:
        try:
            i, n, image, dpi = next(pages)
        except StopIteration:
            break
        except Exception as e:
            records.append({"name": stem, "file": file_name, "page": len(records) + 1, "status": "error",
                            "error": f"lecture impossible : {e}"})
            break
        rec: dict[str, Any] = {"name": output_name(stem, i, n), "file": file_name, "page": i, "n_pages": n}
        if keep_image:
            rec["image"] = image
        try:
            data, header, meta, raw = convert_page(model, image, dpi or default_dpi, settings, file_name, i)
            rec.update(status="ok", data=data, header=header, meta=meta, raw=raw)
        except Exception as e:
            rec.update(status="error", error=str(e), trace=traceback.format_exc(limit=3))
        records.append(rec)
    return records
