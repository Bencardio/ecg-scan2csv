"""Ligne de commande : ecgscan setup | convert | compare | info."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

from . import __version__, post


def cmd_setup(args: argparse.Namespace) -> int:
    from . import digitizer

    digitizer.install(force=args.force)
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    from . import digitizer

    print(json.dumps({"ecgscan": __version__, **digitizer.version_info()}, indent=2, ensure_ascii=False))
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    from . import digitizer

    in_dir, out_dir = Path(args.input).resolve(), Path(args.output).resolve()
    files = sorted(p for p in in_dir.rglob("*") if p.suffix.lower() in digitizer.INPUT_EXT)
    if not files:
        print(f"Aucun scan (PDF, PNG, JPG, TIFF) dans {in_dir}")
        return 1
    if not digitizer.is_installed():
        digitizer.install()
    settings = post.Settings(
        layout=args.layout,
        rhythm_leads=[r.strip() for r in args.rhythm.split(",") if r.strip()],
        speed_mm_s=args.speed,
        gain_mm_mV=args.gain if args.gain == "auto" else float(args.gain),
        nan_threshold_pct=args.nan_threshold,
    )
    model = digitizer.Digitizer(device=args.device, resample_size=args.resample_size)
    print(f"{len(files)} fichiers, appareil de calcul : {model.device}")

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for path in files:
        rel = path.relative_to(in_dir)
        pages = digitizer.iter_pages(path, args.pdf_dpi, args.rotate)
        while True:
            try:
                i, n, image, dpi = next(pages)
            except StopIteration:
                break
            except Exception as e:
                summary.append({"file": str(rel), "status": "error", "error": f"lecture impossible : {e}"})
                print(f"!!  {rel} : lecture impossible : {e}")
                break
            # Un PDF de plusieurs pages donne <nom>_p1, <nom>_p2… ; une page seule garde le nom du scan.
            name = rel.with_suffix("").name + (f"_p{i}" if n > 1 else "")
            base = out_dir / rel.parent / name
            base.parent.mkdir(parents=True, exist_ok=True)
            label = str(rel) + (f" (page {i})" if n > 1 else "")
            entry: dict = {"file": str(rel), "page": i, "output": name + ".csv"}
            try:
                if image.height > image.width:
                    raise ValueError(
                        "page en portrait : l'ECG est probablement tourné, relancer avec --rotate 90 (ou 270)"
                    )
                got = model(image)
                data, header, meta = post.process(got["lines_uV"], got["px_per_mm_x"], settings)
                meta = {"file": path.name, "page": i, **meta, "scan_dpi": dpi or args.dpi}
                meta["pipeline"].update(
                    ecgscan=__version__,
                    digitizer_commit=digitizer.COMMIT,
                    digitizer_layout_guess=got["layout_guess"],
                    resample_size=int(model.config.MODEL.KWARGS.resample_size),
                )
                post.write_outputs(data, header, meta, str(base) + ".csv", str(base) + ".json")
                if args.save_raw:
                    np.save(str(base) + "_raw_lines_uV.npy", got["lines_uV"])
                entry.update(status="ok", warnings=meta["qc"]["warnings"], over=meta["qc"]["leads_over_threshold"])
            except Exception as e:  # le lot continue ; l'échec est tracé dans le résumé
                entry.update(status="error", error=str(e), trace=traceback.format_exc(limit=3))
            summary.append(entry)
            ok = entry["status"] == "ok" and not entry.get("warnings")
            print(("OK  " if ok else "!!  ") + label + ("" if entry["status"] == "ok" else f" : {entry['error']}"))
            for w in entry.get("warnings", []):
                print(f"      - {w}")

    with open(out_dir / "ecgscan_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    n_ok = sum(e["status"] == "ok" for e in summary)
    print(f"{n_ok}/{len(summary)} ECG convertis ; détail dans {out_dir / 'ecgscan_summary.json'}")
    return 0 if n_ok == len(summary) else 2


def cmd_compare(args: argparse.Namespace) -> int:
    from .compare import compare_dirs, write_results

    results = compare_dirs(args.scan, args.ref, args.max_lag_ms)
    write_results(results, args.out)
    if results:
        r = np.array([x[4] for x in results if np.isfinite(x[4])])
        e = np.array([x[5] for x in results if np.isfinite(x[5])])
        print(f"{len(results)} dérivations : r médian {np.median(r):.3f}, RMSE médiane {np.median(e):.0f} µV")
    print(f"détail : {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ecgscan", description="ECG papier scannés -> CSV 500 Hz + JSON (format ecg-vcg)")
    p.add_argument("--version", action="version", version=f"ecgscan {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("setup", help="télécharge Open-ECG-Digitizer et ses poids (une seule fois)")
    s.add_argument("--force", action="store_true", help="retélécharger")
    s.set_defaults(func=cmd_setup)

    s = sub.add_parser("info", help="versions, dossier du digitizer, GPU")
    s.set_defaults(func=cmd_info)

    s = sub.add_parser("convert", help="convertit un dossier de scans")
    s.add_argument("input", help="dossier des scans (PDF, PNG, JPG, TIFF ; sous-dossiers inclus)")
    s.add_argument("output", help="dossier de sortie")
    s.add_argument("--layout", default="3x4", choices=sorted(post.GRIDS), help="grille imprimée (défaut 3x4)")
    s.add_argument("--rhythm", default="V1", help="bandes rythme de haut en bas, ex. V1 ou V1,II,V5 ; '' si aucune")
    s.add_argument("--speed", type=float, default=25.0, help="vitesse en mm/s (défaut 25)")
    s.add_argument("--gain", default="auto", help="mm/mV, ou 'auto' : lu sur l'impulsion d'étalonnage (défaut)")
    s.add_argument("--dpi", type=float, default=None, help="résolution du scan si absente du fichier")
    s.add_argument("--pdf-dpi", type=int, default=None, help="résolution de rendu des PDF (défaut : celle du scan, max 600)")
    s.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270], help="rotation horaire des pages, en degrés")
    s.add_argument("--nan-threshold", type=float, default=20.0, help="%% de NaN au-delà duquel une dérivation est signalée")
    s.add_argument("--device", default=None, help="cpu, cuda:0… (défaut : GPU si disponible)")
    s.add_argument("--resample-size", type=int, default=None, help="côté de l'image pour le modèle (défaut du digitizer : 3000)")
    s.add_argument("--save-raw", action="store_true", help="garde aussi les lignes brutes du digitizer (.npy)")
    s.set_defaults(func=cmd_convert)

    s = sub.add_parser("compare", help="compare des CSV de scans aux CSV de référence (PDF vectoriels)")
    s.add_argument("--scan", required=True)
    s.add_argument("--ref", required=True)
    s.add_argument("--out", default="comparaison.csv")
    s.add_argument("--max-lag-ms", type=float, default=200.0)
    s.set_defaults(func=cmd_compare)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
