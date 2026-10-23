"""Installation et appel d'Open-ECG-Digitizer (https://github.com/Ahus-AIM/Open-ECG-Digitizer).

Le dépôt du digitizer n'est pas un paquet pip : on télécharge une version figée (archive zip d'un commit)
et ses poids (fichiers Git LFS), vérifiés par SHA-256, dans un dossier de cache. Ni git ni git-lfs ne
sont nécessaires.
"""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import types
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

REPO = "Ahus-AIM/Open-ECG-Digitizer"
COMMIT = "97a15087d4abcda843da8c58ee74b1d8f47e6f9a"  # 4 juin 2026
WEIGHTS = {
    "weights/unet_weights_07072025.pt": "17fe7071ef270102631306127262fc08c250d79d4e3aeb572ab1719dd34d320b",
    "weights/lead_name_unet_weights_07072025.pt": "840bd6bf2433ee6c22db67f57c861d9d427f29e10a32eeb334f0bcf061b175a2",
}
DEFAULT_CONFIG = "inference_wrapper_ahus_testset.yml"
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
INPUT_EXT = IMAGE_EXT + (".pdf",)
PDF_FALLBACK_DPI = 300  # A4 à 300 dpi ≈ 3500 px, au-dessus des 3000 px vus par le modèle
PDF_MAX_DPI = 300  # au-delà, le modèle réduit de toute façon l'image : rendre plus fin ne coûte que temps et mémoire


def home() -> Path:
    """Dossier de cache : $ECGSCAN_HOME, sinon ~/.ecgscan."""
    return Path(os.environ.get("ECGSCAN_HOME", Path.home() / ".ecgscan")).expanduser()


def digitizer_dir() -> Path:
    return home() / f"Open-ECG-Digitizer-{COMMIT[:7]}"


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "ecgscan"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_installed() -> bool:
    d = digitizer_dir()
    return (d / "src" / "digitize.py").exists() and all(
        (d / rel).exists() and (d / rel).stat().st_size > 1000 for rel in WEIGHTS
    )


def install(force: bool = False, log=print) -> Path:
    """Télécharge le code figé du digitizer et ses poids ; idempotent."""
    target = digitizer_dir()
    if force and target.exists():
        shutil.rmtree(target)
    if not (target / "src" / "digitize.py").exists():
        log(f"Téléchargement d'Open-ECG-Digitizer ({COMMIT[:7]})…")
        data = _download(f"https://github.com/{REPO}/archive/{COMMIT}.zip")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                z.extractall(tmp)
            (inner,) = [p for p in Path(tmp).iterdir() if p.is_dir()]
            shutil.move(str(inner), target)
    for rel, sha in WEIGHTS.items():
        path = target / rel
        if path.exists() and path.stat().st_size > 1000 and _sha256(path) == sha:
            continue
        log(f"Téléchargement des poids {path.name}…")
        blob = _download(f"https://media.githubusercontent.com/media/{REPO}/{COMMIT}/{rel}")
        if hashlib.sha256(blob).hexdigest() != sha:
            raise RuntimeError(f"Somme de contrôle invalide pour {rel}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    log(f"Digitizer prêt dans {target}")
    return target


def _stub_ray() -> None:
    """src.utils importe ray.tune.Stopper pour l'entraînement seulement : un module vide suffit à l'inférence."""
    try:
        import ray.tune  # noqa: F401
    except ImportError:
        ray = types.ModuleType("ray")
        tune = types.ModuleType("ray.tune")

        class Stopper:  # même nom que la classe de base attendue
            pass

        tune.Stopper = Stopper
        ray.tune = tune
        sys.modules["ray"], sys.modules["ray.tune"] = ray, tune


def default_device() -> str:
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _pdf_scan_dpi(page) -> float | None:
    """Résolution du scan inclus dans une page PDF : pixels de la plus grande image / largeur imprimée (pouces)."""
    import pypdfium2.raw as pdfium_c

    best = None
    for obj in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE]):
        left, _, right, _ = obj.get_bounds() if hasattr(obj, "get_bounds") else obj.get_pos()
        width_px, _ = obj.get_px_size()
        width_in = (right - left) / 72
        if width_in > 0 and (best is None or width_in > best[0]):
            best = (width_in, width_px / width_in)
    return float(round(best[1])) if best else None


def _rotate(image, degrees_cw: int):
    return image.rotate(-degrees_cw, expand=True) if degrees_cw % 360 else image


def iter_pages(path: str | Path, pdf_dpi: int | None = None, rotate: int = 0):
    """Une image RGB par page, l'une après l'autre : (n° de page, nombre de pages, image PIL, dpi du scan ou None).

    Les pages PDF sont rendues à la résolution du scan qu'elles contiennent (plafonnée à 300 dpi ; 300 dpi si
    elle est inconnue), sauf si pdf_dpi est imposé. rotate : rotation horaire en degrés appliquée à chaque page.
    """
    from PIL import Image

    path = Path(path)
    if path.suffix.lower() != ".pdf":
        with Image.open(path) as im:
            dpi = im.info.get("dpi")
            yield 1, 1, _rotate(im.convert("RGB"), rotate), (float(round(dpi[0])) if dpi else None)
        return
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        for i, page in enumerate(pdf, 1):
            dpi = _pdf_scan_dpi(page)
            render_dpi = pdf_dpi or min(dpi or PDF_FALLBACK_DPI, PDF_MAX_DPI)
            image = page.render(scale=render_dpi / 72).to_pil().convert("RGB")
            page.close()
            yield i, len(pdf), _rotate(image, rotate), dpi
    finally:
        pdf.close()


class Digitizer:
    """Enveloppe l'InferenceWrapper du digitizer ; renvoie les lignes brutes (µV lus à 10 mm/mV) et la grille."""

    def __init__(self, device: str | None = None, config: str = DEFAULT_CONFIG, resample_size: int | None = None):
        if not is_installed():
            raise RuntimeError("Digitizer absent : lancez d'abord « ecgscan setup ».")
        root = digitizer_dir()
        _stub_ray()
        sys.path.insert(0, str(root))
        cwd = os.getcwd()
        os.chdir(root)  # les chemins des poids dans la configuration sont relatifs au dépôt
        try:
            from src.config.default import get_cfg
            from src.model.lead_identifier import LeadIdentifier
            from src.utils import find_config_path, import_class_from_path

            # Une colonne de pixels = un échantillon ; le temps est reconstruit dans ecgscan.post.
            LeadIdentifier._interpolate_lines = lambda self, lines, target_num_samples: lines

            self.device = device or default_device()
            cfg = get_cfg(find_config_path(config))
            cfg.defrost()
            cfg.MODEL.KWARGS.device = self.device
            cfg.MODEL.KWARGS.config.LAYOUT_IDENTIFIER.KWARGS.device = self.device
            if resample_size:
                cfg.MODEL.KWARGS.resample_size = int(resample_size)
            cfg.freeze()
            self.config = cfg
            self.wrapper = import_class_from_path(cfg.MODEL.class_path)(**cfg.MODEL.KWARGS)
        finally:
            os.chdir(cwd)

    def __call__(self, image) -> dict[str, Any]:
        """image : image PIL RGB d'une page (voir iter_pages)."""
        import torch
        from PIL import Image

        # Le modèle ramène l'image à resample_size px de grand côté, mais après l'avoir normalisée en flottants à
        # pleine résolution (plus de 1 Go pour une page à 600 dpi). La réduire ici donne le même résultat, plus vite
        # et sans saturer la mémoire d'un portable.
        size = self.config.MODEL.KWARGS.resample_size
        if isinstance(size, int) and max(image.size) > size:
            scale = size / max(image.size)
            image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
        tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).contiguous()
        with torch.no_grad():
            got = self.wrapper(tensor, layout_should_include_substring=None)
        lines = got["signal"]["lines"]
        if lines is None:
            raise RuntimeError("le digitizer n'a extrait aucun tracé")
        lines = lines.squeeze().cpu().numpy() if hasattr(lines, "cpu") else np.asarray(lines)
        if lines.ndim == 1:
            lines = lines[None, :]
        return {
            "lines_uV": lines,
            "px_per_mm_x": 1.0 / float(got["pixel_spacing_mm"]["x"]),
            "layout_guess": got.get("layout_name"),
        }


def version_info() -> dict[str, str]:
    info = {"digitizer_commit": COMMIT, "digitizer_dir": str(digitizer_dir()), "installed": str(is_installed())}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = str(torch.cuda.is_available())
    except ImportError:
        info["torch"] = "absent"
    try:
        info["git"] = subprocess.run(["git", "--version"], capture_output=True, text=True).stdout.strip()
    except OSError:
        pass
    return info
