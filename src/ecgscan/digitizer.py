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

    @staticmethod
    def load_image(path: str | Path):
        import torch
        from PIL import Image

        with Image.open(path) as im:
            dpi = im.info.get("dpi")
            rgb = np.array(im.convert("RGB"))
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).contiguous()
        return tensor, (float(round(dpi[0])) if dpi else None)

    def __call__(self, path: str | Path) -> dict[str, Any]:
        import torch

        image, dpi = self.load_image(path)
        with torch.no_grad():
            got = self.wrapper(image, layout_should_include_substring=None)
        lines = got["signal"]["lines"]
        if lines is None:
            raise RuntimeError("le digitizer n'a extrait aucun tracé")
        lines = lines.squeeze().cpu().numpy() if hasattr(lines, "cpu") else np.asarray(lines)
        if lines.ndim == 1:
            lines = lines[None, :]
        return {
            "lines_uV": lines,
            "px_per_mm_x": 1.0 / float(got["pixel_spacing_mm"]["x"]),
            "dpi": dpi,
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
