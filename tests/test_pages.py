"""Lecture des scans : images et PDF (une image scannée par page, comme un PDF de scanner)."""

import numpy as np
import pytest
from PIL import Image

from ecgscan import digitizer


def scan_like(width=1200, height=850):
    rgb = np.full((height, width, 3), 255, np.uint8)
    rgb[:, ::12] = (244, 182, 182)  # quadrillage rose
    rgb[400:410, 100:300] = 0  # un trait noir à gauche pour repérer l'orientation
    return Image.fromarray(rgb)


def test_png_keeps_dpi(tmp_path):
    scan_like().save(tmp_path / "a.png", dpi=(600, 600))
    ((i, n, image, dpi),) = digitizer.iter_pages(tmp_path / "a.png")
    assert (i, n, dpi, image.size) == (1, 1, 600.0, (1200, 850))


def test_pdf_pages_rendered_at_scan_resolution(tmp_path):
    pages = [scan_like(), scan_like()]
    pages[0].save(tmp_path / "b.pdf", save_all=True, append_images=pages[1:], resolution=150)
    got = list(digitizer.iter_pages(tmp_path / "b.pdf"))
    assert [(i, n) for i, n, *_ in got] == [(1, 2), (2, 2)]
    for _, _, image, dpi in got:
        assert dpi == 150.0
        assert abs(image.width - 1200) <= 2 and abs(image.height - 850) <= 2


def test_pdf_render_dpi_can_be_forced(tmp_path):
    scan_like().save(tmp_path / "c.pdf", resolution=150)
    ((_, _, image, dpi),) = digitizer.iter_pages(tmp_path / "c.pdf", pdf_dpi=300)
    assert dpi == 150.0
    assert abs(image.width - 2400) <= 2


@pytest.mark.parametrize("ext", [".png", ".pdf"])
def test_rotate_clockwise(tmp_path, ext):
    portrait = scan_like().rotate(90, expand=True)  # page scannée en portrait, ECG tourné de 90° antihoraire
    portrait.save(tmp_path / f"d{ext}", **({"resolution": 150} if ext == ".pdf" else {}))
    ((_, _, image, _),) = digitizer.iter_pages(tmp_path / f"d{ext}", rotate=90)
    assert image.width > image.height
    rgb = np.array(image)
    assert rgb[:, : image.width // 3].min() == 0  # le trait noir est revenu à gauche
