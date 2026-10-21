"""Chaîne complète sur une page synthétique ; nécessite « pip install .[digitizer] » et le réseau.

Lancé seulement si ECGSCAN_INTEGRATION=1 (job dédié de la CI).
"""

import json
import os

import pytest

from ecgscan import cli

from synthetic_page import render

pytestmark = pytest.mark.skipif(os.environ.get("ECGSCAN_INTEGRATION") != "1", reason="test d'intégration désactivé")


def test_full_chain_on_synthetic_page(tmp_path):
    scans = tmp_path / "scans"
    scans.mkdir()
    render(str(scans / "page.png"))
    # Même page en PDF de scanner (une image par page), comme les ECG de l'étude
    from PIL import Image

    with Image.open(scans / "page.png") as im:
        im.convert("RGB").save(scans / "page_pdf.pdf", resolution=300)
    assert cli.main(["setup"]) == 0
    code = cli.main(["convert", str(scans), str(tmp_path / "out"), "--device", "cpu"])
    summary = json.loads((tmp_path / "out" / "ecgscan_summary.json").read_text())
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    assert summary[0]["status"] == "ok", summary[0].get("error")
    meta = json.loads((tmp_path / "out" / "page.json").read_text())
    print(json.dumps({k: meta[k] for k in ("gain_mm_mV", "gain_source", "duration_s", "nan_pct", "qc")}, indent=2))
    assert (tmp_path / "out" / "page.csv").read_text().startswith("t_s,I,II,III")
    assert code in (0, 2)
    by_file = {e["file"]: e for e in summary}
    assert by_file["page_pdf.pdf"]["status"] == "ok", by_file["page_pdf.pdf"].get("error")
    meta_pdf = json.loads((tmp_path / "out" / "page_pdf.json").read_text())
    assert meta_pdf["scan_dpi"] == 300.0
    assert abs(meta_pdf["gain_mm_mV"] - meta["gain_mm_mV"]) < 0.5
