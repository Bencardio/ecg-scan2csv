"""Conversion d'un scan et application, avec un faux modèle (le vrai est testé dans test_integration.py)."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from ecgscan import cli, convert, digitizer, post

from test_post import PX_PER_MM, make_lines


class FakeModel:
    device = "cpu"
    config = SimpleNamespace(MODEL=SimpleNamespace(KWARGS=SimpleNamespace(resample_size=3000)))

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, image):
        lines, *_ = make_lines()
        return {"lines_uV": lines, "px_per_mm_x": PX_PER_MM, "layout_guess": "3x4+1"}


def landscape(path, pages=1, portrait=False):
    size = (850, 1200) if portrait else (1200, 850)
    ims = [Image.new("RGB", size, (255, 240, 240)) for _ in range(pages)]
    ims[0].save(path, save_all=True, append_images=ims[1:], resolution=150)


def test_convert_scan_names_each_pdf_page(tmp_path):
    landscape(tmp_path / "patient_012.pdf", pages=2)
    recs = convert.convert_scan(FakeModel(), tmp_path / "patient_012.pdf", post.Settings())
    assert [(r["name"], r["status"]) for r in recs] == [("patient_012_p1", "ok"), ("patient_012_p2", "ok")]
    assert recs[1]["meta"]["page"] == 2 and recs[1]["meta"]["scan_dpi"] == 150.0
    assert recs[0]["header"][-1] == "rhythm_V1"


def test_portrait_page_is_refused_with_hint(tmp_path):
    landscape(tmp_path / "couche.pdf", portrait=True)
    (rec,) = convert.convert_scan(FakeModel(), tmp_path / "couche.pdf", post.Settings())
    assert rec["status"] == "error" and "rotation" in rec["error"]
    (rec,) = convert.convert_scan(FakeModel(), tmp_path / "couche.pdf", post.Settings(), rotate=90)
    assert rec["status"] == "ok"


def test_cli_convert_writes_csv_and_json(tmp_path, monkeypatch):
    monkeypatch.setattr(digitizer, "is_installed", lambda: True)
    monkeypatch.setattr(digitizer, "Digitizer", FakeModel)
    (tmp_path / "scans").mkdir()
    landscape(tmp_path / "scans" / "a.pdf")
    landscape(tmp_path / "scans" / "b.pdf", portrait=True)
    code = cli.main(["convert", str(tmp_path / "scans"), str(tmp_path / "out")])
    assert code == 2  # b.pdf refusé
    out = tmp_path / "out"
    assert (out / "a.csv").read_text().startswith("t_s,I,II,III")
    assert json.loads((out / "a.json").read_text())["source"] == "scan"
    summary = json.loads((out / "ecgscan_summary.json").read_text())
    assert [e["status"] for e in summary] == ["ok", "error"]


def test_plot_ecg_draws_all_rows():
    from ecgscan.plot import plot_ecg

    lines, *_ = make_lines()
    data, header, meta = post.process(lines, PX_PER_MM, post.Settings())
    fig = plot_ecg(data, header, meta)
    texts = {t.get_text().split()[0] for t in fig.axes[0].texts}
    assert texts == set(header[1:])


def test_app_renders_results(monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(digitizer, "is_installed", lambda: True)
    lines, *_ = make_lines()
    data, header, meta = post.process(lines, PX_PER_MM, post.Settings())
    ok = {"name": "e1", "file": "e1.pdf", "page": 1, "status": "ok", "data": data, "header": header, "meta": meta,
          "csv": post.csv_text(data, header), "json": post.json_text(meta), "preview": _jpeg()}
    bad = {"name": "e2", "file": "e2.pdf", "page": 1, "status": "error", "error": "page en portrait"}
    at = AppTest.from_file(_app_path(), default_timeout=60)
    at.session_state["results"] = [ok, bad]
    at.run()
    assert not at.exception, at.exception
    assert at.dataframe[0].value["Statut"].tolist()[1] == "❌ échec"
    assert any("e1" in str(s.value) for s in at.selectbox)


def _jpeg():
    import io

    buf = io.BytesIO()
    Image.new("RGB", (120, 85), (255, 240, 240)).save(buf, format="JPEG")
    return buf.getvalue()


def _app_path():
    import importlib.util

    return importlib.util.find_spec("ecgscan").submodule_search_locations[0] + "/app.py"


def test_app_offers_install_when_model_missing(monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(digitizer, "is_installed", lambda: False)
    at = AppTest.from_file(_app_path(), default_timeout=60)
    at.run()
    assert not at.exception
    assert at.button[0].label == "Installer le modèle"


def test_ecgscan_app_serves_page():
    pytest.importorskip("streamlit")
    import socket
    import subprocess
    import sys
    import time
    import urllib.request

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    proc = subprocess.Popen([sys.executable, "-m", "ecgscan", "app", "--no-browser", "--port", str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                with urllib.request.urlopen(f"http://localhost:{port}/_stcore/health", timeout=2) as r:
                    assert r.read() == b"ok"
                    return
            except OSError:
                time.sleep(1)
        pytest.fail("l'application n'a pas démarré")
    finally:
        proc.terminate()
        proc.wait(timeout=20)


def test_convert_resumes_after_interruption(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(digitizer, "is_installed", lambda: True)
    monkeypatch.setattr(digitizer, "Digitizer", FakeModel)
    (tmp_path / "scans").mkdir()
    for name in ("a", "b", "c"):
        landscape(tmp_path / "scans" / f"{name}.pdf")
    out = tmp_path / "out"
    assert cli.main(["convert", str(tmp_path / "scans"), str(out)]) == 0
    (out / "c.json").unlink()  # comme si le lot s'était arrêté avant c
    calls = []
    monkeypatch.setattr(convert, "convert_scan", lambda m, path, *a, **k: calls.append(path.name) or [])
    assert cli.main(["convert", str(tmp_path / "scans"), str(out)]) == 0
    assert calls == ["c.pdf"]
    assert "déjà converti" in capsys.readouterr().out


def test_digitizer_downscales_before_model():
    seen = {}

    def wrapper(tensor, **kwargs):
        seen["shape"] = tuple(tensor.shape)
        return {"signal": {"lines": np.zeros((4, 10))}, "pixel_spacing_mm": {"x": 0.1}, "layout_name": "x"}

    pytest.importorskip("torch")
    model = digitizer.Digitizer.__new__(digitizer.Digitizer)
    model.config = FakeModel.config
    model.wrapper = wrapper
    model(Image.new("RGB", (7000, 4950), "white"))
    assert seen["shape"] == (1, 3, 2121, 3000)
