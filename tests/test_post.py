"""Tests du post-traitement sur des lignes synthétiques (sans le modèle) : python -m pytest"""

import numpy as np
import pytest

from ecgscan import post

PX_PER_MM = 10.1  # ordre de grandeur après le redimensionnement à 3000 px du digitizer
SPEED = 25.0


def synthetic_ecg(t: np.ndarray, lead_idx: int) -> np.ndarray:
    """ECG grossier en µV : P, QRS de type rS (V1) ou qR selon la dérivation, T ; 75 bpm."""
    sig = np.zeros_like(t)
    amp = 1.0 + 0.1 * lead_idx
    rs = lead_idx == 6  # V1
    for beat in np.arange(0.3, 10, 0.8):
        sig += 120 * amp * np.exp(-((t - beat + 0.16) ** 2) / (2 * 0.02**2))
        sig += (300 if rs else 1200) * amp * np.exp(-((t - beat) ** 2) / (2 * 0.012**2))
        sig -= (1400 if rs else 250) * amp * np.exp(-((t - beat - 0.03) ** 2) / (2 * 0.012**2))
        sig += 250 * amp * np.exp(-((t - beat - 0.3) ** 2) / (2 * 0.05**2))
    return sig


def make_lines(gain=10.0, rhythm="V1", with_pulse=True, nan_lead=None):
    dt = 1.0 / (PX_PER_MM * SPEED)
    t = np.arange(0, 10, dt)
    truth = {lead: synthetic_ecg(t, i) for i, lead in enumerate(post.LEADS)}
    read_factor = gain / 10.0  # le digitizer lit en supposant 10 mm/mV
    rows = []
    for row in post.GRIDS["3x4"]:
        r = np.empty_like(t)
        for c, lead in enumerate(row):
            sl = slice(c * len(t) // 4, (c + 1) * len(t) // 4)
            r[sl] = truth[lead][sl]
        rows.append(r)
    rows.append(truth[rhythm].copy())
    pulse_n = int(round(0.2 / dt))
    pre_n = int(round(0.05 / dt))
    post_n = int(round(0.02 / dt))  # le tracé reprend 20 ms après le front descendant
    out = []
    for r in rows:
        r = r * read_factor
        if with_pulse:
            pulse = np.r_[np.zeros(pre_n), np.full(pulse_n, 1000.0 * read_factor), np.zeros(post_n)]
            r = np.r_[pulse, r]
        out.append(r)
    lines = np.vstack(out)
    if nan_lead is not None:
        # lead I : rangée 0, colonne 0 ; on efface 30 % de sa fenêtre
        off = (pulse_n + pre_n + post_n) if with_pulse else 0
        n = len(t) // 4
        lines[0, off + n // 10 : off + n // 10 + int(0.3 * n)] = np.nan
    return lines, t, truth


def corr_with_truth(data, header, t, truth, lead):
    j = header.index(lead)
    y = data[:, j]
    ref = np.interp(data[:, 0], t, truth[lead]) / 1000.0
    ok = ~np.isnan(y)
    return np.corrcoef(y[ok], ref[ok])[0, 1], ok.sum()


def test_gain_time_and_layout():
    lines, t, truth = make_lines(gain=10.0)
    data, header, meta = post.process(lines, PX_PER_MM, post.Settings())
    assert header == ["t_s"] + post.LEADS + ["rhythm_V1"]
    assert meta["gain_mm_mV"] == pytest.approx(10.0, abs=0.3)
    assert meta["gain_source"] == "calibration_pulse"
    assert meta["duration_s"] == pytest.approx(10.0, abs=0.1)
    assert meta["fs_hz"] == 500.0
    assert abs(len(data) - 5000) <= 50
    # fenêtres 3x4 et bande rythme
    assert meta["lead_windows_s"]["I"][0] == pytest.approx(0.0, abs=0.05)
    assert meta["lead_windows_s"]["V1"][0] == pytest.approx(5.0, abs=0.1)
    assert meta["lead_windows_s"]["V4"][1] == pytest.approx(10.0, abs=0.1)
    for lead in ["I", "aVF", "V1", "V6", "rhythm_V1"]:
        name = "V1" if lead == "rhythm_V1" else lead
        j = header.index(lead)
        y = data[:, j]
        ref = np.interp(data[:, 0], t, truth[name]) / 1000.0
        ok = ~np.isnan(y)
        assert np.corrcoef(y[ok], ref[ok])[0, 1] > 0.99, lead
    # I hors de sa fenêtre = NaN
    jI = header.index("I")
    assert np.isnan(data[data[:, 0] > 3.0, jI]).all()
    assert meta["qc"]["rhythm"]["rhythm_V1"]["corr_with_grid_segment"] > 0.99
    assert not meta["qc"]["leads_over_threshold"]


def test_gain_5mm_rescales_to_mV():
    lines, t, truth = make_lines(gain=5.0)
    data, header, meta = post.process(lines, PX_PER_MM, post.Settings())
    assert meta["gain_mm_mV"] == pytest.approx(5.0, abs=0.2)
    j = header.index("rhythm_V1")
    ref = np.interp(data[:, 0], t, truth["V1"]) / 1000.0
    ok = ~np.isnan(data[:, j])
    # amplitudes en mV réels malgré le tracé à demi-gain
    ratio = np.ptp(data[ok, j]) / np.ptp(ref[ok])
    assert ratio == pytest.approx(1.0, abs=0.05)


def test_no_pulse_falls_back_and_warns():
    lines, *_ = make_lines(with_pulse=False)
    _, _, meta = post.process(lines, PX_PER_MM, post.Settings())
    assert meta["gain_source"] == "assumed"
    assert any("impulsion" in w for w in meta["qc"]["warnings"])


def test_nan_threshold_flags_lead():
    lines, *_ = make_lines(nan_lead="I")
    _, _, meta = post.process(lines, PX_PER_MM, post.Settings())
    assert meta["nan_pct"]["I"] > 20
    assert meta["qc"]["leads_over_threshold"] == ["I"]


def test_wrong_rhythm_assignment_is_flagged():
    lines, *_ = make_lines(rhythm="II")  # la bande imprimée est II, mais on déclare V1
    _, _, meta = post.process(lines, PX_PER_MM, post.Settings(rhythm_leads=["V1"]))
    assert any("ressemble peu" in w for w in meta["qc"]["warnings"])


def test_wrong_row_count_raises():
    lines, *_ = make_lines()
    with pytest.raises(ValueError):
        post.process(lines[:3], PX_PER_MM, post.Settings())


def test_wrong_speed_warns():
    lines, *_ = make_lines()
    _, _, meta = post.process(lines, PX_PER_MM, post.Settings(speed_mm_s=50.0))
    assert any("durée lue" in w for w in meta["qc"]["warnings"])


def test_write_outputs(tmp_path):
    lines, *_ = make_lines()
    data, header, meta = post.process(lines, PX_PER_MM, post.Settings())
    post.write_outputs(data, header, meta, str(tmp_path / "x.csv"), str(tmp_path / "x.json"))
    first = (tmp_path / "x.csv").read_text().splitlines()[:2]
    assert first[0].startswith("t_s,I,II,III,aVR")
    assert "NaN" in first[1]
