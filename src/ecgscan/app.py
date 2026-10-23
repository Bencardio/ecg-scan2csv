"""
Application Streamlit : numérisation d'ECG papier scannés (PDF, PNG, JPG, TIFF) au format d'échange d'ecg-vcg.

Lancement :
    ecgscan app

Tout s'exécute sur ce poste : les scans ne sont envoyés à aucun serveur. Outil de recherche, pas un dispositif
médical certifié : chaque tracé reconstruit doit être relu.
"""

import gc
import io
import json
import os
import tempfile
import zipfile

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from ecgscan import __version__, convert, digitizer, post
from ecgscan.plot import plot_ecg

SCAN_TYPES = [e.lstrip(".") for e in digitizer.INPUT_EXT]

st.set_page_config(page_title="Numérisation ECG papier", layout="wide")

st.title("Numérisation d'ECG papier scannés")
st.caption(
    "Outil de recherche (ecgscan v%s) : ECG papier scannés → signal 500 Hz en mV (CSV) + métadonnées (JSON), "
    "au format lu par Analyse ECG-VCG. Ne constitue pas un dispositif médical certifié — chaque tracé reconstruit "
    "doit être relu. Les scans sont traités sur cet ordinateur ; aucune donnée n'est envoyée à un serveur. "
    "Segmentation : Open-ECG-Digitizer (Stenhede et al., npj Digit Med 2026)." % __version__
)


@st.cache_resource(show_spinner="Chargement du modèle de numérisation…")
def _model(device, resample_size):
    return digitizer.Digitizer(device=device or None, resample_size=resample_size or None)


if not digitizer.is_installed():
    st.warning(
        "Le modèle de numérisation n'est pas encore installé sur ce poste (téléchargement unique d'environ 115 Mo)."
    )
    if st.button("Installer le modèle", type="primary"):
        with st.spinner("Téléchargement d'Open-ECG-Digitizer et de ses poids…"):
            digitizer.install(log=lambda *_: None)
        st.rerun()
    st.stop()

with st.sidebar:
    st.header("Fichiers")
    mode_depot = st.radio(
        "Charger", ["Fichiers", "Dossier"], horizontal=True,
        help="« Dossier » prend tous les scans du dossier choisi, sous-dossiers compris.",
    )
    if mode_depot == "Dossier":
        uploaded = st.file_uploader(
            "Choisir un dossier contenant des ECG scannés", accept_multiple_files="directory", key="depot_dossier"
        )
    else:
        uploaded = st.file_uploader(
            "Déposer un ou plusieurs ECG scannés (PDF, PNG, JPG, TIFF)",
            type=SCAN_TYPES, accept_multiple_files=True, key="depot_fichiers",
        )
    tous = list(uploaded or [])
    uploaded = [f for f in tous if os.path.splitext(f.name)[1].lower() in digitizer.INPUT_EXT]
    if tous:
        ignores = len(tous) - len(uploaded)
        st.caption("%d scan(s) prêt(s)%s." % (
            len(uploaded), " (%d autre(s) fichier(s) ignoré(s))" % ignores if ignores else ""))

    st.header("Format du tracé")
    layout = st.selectbox("Grille", ["3x4", "6x2", "12x1"], help="Disposition des 12 dérivations sur la page.")
    rhythm = st.text_input(
        "Bandes rythme (de haut en bas)", value="V1" if layout != "12x1" else "", disabled=layout == "12x1",
        help="Dérivations des bandes de 10 s sous la grille, séparées par des virgules : V1, ou V1,II,V5. "
             "Laisser vide s'il n'y en a pas.",
    )
    speed = st.radio("Vitesse (mm/s)", [25.0, 50.0], horizontal=True, format_func=lambda v: "%g" % v)
    gain_choice = st.radio(
        "Gain (mm/mV)", ["auto", "5", "10", "20"], horizontal=True,
        help="« auto » : lu sur l'impulsion d'étalonnage en début de rangée (recommandé). "
             "Imposer une valeur seulement si l'impulsion n'est pas lue.",
    )
    rotate = st.selectbox(
        "Rotation des pages", [0, 90, 180, 270], format_func=lambda d: "aucune" if d == 0 else "%d° (horaire)" % d,
        help="Pour des pages scannées couchées (portrait) : essayer 90°, sinon 270°.",
    )
    with st.expander("Réglages avancés"):
        nan_threshold = st.number_input("Seuil d'exclusion d'une dérivation (% de NaN)", 0.0, 100.0, 20.0, 5.0)
        resample_size = st.number_input(
            "Résolution vue par le modèle (px)", 1000, 8000, 3000, 500,
            help="Plus grand côté de l'image donné au modèle (entraîné à 3000).",
        )
        device = st.text_input("Calcul", value="", placeholder="automatique (GPU si disponible)",
                               help="cpu, cuda:0… ; vide = automatique")
    run = st.button("Numériser", type="primary", disabled=not uploaded)
    st.divider()
    st.caption(
        "Conseils : scan à plat, en couleur, 300 à 600 dpi, page en paysage, un ECG par page. "
        "Un PDF de plusieurs pages donne un ECG par page (suffixe _p1, _p2…)."
    )

if "results" not in st.session_state:
    st.session_state["results"] = []


def _preview(image, max_px=1600):
    im = image.copy()
    im.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


if run and uploaded:
    settings = post.Settings(
        layout=layout,
        rhythm_leads=[r.strip() for r in (rhythm or "").split(",") if r.strip()],
        speed_mm_s=float(speed),
        gain_mm_mV="auto" if gain_choice == "auto" else float(gain_choice),
        nan_threshold_pct=float(nan_threshold),
    )
    model = _model(device.strip(), int(resample_size))
    results = []
    progress = st.progress(0.0, text="Numérisation en cours…")
    with tempfile.TemporaryDirectory() as tmpdir:
        for k, f in enumerate(uploaded):
            nom = os.path.basename(f.name.replace("\\", "/"))
            progress.progress(k / len(uploaded), text="Numérisation %d/%d : %s" % (k + 1, len(uploaded), nom))
            path = os.path.join(tmpdir, "scan%d%s" % (k, os.path.splitext(nom)[1].lower()))
            with open(path, "wb") as out:
                out.write(f.getvalue())
            for rec in convert.convert_scan(model, path, settings, display_name=nom, rotate=int(rotate),
                                            keep_image=True):
                image = rec.pop("image", None)
                rec.pop("raw", None)
                if image is not None:
                    rec["preview"] = _preview(image)
                if rec["status"] == "ok":
                    rec["csv"] = post.csv_text(rec["data"], rec["header"])
                    rec["json"] = post.json_text(rec["meta"])
                base, n = rec["name"], 2
                while any(r["name"] == rec["name"] for r in results):  # même nom dans deux sous-dossiers
                    rec["name"], n = "%s (%d)" % (base, n), n + 1
                results.append(rec)
            gc.collect()
    progress.progress(1.0, text="Numérisation terminée : %d ECG." % len(results))
    st.session_state["results"] = results
    st.session_state["ecg_idx"] = 0

results = st.session_state["results"]

if not results:
    st.info("Déposez un ou plusieurs ECG scannés dans la barre latérale, vérifiez le format du tracé, puis cliquez "
            "sur « Numériser ».")
    st.stop()

# ---------------------------------------------------------------- tableau du lot
st.subheader("Résultats du lot (%d ECG)" % len(results))
rows = []
for r in results:
    m = r.get("meta") or {}
    qc = m.get("qc", {})
    nan = m.get("nan_pct", {})
    rows.append({
        "ECG": r["name"],
        "Statut": "✅ ok" if r["status"] == "ok" and not qc.get("warnings") else
                  ("⚠️ à vérifier" if r["status"] == "ok" else "❌ échec"),
        "Gain (mm/mV)": m.get("gain_mm_mV"),
        "Gain lu sur": {"calibration_pulse": "impulsion", "user": "imposé", "assumed": "supposé"}.get(
            m.get("gain_source"), m.get("gain_source")),
        "Durée (s)": m.get("duration_s"),
        "NaN max (%)": max(nan.values()) if nan else None,
        "Dérivations exclues": ", ".join(qc.get("leads_over_threshold", [])),
        "Avertissements / erreur": " ; ".join(qc.get("warnings", [])) if r["status"] == "ok" else r.get("error"),
        "dpi du scan": m.get("scan_dpi"),
    })
df = pd.DataFrame(rows)
st.dataframe(df, hide_index=True, width="stretch")

ok = [r for r in results if r["status"] == "ok"]
if ok:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for r in ok:
            z.writestr(r["name"] + ".csv", r["csv"])
            z.writestr(r["name"] + ".json", r["json"])
        summary = [{k: r.get(k) for k in ("name", "file", "page", "status", "error")}
                   | {"warnings": (r.get("meta") or {}).get("qc", {}).get("warnings")} for r in results]
        z.writestr("ecgscan_summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
        z.writestr("ecgscan_resultats.csv", df.to_csv(index=False))
    st.download_button(
        "Télécharger les ECG numérisés (%d, CSV + JSON, .zip)" % len(ok), buf.getvalue(),
        file_name="ecg_numerises.zip", mime="application/zip", type="primary", on_click="ignore",
    )
    st.caption("Chaque CSV : t_s, I…V6 et une colonne par bande rythme (rhythm_V1…), 500 Hz, mV, NaN hors de la "
               "fenêtre imprimée de chaque dérivation. À ouvrir dans Analyse ECG-VCG (entrée « signal numérisé »).")


# ---------------------------------------------------------------- détail
st.subheader("Contrôle par ECG")


@st.fragment
def _detail(results):
    names = [r["name"] for r in results]
    idx = st.session_state.get("ecg_idx", 0)
    if not 0 <= idx < len(names):
        idx = 0
    st.session_state["ecg_idx"] = idx

    def _go(step):
        st.session_state["ecg_idx"] = (st.session_state["ecg_idx"] + step) % len(names)

    def _pick():
        st.session_state["ecg_idx"] = names.index(st.session_state["ecg_pick"])

    st.session_state["ecg_pick"] = names[idx]
    c_prev, c_sel, c_next = st.columns([1, 6, 1], vertical_alignment="bottom")
    c_prev.button("◀ Précédent", on_click=_go, args=(-1,), width="stretch")
    c_sel.selectbox("Choisir un ECG (%d/%d)" % (idx + 1, len(names)), names, key="ecg_pick", on_change=_pick)
    c_next.button("Suivant ▶", on_click=_go, args=(1,), width="stretch")
    r = results[idx]

    if r["status"] != "ok":
        st.error(r.get("error"))
        if r.get("preview"):
            st.image(r["preview"], caption="Page scannée", width="stretch")
        return

    meta = r["meta"]
    warnings = meta["qc"]["warnings"]
    if warnings:
        with st.expander("⚠️ Avertissements (%d)" % len(warnings), expanded=True):
            for w in warnings:
                st.write("- " + w)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gain (%s)" % {"calibration_pulse": "lu sur l'impulsion", "user": "imposé", "assumed": "supposé"}[
        meta["gain_source"]], "%g mm/mV" % meta["gain_mm_mV"])
    c2.metric("Durée", "%.2f s" % meta["duration_s"])
    c3.metric("Mise en page", meta["layout"])
    c4.metric("Dérivations exclues", len(meta["qc"]["leads_over_threshold"]))

    tab1, tab2, tab3 = st.tabs(["Tracé reconstruit", "Scan d'origine", "Métadonnées (JSON)"])
    with tab1:
        st.caption("Signal reconstruit à l'échelle du papier (25 mm/s, 10 mm/mV) : à comparer au scan, en "
                   "particulier les ondes S profondes de la bande rythme et l'ordre des rangées.")
        fig = plot_ecg(r["data"], r["header"], meta)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    with tab2:
        if r.get("preview"):
            st.image(r["preview"], width="stretch")
    with tab3:
        st.json(meta, expanded=1)

    d1, d2 = st.columns(2)
    d1.download_button("Télécharger %s.csv" % r["name"], r["csv"], file_name=r["name"] + ".csv",
                       mime="text/csv", on_click="ignore", width="stretch")
    d2.download_button("Télécharger %s.json" % r["name"], r["json"], file_name=r["name"] + ".json",
                       mime="application/json", on_click="ignore", width="stretch")


_detail(results)
