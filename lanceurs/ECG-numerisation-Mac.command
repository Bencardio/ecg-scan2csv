#!/bin/bash
# Lance l'application de numérisation d'ECG papier (ecgscan).
# Première utilisation : installe tout dans ~/ecgscan-env (quelques minutes, ~1 Go avec PyTorch).
# Prérequis : Python 3.12 ou plus récent (python.org).
ENV="$HOME/ecgscan-env"
SRC="ecgscan[digitizer] @ https://github.com/Bencardio/ecg-scan2csv/archive/main.zip"

if [ ! -x "$ENV/bin/python" ]; then
    echo "Première utilisation : installation de l'application, patientez quelques minutes..."
    PY=""
    for cand in python3.13 python3.12 python3; do
        if command -v "$cand" >/dev/null && "$cand" -c "import sys; sys.exit(sys.version_info < (3, 12))"; then
            PY="$cand"; break
        fi
    done
    if [ -z "$PY" ]; then
        echo "Python 3.12 ou plus récent est introuvable. Installez-le depuis https://www.python.org/downloads/ puis relancez."
        read -r -p "Appuyez sur Entrée pour fermer."; exit 1
    fi
    "$PY" -m venv "$ENV" && "$ENV/bin/python" -m pip install --upgrade pip && "$ENV/bin/python" -m pip install "$SRC" || {
        echo "L'installation a échoué : vérifiez la connexion internet et relancez."
        rm -rf "$ENV"; read -r -p "Appuyez sur Entrée pour fermer."; exit 1
    }
else
    echo "Recherche d'une mise à jour..."
    "$ENV/bin/python" -m pip install --quiet --upgrade "$SRC" >/dev/null 2>&1
fi

echo "L'application s'ouvre dans le navigateur. Fermez cette fenêtre (ou Ctrl+C) pour l'arrêter."
"$ENV/bin/ecgscan" app
