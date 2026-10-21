@echo off
rem Lance l'application de numérisation d'ECG papier (ecgscan).
rem Première utilisation : installe tout dans %USERPROFILE%\ecgscan-env (quelques minutes, ~1 Go avec PyTorch).
rem Prérequis : Python 3.12 ou plus récent (python.org, cocher "Add python.exe to PATH").
setlocal
set "ENV=%USERPROFILE%\ecgscan-env"
set "SRC=ecgscan[digitizer] @ https://github.com/Bencardio/ecg-scan2csv/archive/main.zip"

if not exist "%ENV%\Scripts\python.exe" (
    echo Premiere utilisation : installation de l'application, patientez quelques minutes...
    py -3 -c "import sys; sys.exit(sys.version_info < (3, 12))" >nul 2>&1 && (py -3 -m venv "%ENV%") || (
        python -c "import sys; sys.exit(sys.version_info < (3, 12))" >nul 2>&1 && (python -m venv "%ENV%") || (
            echo Python 3.12 ou plus recent est introuvable. Installez-le depuis https://www.python.org/downloads/
            echo en cochant "Add python.exe to PATH", puis relancez ce fichier.
            pause
            exit /b 1
        )
    )
    "%ENV%\Scripts\python.exe" -m pip install --upgrade pip
    "%ENV%\Scripts\python.exe" -m pip install "%SRC%" || (
        echo L'installation a echoue : verifiez la connexion internet et relancez.
        rmdir /s /q "%ENV%"
        pause
        exit /b 1
    )
) else (
    echo Recherche d'une mise a jour...
    "%ENV%\Scripts\python.exe" -m pip install --quiet --upgrade "%SRC%" >nul 2>&1
)

echo L'application s'ouvre dans le navigateur. Fermez cette fenetre pour l'arreter.
"%ENV%\Scripts\ecgscan.exe" app
pause
