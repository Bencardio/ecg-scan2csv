# ecg-scan2csv

Convertit des **ECG papier scannés** en signaux numériques exploitables par l'application
[ecg-vcg](https://github.com/Bencardio/ecg-vcg) (et, au-delà, par BRAVEHEART).

Pour chaque scan, l'outil produit :

- `<nom_du_scan>.csv` : `t_s, I, II, III, aVR, aVL, aVF, V1…V6, rhythm_V1` à **500 Hz**, en **mV**,
  t = 0 au début du tracé, `NaN` hors de la fenêtre imprimée de chaque dérivation et là où rien n'a pu être lu ;
- `<nom_du_scan>.json` : mise en page, vitesse, gain (lu sur l'impulsion d'étalonnage), résolution du scan,
  % de NaN par dérivation, contrôle qualité, `source: "scan"`.

La segmentation et l'extraction des tracés reposent sur
[Open-ECG-Digitizer](https://github.com/Ahus-AIM/Open-ECG-Digitizer)
(Stenhede, Bjørnstad, Ranjbar, *npj Digital Medicine* 2026, [doi:10.1038/s41746-025-02327-1](https://doi.org/10.1038/s41746-025-02327-1)).

## Marche à suivre pour l'étude (PDF scannés, GE 3x4 + bande V1)

1. **Une fois par poste** : installer Python 3.12 ou plus récent ([python.org](https://www.python.org/downloads/) ;
   sous Windows, cocher « Add python.exe to PATH »), puis dans un terminal :

   ```bash
   python -m venv ecgscan-env
   ecgscan-env\Scripts\activate          # Windows ; macOS / Linux : source ecgscan-env/bin/activate
   pip install "ecgscan[digitizer] @ https://github.com/Bencardio/ecg-scan2csv/archive/main.zip"
   ecgscan setup                          # ~115 Mo, une seule fois
   ```

2. **Ranger les PDF** dans un dossier, par exemple `scans/`, nommés comme l'identifiant à apparier
   (`patient_012.pdf`). Un PDF de plusieurs pages donne `patient_012_p1.csv`, `patient_012_p2.csv`…
   Ce dossier ne doit jamais être placé dans le dépôt git.

3. **Convertir** (à chaque nouvelle session, réactiver l'environnement de l'étape 1) :

   ```bash
   ecgscan convert scans/ csv/ --layout 3x4 --rhythm V1 --speed 25
   ```

   L'écran affiche `OK` ou `!!` par ECG. Si une page est en portrait (ECG couché), le fichier est refusé avec un
   message : relancer ces fichiers avec `--rotate 90` (ou `270` s'ils sont tournés dans l'autre sens).

4. **Contrôler** dans `csv/ecgscan_summary.json` et dans le JSON de chaque ECG :
   `gain_source` doit valoir `calibration_pulse` (sinon relancer avec `--gain 10` après lecture de l'en-tête),
   `duration_s` ≈ 10, `qc.leads_over_threshold` vide, aucun avertissement sur `rhythm_V1` (corrélation avec
   le segment V1 de la grille, écrêtage des S profondes).

5. **Valider** une fois sur ~10 ECG disponibles aussi en PDF vectoriel (voir [Validation](#validation)),
   avant de lancer toute la série.

6. **Ouvrir les CSV** dans ecg-vcg (entrée « signal numérisé »), puis exporter vers BRAVEHEART.

## Installation (Windows, macOS, Linux)

Prérequis : **Python 3.12 ou plus récent** ([python.org](https://www.python.org/downloads/)). Git n'est pas nécessaire.

```bash
python -m venv ecgscan-env
# Windows : ecgscan-env\Scripts\activate      macOS / Linux : source ecgscan-env/bin/activate
pip install "ecgscan[digitizer] @ https://github.com/Bencardio/ecg-scan2csv/archive/main.zip"
ecgscan setup        # télécharge une seule fois le digitizer figé et ses poids (~115 Mo) dans ~/.ecgscan
ecgscan info         # vérifie l'installation et la présence d'un GPU
```

Avec une carte NVIDIA, installer d'abord PyTorch avec CUDA depuis [pytorch.org](https://pytorch.org/get-started/locally/)
(environ 6 s par image sur GPU, 17 s sur CPU d'après l'article). Le dossier de cache se change avec la variable
d'environnement `ECGSCAN_HOME`.

## Utilisation

```bash
# Étude pré-immunothérapie : GE 3x4, une bande rythme V1 de 10 s, 25 mm/s, gain lu sur l'impulsion
ecgscan convert scans/ csv/ --layout 3x4 --rhythm V1 --speed 25

# Autres mises en page
ecgscan convert scans/ csv/ --layout 3x4 --rhythm V1,II,V5      # MUSE 12SL avec trois bandes rythme
ecgscan convert scans/ csv/ --layout 6x2 --rhythm ""            # Dräger 6x2 sans bande rythme
ecgscan convert scans/ csv/ --layout 12x1
```

Options utiles : `--rotate 90|180|270` (pages scannées couchées), `--pdf-dpi 600`, `--gain 10` (impose le gain si l'impulsion n'est pas lue), `--device cpu|cuda:0`,
`--resample-size 4500` (résolution vue par le modèle, 3000 par défaut), `--save-raw` (garde les lignes brutes),
`--nan-threshold 20`.

Un résumé du lot est écrit dans `csv/ecgscan_summary.json` ; chaque avertissement est aussi affiché à l'écran.

### Préparer les scans

- Formats lus : **PDF de scanner** (une image par page, une page = un ECG), PNG, JPG, TIFF.
  Les pages PDF sont rendues à la résolution du scan qu'elles contiennent (notée dans `scan_dpi`, plafonnée à
  600 dpi ; `--pdf-dpi` pour l'imposer).
- Scanner à plat, **en couleur, 300 à 600 dpi**, page en paysage (sinon `--rotate`).
- Nommer chaque fichier comme l'ECG à apparier : le CSV et le JSON gardent ce nom (suffixe `_p2`… par page).
- Les photos fonctionnent mais sont nettement moins précises (corrélation 0,935–0,947 contre 0,987 pour les scans
  dans l'article).

## Ce que l'outil corrige dans la sortie brute du digitizer

| Sortie brute (`*_timeseries_canonical.csv`) | ecg-scan2csv |
| --- | --- |
| Gain codé en dur à 10 mm/mV | Gain mesuré sur l'impulsion d'étalonnage (ou imposé), amplitudes converties en mV réels |
| Largeur du tracé supposée = 10 s, rééchantillonnée à 5 000 points | Temps reconstruit depuis la taille de la grille (px/mm) et la vitesse ; durée vérifiée (10 ± 0,3 s) |
| Impulsion d'étalonnage incluse dans le tracé | Détectée, retirée ; t = 0 placé juste après |
| Bande rythme affectée par similarité (bonus pour II) et recopiée sur la dérivation de la grille | Mise en page déclarée ; bande rythme gardée à part (`rhythm_V1`) et comparée au segment V1 de la grille |
| µV, sans contrôle qualité | mV, % de NaN par dérivation, alerte d'écrêtage sur la bande rythme |

Le décalage continu de chaque fenêtre est retiré (médiane) ; aucun filtrage n'est appliqué.

## Validation

1. Choisir une dizaine d'ECG disponibles en PDF vectoriel, les imprimer, les scanner à 600 dpi.
2. `ecgscan convert scans/ csv_scan/ …`
3. Exporter les mêmes ECG depuis ecg-vcg au même format CSV dans `csv_pdf/`.
4. `ecgscan compare --scan csv_scan/ --ref csv_pdf/ --out comparaison.csv`
   → r, RMSE (µV) et décalage par dérivation, après recalage (± 200 ms).
5. Comparer ensuite les mesures VCG (angle QRS-T, SVG) calculées par ecg-vcg sur les deux versions.

Critère indicatif : r ≥ 0,95 et RMSE ≤ 50 µV par dérivation (scans de l'article : r 0,987, RMSE 19 µV).

## Points à surveiller

- **Impulsion d'étalonnage** : plateau de 80 à 320 ms et d'au moins 0,3 mV dans les 0,6 premières secondes de
  chaque rangée. Si `gain_source` vaut `assumed`, relancer avec `--gain` lu sur l'en-tête. Une erreur de t = 0 de
  quelques dizaines de ms décale toutes les rangées ensemble : l'alignement sur les R de la bande rythme est conservé.
- **Ondes S profondes en V1** : voir `qc.rhythm.rhythm_V1.clipped_low_ms`. Un chevauchement avec la rangée voisine
  donne des NaN plutôt qu'un écrêtage.
- **Résolution** : à 3000 px, environ 10 px/mm, soit ~250 échantillons/s avant le passage à 500 Hz.
- **Nombre de rangées** : il doit correspondre à la mise en page déclarée (3 + bandes rythme en 3x4),
  sinon le scan est rejeté avec un message explicite.

## Développement

```bash
pip install -e ".[test]"
python -m pytest              # tests sur signaux synthétiques, sans le modèle
ECGSCAN_INTEGRATION=1 python -m pytest tests/test_integration.py -s   # chaîne complète (réseau, ~1 Go avec PyTorch CPU)
```

La CI (GitHub Actions) exécute les tests sur Windows, macOS et Linux, et la chaîne complète sur CPU sur une page
synthétique.

## Licences

Code de ce dépôt : MIT. Open-ECG-Digitizer, téléchargé par `ecgscan setup` et non redistribué ici :
CC BY-SA 4.0. Ne jamais versionner de scans ni de CSV de patients (exclus par `.gitignore`).
