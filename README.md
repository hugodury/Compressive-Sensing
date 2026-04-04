# Compressive sensing — reconstruction d’images (ING2)

Pipeline de **block compressive sensing (BCS)** : découpage en patchs B×B, mesures `y = Φx`, coefficients parcimonieux sur un dictionnaire `D` (DCT, mixte ou appris par **K-SVD**), recollement des blocs.  
Code principal : `backend/`, entrée CLI : `main.py`, interface : `frontend/`.

## Installation

À la racine du dépôt :

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

`matplotlib` sert aux graphiques ; le reste couvre numpy / scipy / BP-LP (voir `requirements.txt`).  
Les commandes suivantes supposent que le répertoire courant est la **racine du repo** (sinon `import backend` échoue).

## Interface graphique (IHM)

Bonus sujet : pilotage local sans tout passer par la ligne de commande.

```bash
python3 frontend/app.py
```

Lancer **depuis un terminal** à la racine (pas un double-clic sans console) pour voir les erreurs éventuelles.

**Onglets** : Accueil (rappel du pipeline + schéma), Reconstruction, Résultats, Comparaisons (sweep PSNR vs ratio), Cohérence & erreurs (export CSV §6), Patchs (une image avec **grille des blocs** sur l’image recadrée, même géométrie que le backend).  
Les pages longues ont un **défilement vertical** ; la molette cible le canvas sous le curseur.

Détails supplémentaires : `frontend/README_UI.md`.

## Lancer une reconstruction (CLI)

Image par défaut : `lena.jpg` à la racine (si présente).

```bash
python3 main.py -h
python3 main.py -i lena.jpg --etapes reconstruct,save
python3 main.py --etapes reconstruct,tableaux_s6 --no-tableaux-erreurs
python3 main.py --etapes reconstruct,sweep_graph --sweep-ratios 15,30,50
```

En Python : `from main import run_pipeline, setupParam` puis  
`run_pipeline(params, etapes=("reconstruct", "save", "tableaux_s6"))`.

## Critère d’arrêt PSNR (optionnel)

Arrêt lorsque le **patch reconstruit** atteint un PSNR cible (le vrai patch est connu en simulation) :

```python
patch_params={
    "psnr_stop": True,
    "psnr_target_db": 40.0,
}
```

S’applique aux solveurs itératifs qui utilisent `reference_for_psnr` (MP, OMP, StOMP, CoSaMP, IRLS, LASSO — pas BP/LP en une passe).

## CoSaMP et le paramètre `s`

- **Fixe** : `method_params["cosamp"]["s"]` (ou `s_cosamp` dans `patch`).
- **Estimé** : `patch_params["s_cosamp_auto"] = True` (médiane des supports OMP sur des patchs d’entraînement, même `D` que pour la reco).

Les métriques exposent `s_cosamp_utilise` et `cosamp_s_mode` (`fixe` ou `estime_omp`).

## Empreinte carbone (indicatif)

Aligné avec l’objectif « sensibilisation » du sujet : estimation **CO₂eq** à partir du temps CPU, d’une puissance machine supposée et d’un facteur g/kWh. Sortie **stderr** + `empreinte_estimation.txt` avec les résultats. Voir **`EMPREINTE.md`**.

Paramètres : `empreinte_carbone`, `empreinte_afficher_console`, `empreinte_puissance_w`, `empreinte_g_co2_par_kwh`. Avec `run_pipeline`, synthèse dans `empreinte_session` ; l’affichage par étape dans `main_backend` est limité pour éviter le double compte.

## Essai rapide (Python)

```python
from main import main

r = main(
    image_path="ton_image.png",
    block_size=8,
    ratio=0.25,      # ou 25 pour 25 % de mesures
    methodes=["omp", "cosamp"],
    dictionary_type="dct",
    measurement_mode="phi4",
    patch_params={"max_patches": 50},   # retirer pour toute l’image
    method_params={
        "omp": {"max_iter": 40, "epsilon": 1e-6},
        "cosamp": {"s": 6, "max_iter": 30},
    },
    seed=42,
)
print(r["metrics"])
```

Sauvegarde PNG + CSV dans `Data/Result/<date>/` :

```python
from backend.utils.save import save_results
save_results(r, "Data/Result")
```

## Fonctionnement (résumé)

1. **`backend/Tratement_Image.py`** : patchs, `Φ` (`mesure.py`), `D` (`Dictionnaire.py`), pour chaque bloc résolution du type `(ΦD)α ≈ y` via **`Methode.py`** (MP, OMP, StOMP, CoSaMP, IRLS, BP, LP, LASSO, …).
2. **`main_backend.py`** : enchaîne les méthodes, calcule PSNR / MSE / temps (`Metrics.py`).

StOMP : seuil `t` ; CoSaMP : `s` ou `s_cosamp_auto`.

## Paramètres utiles

| Thème | Détail |
|--------|--------|
| **Bloc** | `block_size` (= B) ; optionnel `patch_params` : `nrows`, `ncols`, `order`. |
| **Mesures** | `ratio` (0–1 = fraction, 1–100 = %) ou `M` ; `measurement_mode` / `mode_phi` : **`phi1` … `phi4`** (cours), ou alias `gaussian`, `uniform`, etc. |
| **Dictionnaire** | `dct` (DCT tronquée fixe) ; `mixte` (≈ moitié DCT + moitié colonnes de patchs) ; `ksvd` (K-SVD, init. aléatoire) ; `ksvd_dct` ; `ksvd_mixte`. `n_atoms`, `n_iter_ksvd`, `ksvd_train_patches`. §7 : **`dictionary_train_image_path`** (image test ≠ entraînement du dico). |
| **Solveurs** | `method_params[méthode]` : `max_iter`, `epsilon`, StOMP → `t`, CoSaMP → `s`, IRLS → `norm_p`, etc. |

Si `max_patches` &lt; nombre de blocs, seule une partie de l’image est reconstruite (le reste reste à zéro).

## Tableaux section 6 (PDF sujet)

Pourcentages **P = 15, 20, 25, 30, 50, 75** ; matrices **Φ₁–Φ₄** ; cohérence mutuelle **μ(Φ, D)** ; **trois vecteurs** de test et erreurs relatives pour **MP, OMP, StOMP, CoSaMP, IRLS**.

```bash
python3 -m backend.utils.projet_tableaux
```

Sortie typique : `Data/Result/<jj.mm.hh.mm>/Graph/` avec notamment `M_pour_P.csv`, `coherence_mutuelle.csv`, **`erreurs_relatives.csv`** (une seule table, colonne **`vecteur_test`** ∈ {1,2,3}).

Courbes PSNR vs ratios : `backend/utils/graphiques_projet.py` (matplotlib).

## Structure du dépôt (aperçu)

```
main.py                 # CLI, setupParam, run_pipeline
backend/
  Tratement_Image.py    # patch, mesures, reco par bloc
  main_backend.py
  utils/
    mesure.py           # Φ₁–Φ₄, P, M
    Dictionnaire.py     # DCT, mixte, K-SVD
    Methode.py          # MP, OMP, StOMP, CoSaMP, IRLS, BP, LP, LASSO
    projet_tableaux.py  # exports CSV §6
    graphiques_projet.py
    save.py, empreinte.py, …
frontend/
  app.py                # lance l’IHM Tk
  visualize_patches.py  # grille B×B sur image recadrée
  Pages/                # onglets Reconstruction, Résultats, …
Data/Result/            # sorties horodatées (gitignore conseillé)
EMPREINTE.md
fonction.md             # arborescence / rôles (complément)
```

## Sujet PDF (CY Tech / stockage images)

Une copie du sujet peut être placée à la racine (ex. **`Projet CS - Stockage images (1).pdf`**). Le rendu **code** est aligné sur les exigences principales : BCS, K-SVD, méthodes §5–§6, Φ₁–Φ₄, tableaux §6, §7 (image hors entraînement), IHM bonus, empreinte.  
**Hors dépôt ou hors code** : rapport (analyse, commentaires, critique), démonstrations math du §5.2 si demandées à l’écrit.

## Archive de rendu (conseil)

Inclure : sources, `requirements.txt`, ce README, éventuellement `EMPREINTE.md` et le PDF du sujet.  
**Exclure** : dossiers lourds de résultats si non demandés, fichiers accidentels type archives `.zip` dupliquées ou artefacts non sources.

## Déjà fait / reste côté rendu

| Fait dans le repo | À charge du groupe |
|-------------------|---------------------|
| BCS, Φ₁–Φ₄, P du sujet, CSV §6, métriques, graphes, K-SVD + DCT/mixte, extensions BP/LP/LASSO, IHM, empreinte | Rapport, figures commentées, preuves §5.2 si au programme |

Pour une vue fonctionnelle détaillée des modules : **`fonction.md`**.
