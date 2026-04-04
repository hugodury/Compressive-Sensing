# Compressive sensing — reconstruction d’images (ING2)

Petit pipeline de block compressive sensing : on découpe l’image en blocs, on simule des mesures `y = Φx` par patch, on cherche des coefficients parcimonieux dans un dictionnaire `D` (souvent DCT, parfois appris au K-SVD), puis on recolle les blocs. Le code vit dans `backend/`, l’entrée simple est `main.py`.

## Installation

Depuis la racine du projet :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pillow scipy
```

`scipy` sert surtout pour BP/LP. Pour tracer des courbes PSNR en fonction du ratio, ajoute `matplotlib` si besoin.

Toutes les commandes ci-dessous supposent que tu es **à la racine du repo** (sinon `import backend` ne marchera pas).

## Lancer une reco

Le script par défaut attend une image `lena.jpg` à la racine (à toi de la mettre ou de changer le chemin dans `main.py`).

```bash
python3 main.py
```

Sinon, depuis Python :

```python
from main import main

r = main(
    image_path="ton_image.png",
    block_size=8,
    ratio=0.25,      # ou 25 pour 25 % de mesures
    methodes=["omp", "cosamp"],
    dictionary_type="dct",
    measurement_mode="phi4",   # ou gaussian, phi1, etc.
    patch_params={"max_patches": 50},   # enlève ça pour toute l’image
    method_params={
        "omp": {"max_iter": 40, "epsilon": 1e-6},
        "cosamp": {"s": 6, "max_iter": 30},
    },
    seed=42,
)
print(r["metrics"])
```

Pour sauver images + CSV dans `Data/Result/<date>/` :

```python
from backend.utils.save import save_results
save_results(r, "Data/Result")
```

## Comment ça marche (résumé)

1. `Tratement_Image.patch` découpe l’image en patchs, construit `Φ` (`mesure.py`), éventuellement `D` (`Dictionnaire.py`), puis pour chaque patch résout quelque chose du type `(ΦD)α ≈ y` avec MP, OMP, StOMP, CoSaMP, IRLS, BP, LASSO, etc. (`Methode.py`).
2. `main_backend` enchaîne les méthodes demandées et calcule PSNR / MSE / temps (`Metrics.py`).

StOMP utilise surtout le seuil `t` ; CoSaMP utilise `s`, ou bien `s_cosamp_auto` dans `patch_params` pour estimer `s` à partir d’OMP sur les patchs (utile quand `D` vient du K-SVD).

## Paramètres utiles

- **Bloc** : `block_size` (= B), ou grille via `patch_params` : `nrows`, `ncols`, `order`.
- **Mesures** : `ratio` (entre 0 et 1 = fraction, entre 1 et 100 = pourcentage) ou `M` / `patch_params["M"]`. Mode de `Φ` : `measurement_mode` ou `patch_params["mode_phi"]` (`phi1` … `phi4` comme au cours, ou `gaussian`, `uniform`, etc.).
- **Dictionnaire** : `dictionary_type` (`dct`, `mixte`, `ksvd_dct`, …), `n_atoms`, `n_iter_ksvd`, `ksvd_train_patches`. Pour le sujet §7 (image test ≠ image d’entraînement) : `dictionary_train_image_path` dans `main` ou dans `patch_params`.
- **Solveurs** : dans `method_params[nom_méthode]` : `max_iter`, `epsilon`, pour StOMP `t`, pour CoSaMP `s`, pour IRLS `norm_p`, etc.

Si tu mets `max_patches` plus petit que le nombre réel de blocs, seule une partie de l’image est reconstruite — le reste reste à zéro (effet visuel brutal, normal).

## Tableaux du sujet (section 6)

Un module génère les CSV (M pour chaque %, cohérence mutuelle Φ₁–Φ₄, erreurs sur trois vecteurs test) :

```bash
python3 -m backend.utils.projet_tableaux
```

Sortie typique : `Data/Result/<jj.mm.hh.mm>/Graph/`. Tu peux aussi appeler `exporter_tableaux_section6` depuis Python avec ton propre `D`.

Courbes PSNR vs plusieurs ratios : voir `backend/utils/graphiques_projet.py` (nécessite matplotlib).

## Déjà fait / pas fait

**Côté code backend** : BCS, dictionnaires DCT + mixte + K-SVD, méthodes demandées dans le sujet et extensions type BP/LP/LASSO, matrices de mesure du cours (dont alias `phi1`–`phi4`), tableaux §6, métriques, sauvegarde, option image d’entraînement séparée pour le dico, scripts de graphes.

**Pas dans ce dépôt** : l’interface graphique (bonus IHM), et tout ce qui est **rapport** : rédaction, commentaires des résultats, démos math du PDF §5.2, choix argumenté des réglages pour les figures.

Pour l’arborescence des fichiers visée par le sujet, voir aussi `fonction.md`.
