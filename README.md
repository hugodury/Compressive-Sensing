# Compressive-Sensing

Projet ING2 (CY Tech) : block compressive sensing sur une image, reconstruction patch par patch, comparaison de méthodes.

## Dépendances

À la racine du dépôt :

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install numpy pillow scipy
```

`scipy` sert surtout pour **BP/LP** (`linprog`). Le reste tourne avec numpy + pillow.

Lancer les scripts **depuis la racine** du repo (`compressive/`), sinon les imports `backend.*` plantent.

## Lancer le pipeline principal

`main.py` appelle `main_backend` : découpage, mesures simulées, reco pour chaque méthode demandée, métriques (MSE, PSNR, etc.).

```bash
cd /path/to/compressive
python3 main.py
```

Tu peux modifier le bloc `if __name__ == "__main__"` dans `main.py` (image, `methodes`, `ratio`, `dictionary_type`, `n_iter_ksvd`, …). Par défaut `n_iter_ksvd=0` : DCT ou mixte **sans** apprentissage ; passe par ex. `n_iter_ksvd=5` pour comparer avec un dictionnaire appris.

Exemple en une ligne (OMP + IRLS, peu de patchs pour aller vite) :

```bash
python3 -c "
from main import main
r = main(
    image_path='lena.jpg',
    block_size=8,
    ratio=0.25,
    methodes=['omp', 'irls'],
    dictionary_type='dct',
    patch_params={'max_patches': 30},
    method_params={
        'omp': {'max_iter': 40, 'epsilon': 1e-6},
        'irls': {'max_iter': 80, 'norm_p': 0.5},
    },
)
print(r['metrics'])
"
```

## Découpage / reco directe (`patch`)

Toujours depuis la racine :

```bash
python3 -c "from backend.Tratement_Image import patch; print(patch('lena.jpg', B=8, as_dict=True)['matrice_patchs'].shape)"

python3 -c "from backend.Tratement_Image import patch; o=patch('lena.jpg', B=8, ratio=0.2, method='omp', max_patches=40, as_dict=True); print(o['image_reconstruite'].shape)"
```

## Ce qui est en place (code)

- Découpage en patchs, `y = Phi x`, reconstruction avec `A = Phi @ D` puis `x_hat = D @ alpha`.
- **Méthodes** : MP, OMP, StOMP, CoSaMP, **IRLS** (pseudo-norme ℓp avec `0 < p < 1`, paramètre `norm_p`), **BP / lp**, **lasso** (ISTA).
- **Dictionnaire (K-SVD / DCT)** :
  - `dictionary_type='dct'` : DCT tronquée (fixe). Avec `n_iter_ksvd>0` (dans `patch` ou `patch_params` / racine des `params` pour `main_backend`), même init DCT puis **boucle K-SVD** (OMP sur les patchs + `learn_ksvd_dictionary`).
  - `dictionary_type='mixte'` : moitié DCT + moitié colonnes patchs (sans itération). Avec `n_iter_ksvd>0`, même init puis **K-SVD** à partir de ce mélange.
  - `ksvd_dct` / `ksvd_mixte` / `ksvd_random` : K-SVD obligatoire (init explicite) ; si `n_iter_ksvd` vaut 0, **10 itérations** par défaut.
  - `ksvd_train_patches` : nombre max de colonnes d’entraînement pour K-SVD (sinon tous les patchs).
  - La sortie `patch(..., as_dict=True)` peut contenir `ksvd_meta` (mode, init, itérations) quand un apprentissage a eu lieu.
- **CoSaMP** : option `s_cosamp_auto=True` dans `patch` (ou `patch_params` / `method_params`) pour estimer `s` comme au TD (OMP sur des patchs, médiane des supports). Sinon tu fixes `s` à la main.
- **Mesures** : `mesure.py` — modes `gaussian`, `uniform`, `bernoulli_1`, `bernoulli_01` (proches des Φ du cours, mais pas nommés Φ1…Φ4 dans le code).
- **Cohérence mutuelle** : `compute_coherence(Phi, D)` dans `mesure.py`.
- **Métriques** : `backend/utils/Metrics.py`.
- **K-SVD** : `learn_ksvd_full` + `learn_ksvd_dictionary` dans `Dictionnaire.py` ; câblé dans `patch` / `main_backend` via `n_iter_ksvd` et `dictionary_type`.

## Ce qu’il reste à faire (surtout sujet PDF + rapport)

- **Rapport** : comparer quantitativement DCT fixe, mixte sans itération, et dictionnaires appris (K-SVD depuis DCT / mixte / random) — PSNR, cohérence, temps.
- **Section 6 du PDF** : tous les pourcentages (15 % … 75 %), les **4 matrices** comme dans le cours, **tableaux** cohérence + **3 vecteurs** test + **erreurs relatives** pour chaque méthode — à produire (scripts ou notebook + figures).
- **Section 7** : image « hors entraînement », graphes, commentaires.
- **§5.2** : démos / questions théoriques → **rapport**, pas que du code.
- **Bonus** IHM : le `frontend/` est encore vide côté app utile.

Le fichier `fonction.md` décrit l’arborescence cible du projet si besoin.
