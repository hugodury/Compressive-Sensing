# Compressive-Sensing

Projet ING2 (CY Tech) : block compressive sensing, reconstruction patch par patch, dictionnaires DCT / mixte / K-SVD, matrices Φ₁–Φ₄ du cours.

## Dépendances

À la racine du dépôt :

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install numpy pillow scipy
```

- **scipy** : BP/LP (`linprog`).
- **matplotlib** (optionnel) : courbes PSNR vs ratio (`backend.utils.graphiques_projet`).

Lancer les scripts **depuis la racine** du repo, sinon les imports `backend.*` échouent.

---

## Conformité sujet PDF « Projet CS – Stockage images » (backend)

| Partie du sujet | Exigence | Où c’est fait |
|-----------------|----------|----------------|
| §3 BCS | Découpage B×B, yⱼ = Φ_B xⱼ, reco x̂ⱼ | `Tratement_Image.patch`, `mesure.py` |
| §4 K-SVD | Entraîner D sur imagettes, comparer à DCT | `Dictionnaire.learn_ksvd_full`, `dictionary_type` + `n_iter_ksvd` |
| §5.1 | MP, OMP, StOMP | `Methode.py` |
| §5.1 | CoSaMP + ordre s (dont déduction depuis KSVD) | `cosamp`, `s_cosamp_auto` + `estime_ordre_parcimonie_cosamp` |
| §5.2 IRLS ℓp | Implémentation (démos théoriques = rapport) | `irls` |
| Cours / pipeline | BP, LP, LASSO | `basis_pursuit`, `lp`, `lasso_ista` |
| §6 | P ∈ {15,…,75} %, M = ⌈PN/100⌉ | `mesure.pourcentage_vers_M`, `POURCENTAGES_MESURES_PROJET` |
| §6 | Φ₁…Φ₄, tableau cohérence μ(Φ,D) | `resolve_measurement_mode` (phi1…phi4), `compute_coherence_cours_phi_d`, `projet_tableaux.py` |
| §6 | 3 vecteurs, erreurs relatives MP…IRLS | `trois_vecteurs_validation`, `tableau_erreurs_relatives_vecteurs`, CSV exportés |
| §7 | Image **hors** entraînement du dictionnaire | `dictionary_train_image_path` dans `patch` / `main` / `setupParam` (D appris sur une autre image) |
| §7 | Métriques + graphiques | `Metrics.py`, `save_results`, `graphiques_projet.sweep_ratios_psnr` |
| Bonus IHM | Interface | Non réalisé (`frontend/` volontairement hors périmètre) |

Les questions purement **mathématiques / rédaction** (§5.2 reformulations, commentaires des tableaux, critique) restent du **rapport**, pas du dépôt.

---

## Lancer le pipeline principal

`main.py` → `main_backend` : découpage, mesures, reconstruction, métriques.

**`ratio`** dans `setupParam` / `main` : **fraction** dans `]0, 1]` (ex. `0.25`) **ou** **pourcentage** dans `]0, 100]` (ex. `25`), comme dans `patch`.

```bash
cd /path/to/compressive
python3 main.py
```

Exemple rapide (OMP + IRLS, peu de patchs) :

```bash
python3 -c "
from main import main
r = main(
    image_path='lena.jpg',
    block_size=8,
    ratio=25,
    methodes=['omp', 'irls'],
    dictionary_type='dct',
    measurement_mode='phi4',
    patch_params={'max_patches': 30},
    method_params={
        'omp': {'max_iter': 40, 'epsilon': 1e-6},
        'irls': {'max_iter': 80, 'norm_p': 0.5},
    },
)
print(r['metrics'])
"
```

### §7 — Dictionnaire appris sur une autre image

```python
main(
    image_path="test.png",              # image à reconstruire
    dictionary_train_image_path="train.png",  # patchs pour mixte / K-SVD uniquement
    dictionary_type="ksvd_mixte",
    n_iter_ksvd=5,
    ...
)
# ou patch_params={'dictionary_train_image_path': 'train.png'}
```

Même **B** et même grille (`nrows` / `ncols` dans `patch_params` si tu les fixes) pour que N = B² soit identique.

---

## Tableaux section 6 (CSV)

```bash
python3 -m backend.utils.projet_tableaux
```

Génère sous `Data/Result/jj.mm.hh.mm/Graph/` : `M_pour_P.csv`, `coherence_mutuelle.csv`, `erreurs_relatives_vecteur*.csv`.  
Tu peux passer ton propre `D` appris (charger `.npy` puis `exporter_tableaux_section6(D, N, ...)` en Python).

---

## Graphiques PSNR vs ratio

```python
from main import setupParam
from backend.utils.graphiques_projet import exporter_sweep_graphique

params = setupParam(
    image_path="lena.jpg",
    block_size=8,
    ratio=0.25,
    methodes=["omp", "mp"],
    dictionary_type="dct",
    measurement_mode="phi4",
    patch_params={"max_patches": 40},
)
exporter_sweep_graphique(params, [15, 25, 50, 75], output_path="Data/Result")
```

Nécessite `pip install matplotlib`.

---

## Découpage seul (`patch`)

```bash
python3 -c "from backend.Tratement_Image import patch; print(patch('lena.jpg', B=8, as_dict=True)['matrice_patchs'].shape)"
```

**Qualité de reconstruction :** ne pas utiliser `max_patches` inférieur au nombre total de patchs pour une image « finale » : les blocs non traités restent à **0** (zones noires). Pour un test rapide c’est OK ; pour un rendu visuel, omettre `max_patches` ou le mettre au nombre total de patchs. Les images très « hachées » (damier, bruit fin) sont **peu parcimonieuses en DCT** : avec peu de mesures (petit ratio), le CS donne un PSNR médiocre — c’est attendu ; préférer des images naturelles et un ratio plus élevé (ex. 50–75 %).

---

## Référence rapide code

- **Mesures** : `gaussian`, `uniform`, `bernoulli_1`, `bernoulli_01` ou **`phi1`…`phi4`**.
- **Dictionnaires** : `dct`, `mixte`, `ksvd_*`, `n_iter_ksvd`, `ksvd_train_patches`, `dictionary_train_image_path`.
- **Cohérence cours** : `coherence_mutuelle_cours` dans la sortie `patch` et dans les métriques `main_backend` ; CSV métriques enrichi dans `save_results` si présent.

Le fichier `fonction.md` décrit l’arborescence cible (dont `projet_tableaux.py`, `graphiques_projet.py`).
