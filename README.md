# Compressive-Sensing

Repo du projet (ING2), pour l’instant y’a surtout le découpage d’image en patchs.

## deps

```bash
pip install numpy pillow
```

(ou un venv si tu préfères)

## script

`backend/Tratement_Image.py` : charge une image, découpe en blocs B×B, sort une matrice avec une colonne = un patch aplati (et peut reconstruire si on donne `ratio`/`Phi`).

Par défaut ça prend `lena.jpg` à la racine du dépôt.

```bash
python3 -c "from backend.Tratement_Image import patch; print(patch('lena.jpg', B=8)['matrice_patchs'].shape)"
python3 -c "from backend.Tratement_Image import patch; print(patch('lena.jpg', B=16)['matrice_patchs'].shape)"
python3 -c "from backend.Tratement_Image import patch; print(patch('lena.jpg', nrows=32, ncols=32)['matrice_patchs'].shape)"
python3 -c "from backend.Tratement_Image import patch; print(patch('autre_fichier.png', B=8)['matrice_patchs'].shape)"

# reconstruction (ex : 20% de mesures, OMP), limité à un petit nombre de patchs pour rester rapide
python3 -c \"from backend.Tratement_Image import patch; out=patch('lena.jpg', B=8, ratio=0.2, method='omp', n_atoms=64, max_patches=50); print(out['image_reconstruite'].shape)\" 
```

Le reste du projet arrive plus tard.
