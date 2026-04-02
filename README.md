# Compressive-Sensing

Repo du projet (ING2), pour l’instant y’a surtout le découpage d’image en patchs.

## deps

```bash
pip install numpy pillow
```

(ou un venv si tu préfères)

## script

`backend/Tratement_Image.py` : charge une image, découpe en blocs B×B, sort une matrice avec une colonne = un patch aplati.

Par défaut ça prend `lena.jpg` à la racine du dépôt.

```bash
python3 -c "from backend.Tratement_Image import tratament_image; print(traitament_image('lena.jpg', B=8)['matrice_patchs'].shape)"
python3 -c "from backend.Tratement_Image import tratament_image; print(traitament_image('lena.jpg', B=16)['matrice_patchs'].shape)"
python3 -c "from backend.Tratement_Image import tratament_image; print(traitament_image('lena.jpg', nrows=32, ncols=32)['matrice_patchs'].shape)"
python3 -c "from backend.Tratement_Image import tratament_image; print(traitament_image('autre_fichier.png', B=8)['matrice_patchs'].shape)"
```

Le reste du projet arrive plus tard.
