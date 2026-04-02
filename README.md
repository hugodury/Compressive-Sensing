# Compressive-Sensing

Repo du projet (ING2), pour l’instant y’a surtout le découpage d’image en patchs.

## deps

```bash
pip install numpy pillow
```

(ou un venv si tu préfères)

## script

`backend/image_blocking.py` : charge une image, découpe en blocs B×B, sort une matrice avec une colonne = un patch aplati.

Par défaut ça prend `lena.jpg` à la racine du dépôt.

```bash
python3 backend/image_blocking.py
python3 backend/image_blocking.py -B 16
python3 backend/image_blocking.py --grille 32 32
python3 backend/image_blocking.py --image autre_fichier.png
```

Le reste du projet arrive plus tard.
