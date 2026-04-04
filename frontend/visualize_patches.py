# -*- coding: utf-8 -*-
"""
Visualisation du découpage d'une image en patchs B×B.

Une seule sortie utile : l’image recadrée avec la grille des blocs (même géométrie que la reconstruction).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.Tratement_Image import image_to_patch_vectors, load_grayscale_matrix

# Couleur proche du thème IHM (lisible sur gris clair / peau)
_GRID_RGB = (30, 90, 142)
_GRID_THICKNESS = 2


def _draw_grid_overlay(
    X: np.ndarray,
    B: int,
    *,
    line_color: tuple[int, int, int] = _GRID_RGB,
    thickness: int = _GRID_THICKNESS,
) -> np.ndarray:
    """Image RGB : niveaux de gris + quadrillage aux frontières des patchs."""
    Xn = X.astype(np.float64, copy=False)
    xmin, xmax = float(Xn.min()), float(Xn.max())
    if abs(xmax - xmin) < 1e-12:
        X8 = np.zeros_like(Xn, dtype=np.uint8)
    else:
        X8 = ((Xn - xmin) / (xmax - xmin) * 255.0).clip(0, 255).astype(np.uint8)

    rgb = np.stack([X8, X8, X8], axis=-1)
    nr = X.shape[0] // B
    nc = X.shape[1] // B
    t = max(1, int(thickness))

    for i in range(nr + 1):
        y = i * B
        y0 = max(0, y - t // 2)
        y1 = min(rgb.shape[0], y0 + t)
        rgb[y0:y1, :, :] = line_color

    for j in range(nc + 1):
        x = j * B
        x0 = max(0, x - t // 2)
        x1 = min(rgb.shape[1], x0 + t)
        rgb[:, x0:x1, :] = line_color

    return rgb


def visualize_patches(
    image_path: str,
    *,
    B: int,
    out_dir: str,
    order: str = "C",
) -> str:
    """
    Écrit ``{stem}_grille_sur_image_B{B}.png`` dans ``out_dir``.
    Retourne le chemin du fichier créé.
    """
    os.makedirs(out_dir, exist_ok=True)

    X = load_grayscale_matrix(image_path)
    _, meta = image_to_patch_vectors(X, B=B, order=order)
    n1, n2, nr, nc = meta
    B_eff = n1 // nr
    X_work = X[:n1, :n2]
    stem = Path(image_path).stem or "image"

    grille_rgb = _draw_grid_overlay(X_work, B_eff)
    grille_path = os.path.join(out_dir, f"{stem}_grille_sur_image_B{B_eff}.png")
    Image.fromarray(grille_rgb).save(grille_path)
    return grille_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Grille des patchs B×B sur l’image recadrée.")
    parser.add_argument("--image", default="lena.jpg", help="Chemin vers l'image.")
    parser.add_argument("-B", "--patch-size", type=int, default=8, help="Taille du bloc B×B.")
    parser.add_argument("--out", default=os.path.join("Data", "Result", "patch_vis"), help="Dossier de sortie.")
    args = parser.parse_args()

    path = visualize_patches(args.image, B=args.patch_size, out_dir=args.out)
    print("Image enregistrée :", path)


if __name__ == "__main__":
    main()
