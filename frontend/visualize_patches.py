# -*- coding: utf-8 -*-
"""
Petite visualisation du découpage d'une image en patchs B×B.

Objectif : produire des images pour voir :
1) la grille des patchs sur l'image,
2) la recomposition patch par patch (sur les premières étapes),
3) l'image finale recomposée (zone recadrée).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Permet d'importer le package backend quand on lance:
# `python3 frontend/visualize_patches.py`
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Le script est côté frontend, mais le traitement image reste dans le backend.
from backend.Tratement_Image import image_to_patch_vectors, load_grayscale_matrix


def _save_grayscale(path: str, X: np.ndarray) -> None:
    X = np.asarray(X)
    # Normalisation simple en 0..255 pour l'image de sortie (sinon PIL sature mal).
    xmin = float(X.min())
    xmax = float(X.max())
    if abs(xmax - xmin) < 1e-12:
        arr = np.zeros_like(X, dtype=np.uint8)
    else:
        arr = ((X - xmin) / (xmax - xmin) * 255.0).clip(0, 255).astype(np.uint8)

    im = Image.fromarray(arr, mode="L")
    im.save(path)


def _draw_grid_overlay(X: np.ndarray, B: int, *, line_color=(255, 0, 0), thickness: int = 1) -> np.ndarray:
    """
    Retourne une image RGB avec des lignes verticales/horizontales sur les bords de patchs.
    """
    # base en RGB
    Xn = X.astype(np.float64, copy=False)
    xmin, xmax = float(Xn.min()), float(Xn.max())
    if abs(xmax - xmin) < 1e-12:
        X8 = np.zeros_like(Xn, dtype=np.uint8)
    else:
        X8 = ((Xn - xmin) / (xmax - xmin) * 255.0).clip(0, 255).astype(np.uint8)

    rgb = np.stack([X8, X8, X8], axis=-1)
    nr = X.shape[0] // B
    nc = X.shape[1] // B

    # Lignes horizontales (y = i*B)
    for i in range(nr + 1):
        y = i * B
        y0 = max(0, y - thickness // 2)
        y1 = min(rgb.shape[0], y0 + thickness)
        rgb[y0:y1, :, :] = line_color

    # Lignes verticales (x = j*B)
    for j in range(nc + 1):
        x = j * B
        x0 = max(0, x - thickness // 2)
        x1 = min(rgb.shape[1], x0 + thickness)
        rgb[:, x0:x1, :] = line_color

    return rgb


def _render_patches_separes(
    X_work: np.ndarray,
    *,
    B: int,
    nr: int,
    nc: int,
    gap: int,
) -> np.ndarray:
    """
    Rend chaque patch comme un petit carreau, avec un espace blanc entre les blocs.
    C'est plus proche du style “découpage en patchs” visible dans certaines figures.
    """
    if gap < 0:
        raise ValueError("gap doit etre >= 0")
    if gap == 0:
        # Cas particulier : juste une mise en niveaux de gris (meme image)
        return np.stack([_save_to_uint8(X_work)] * 3, axis=-1)

    Xn = X_work.astype(np.float64, copy=False)
    xmin, xmax = float(Xn.min()), float(Xn.max())
    if abs(xmax - xmin) < 1e-12:
        X8 = np.zeros_like(Xn, dtype=np.uint8)
    else:
        X8 = ((Xn - xmin) / (xmax - xmin) * 255.0).clip(0, 255).astype(np.uint8)

    out_h = nr * B + (nr + 1) * gap
    out_w = nc * B + (nc + 1) * gap
    out = np.full((out_h, out_w), 255, dtype=np.uint8)  # fond blanc

    for i_bloc in range(nr):
        for j_bloc in range(nc):
            i0, j0 = i_bloc * B, j_bloc * B
            y = gap + i_bloc * (B + gap)
            x = gap + j_bloc * (B + gap)
            out[y : y + B, x : x + B] = X8[i0 : i0 + B, j0 : j0 + B]

    return np.stack([out, out, out], axis=-1)


def _save_to_uint8(X: np.ndarray) -> np.ndarray:
    """Convertit une matrice 2D en niveaux de gris 0..255 (uint8)."""
    Xn = np.asarray(X, dtype=np.float64)
    xmin, xmax = float(Xn.min()), float(Xn.max())
    if abs(xmax - xmin) < 1e-12:
        return np.zeros_like(Xn, dtype=np.uint8)
    return ((Xn - xmin) / (xmax - xmin) * 255.0).clip(0, 255).astype(np.uint8)


def visualize_patches(
    image_path: str,
    *,
    B: int,
    max_steps: int,
    out_dir: str,
    order: str = "C",
    gap: int = 2,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # 1) Chargement + recadrage implicite (meta vient du même découpage que le reste du pipeline)
    X = load_grayscale_matrix(image_path)
    # On appelle la fonction de découpage pour récupérer les dimensions recadrées et nr/nc
    _, meta = image_to_patch_vectors(X, B=B, order=order)
    n1, n2, nr, nc = meta
    B_eff = n1 // nr

    X_work = X[:n1, :n2]

    # 2) Image avec grille des patchs
    # Fond blanc + espaces entre les blocs
    grid_rgb = _render_patches_separes(X_work, B=B_eff, nr=nr, nc=nc, gap=gap)
    grid_path = os.path.join(out_dir, f"lena_grid_B{B_eff}_gap{gap}.png")
    Image.fromarray(grid_rgb).save(grid_path)

    # 3) Recompositions patch par patch (on garde seulement les premières étapes)
    current = np.zeros_like(X_work)
    total_patches = nr * nc
    save_every = max(1, total_patches // max_steps) if max_steps > 0 else total_patches

    saved = 0
    idx_patch = 0
    for i_bloc in range(nr):
        for j_bloc in range(nc):
            i0, j0 = i_bloc * B_eff, j_bloc * B_eff
            current[i0 : i0 + B_eff, j0 : j0 + B_eff] = X_work[i0 : i0 + B_eff, j0 : j0 + B_eff]

            if saved < max_steps and (idx_patch % save_every == 0 or idx_patch == total_patches - 1):
                # Image “patchs séparés” avec fond blanc + espaces
                rgb_step = _render_patches_separes(
                    current,
                    B=B_eff,
                    nr=nr,
                    nc=nc,
                    gap=gap,
                )
                step_path = os.path.join(out_dir, f"lena_step_{idx_patch:05d}_B{B_eff}.png")
                Image.fromarray(rgb_step).save(step_path)
                saved += 1

            idx_patch += 1

    # 4) Image finale recomposée (sur la zone recadrée)
    final_path = os.path.join(out_dir, f"lena_recompose_B{B_eff}.png")
    _save_grayscale(final_path, current)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualiser le découpage de Lena en patchs.")
    parser.add_argument("--image", default="lena.jpg", help="Chemin vers l'image (par défaut : lena.jpg à la racine).")
    parser.add_argument("-B", "--patch-size", type=int, default=8, help="Taille du bloc B×B.")
    parser.add_argument("--max-steps", type=int, default=16, help="Nombre d'images 'étape' à sauvegarder (pour ne pas spammer).")
    parser.add_argument("--gap", type=int, default=2, help="Espace blanc entre deux patchs pour l'image 'grille'.")
    parser.add_argument("--out", default=os.path.join("Data", "Result", "patch_vis"), help="Dossier de sortie.")
    args = parser.parse_args()

    # image_to_patch_vectors travaille avec la même convention que le reste du pipeline
    visualize_patches(
        args.image,
        B=args.patch_size,
        max_steps=args.max_steps,
        out_dir=args.out,
        gap=args.gap,
    )

    print("Images sauvegardées dans :", args.out)


if __name__ == "__main__":
    main()

