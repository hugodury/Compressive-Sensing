# -*- coding: utf-8 -*-
"""
Découpage d'une image en petits blocs carrés, puis chaque bloc en un vecteur.
Utile pour la suite du projet (mesures bloc par bloc, dictionnaire, etc.).
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def load_grayscale_matrix(path: str, dtype: type = np.float64) -> np.ndarray:
    """Charge l'image : on travaille sur une matrice 2D (niveaux de gris)."""
    im = Image.open(path)
    arr = np.asarray(im)

    if arr.ndim == 2:
        X = arr.astype(dtype, copy=False)
    else:
        # Couleur ou alpha : on passe en niveaux de gris (pondération classique R/V/B)
        if arr.shape[2] >= 3:
            r = arr[..., 0].astype(np.float64)
            g = arr[..., 1].astype(np.float64)
            b = arr[..., 2].astype(np.float64)
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            X = lum.astype(dtype)
        else:
            X = arr[..., 0].astype(dtype, copy=False)

    return np.ascontiguousarray(X)


def _crop_to_multiple_of_b(X: np.ndarray, B: int) -> tuple[np.ndarray, tuple[int, int]]:
    """
    On enlève les pixels en trop pour que hauteur et largeur soient des multiples de B,
    sinon on ne peut pas paver proprement en carrés B×B (on garde le coin haut-gauche).
    """
    n1, n2 = X.shape
    # plus grande taille multiple de B sans dépasser l'image
    n1c, n2c = (n1 // B) * B, (n2 // B) * B
    if n1c == 0 or n2c == 0:
        raise ValueError(f"B={B} trop grand pour l'image de taille {X.shape}.")
    if n1c != n1 or n2c != n2:
        X = X[:n1c, :n2c].copy()
    return X, (n1c, n2c)


def _resolve_b_and_grid(
    n1: int,
    n2: int,
    *,
    B: int | None,
    nrows: int | None,
    ncols: int | None,
) -> tuple[int, int, int]:
    """
    Deux façons de régler le découpage : soit la taille du carré B, soit le nombre de blocs
    en vertical / horizontal (à ce moment-là B est calculé pour que tout tombe juste).
    """
    if B is not None and (nrows is not None or ncols is not None):
        raise ValueError("Choisir soit le paramètre B, soit le couple (nrows, ncols), pas les deux.")
    if B is not None:
        if B < 1:
            raise ValueError("B doit être >= 1.")
        return B, n1 // B, n2 // B
    if nrows is None or ncols is None:
        raise ValueError("Indiquer B, ou bien nrows et ncols (patchs en hauteur et en largeur).")
    if nrows < 1 or ncols < 1:
        raise ValueError("nrows et ncols doivent être >= 1.")
    # B identique partout : le côté du carré ne peut pas dépasser ce que permet chaque dimension
    B_grid = min(n1 // nrows, n2 // ncols)
    if B_grid < 1:
        raise ValueError(
            f"Grille {nrows}×{ncols} impossible : l'image {n1}×{n2} est trop petite "
            "pour au moins un patch carré par cellule."
        )
    return B_grid, nrows, ncols


def image_to_patch_vectors(
    X: np.ndarray,
    B: int | None = None,
    *,
    nrows: int | None = None,
    ncols: int | None = None,
    order: str = "C",
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Découpe l'image en blocs B×B sans chevauchement, puis empile chaque bloc en colonne.

    Retourne une matrice (B², NB) : une colonne = un bloc aplati, prêt pour les multiplications
    matricielles derrière (mesure, dictionnaire…).

    order='C' : on lit le bloc ligne par ligne (habituel sous Python).
    order='F' : on lit colonne par colonne (plus proche du vec() qu'on voit en maths).
    """
    X_arr = np.asarray(X)
    n1_o, n2_o = X_arr.shape
    B_eff, nr, nc = _resolve_b_and_grid(n1_o, n2_o, B=B, nrows=nrows, ncols=ncols)

    if nrows is not None and ncols is not None:
        # grille imposée : surface exacte nrows*B par ncols*B
        n1c, n2c = nr * B_eff, nc * B_eff
        X_work = X_arr[:n1c, :n2c].copy()
        n1, n2 = n1c, n2c
    else:
        X_work, (n1, n2) = _crop_to_multiple_of_b(X_arr, B_eff)
        nr, nc = n1 // B_eff, n2 // B_eff

    NB = nr * nc
    colonnes = []
    # parcours des blocs : d'abord une ligne de blocs de gauche à droite, puis la ligne suivante
    for i_bloc in range(nr):
        for j_bloc in range(nc):
            i0, j0 = i_bloc * B_eff, j_bloc * B_eff
            morceau = X_work[i0 : i0 + B_eff, j0 : j0 + B_eff]
            # un bloc B×B -> un vecteur de taille B² (une colonne de la grande matrice)
            vecteur = morceau.reshape(B_eff * B_eff, order=order)
            colonnes.append(vecteur)

    matrice_patchs = np.column_stack(colonnes)
    return matrice_patchs, (n1, n2, nr, nc)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Découpe une image en blocs carrés et affiche les infos (taille des vecteurs, nombre de blocs)."
    )
    parser.add_argument(
        "--image",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "Projet", "lena.jpg"),
        help="Fichier image",
    )
    groupe = parser.add_mutually_exclusive_group(required=False)
    groupe.add_argument(
        "-B",
        "--patch-size",
        type=int,
        default=None,
        dest="B",
        help="Côté du bloc en pixels (le nombre total de blocs suit des dimensions de l'image).",
    )
    groupe.add_argument(
        "--grille",
        "--grid",
        nargs=2,
        type=int,
        metavar=("LIGNES", "COLONNES"),
        dest="grille",
        help="Nombre de blocs en hauteur puis en largeur (produit = nombre total de patchs).",
    )
    args = parser.parse_args()
    if args.B is None and args.grille is None:
        args.B = 8

    X = load_grayscale_matrix(args.image)
    if args.grille is not None:
        nl, nc = args.grille
        X_mat, (n1, n2, nrows, ncols) = image_to_patch_vectors(X, nrows=nl, ncols=nc)
        B_eff = n1 // nrows
    else:
        X_mat, (n1, n2, nrows, ncols) = image_to_patch_vectors(X, args.B)
        B_eff = args.B
    NB = nrows * ncols
    print(f"Image recadrée utilisée : {n1} × {n2} (originale {X.shape[0]} × {X.shape[1]})")
    print(f"Taille de patch B = {B_eff}, nombre de blocs NB = {NB} ({nrows} × {ncols})")
    print(f"Matrice des patchs (une colonne = un bloc aplati) : {X_mat.shape}  (= B² × NB)")
