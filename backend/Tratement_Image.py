"""
Traitement des images.

Pour l’instant, on s’occupe du découpage d’une image en blocs carrés,
puis on met chaque bloc sous forme de vecteur (une colonne de matrice).
"""

from __future__ import annotations

from typing import Any

from backend.image_blocking import image_to_patch_vectors, load_grayscale_matrix


def tratament_image(
    image_path: str,
    *,
    B: int | None = 8,
    nrows: int | None = None,
    ncols: int | None = None,
    order: str = "C",
    as_dict: bool = True,
) -> Any:
    """
    Découpe une image et renvoie la matrice des patchs.

    Paramètres principaux
    - `B` : taille du côté des blocs (si fourni, `nrows/ncols` ne doivent pas être donnés)
    - `nrows/ncols` : nombre de blocs sur hauteur/largeur (et donc NB = nrows * ncols)
    - `order` : façon d'aplatir chaque bloc en vecteur
    """
    # 1) On lit l'image sous forme d'une matrice 2D (niveaux de gris)
    X = load_grayscale_matrix(image_path)

    # 2) On découpe en blocs et on empile chaque bloc en vecteur colonne
    matrice_patchs, meta = image_to_patch_vectors(
        X,
        B=B,
        nrows=nrows,
        ncols=ncols,
        order=order,
    )

    # 3) On renvoie soit juste le tableau, soit un petit paquet d’infos
    if as_dict:
        return {
            "matrice_patchs": matrice_patchs,  # forme (B^2, NB)
            "meta": meta,  # (N1, N2, nrows, ncols)
        }
    return matrice_patchs

