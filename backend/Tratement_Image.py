"""
Traitement des images.

Pour l’instant, on s’occupe du découpage d’une image en blocs carrés,
puis on met chaque bloc sous forme de vecteur (une colonne de matrice).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def load_grayscale_matrix(path: str, dtype: type = np.float64) -> np.ndarray:
    """Charge l'image et renvoie une matrice 2D de niveaux de gris."""
    im = Image.open(path)
    arr = np.asarray(im)

    if arr.ndim == 2:
        X = arr.astype(dtype, copy=False)
    else:
        # Couleur (RGB/RGBA) : conversion en luminance
        if arr.shape[2] >= 3:
            r = arr[..., 0].astype(np.float64)
            g = arr[..., 1].astype(np.float64)
            b = arr[..., 2].astype(np.float64)
            X = (0.299 * r + 0.587 * g + 0.114 * b).astype(dtype)
        else:
            # Cas peu probable (canal unique)
            X = arr[..., 0].astype(dtype, copy=False)

    # Stockage contigu : utile pour les calculs numpy
    return np.ascontiguousarray(X)


def _crop_to_multiple_of_b(X: np.ndarray, B: int) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Recadre pour que la hauteur et la largeur soient des multiples de B.

    On garde le coin haut-gauche pour obtenir un pavage strict de blocs BxB.
    """
    n1, n2 = X.shape
    n1c, n2c = (n1 // B) * B, (n2 // B) * B
    if n1c == 0 or n2c == 0:
        raise ValueError(f"B={B} trop grand pour la taille {X.shape}.")
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
    On fixe le decoupage soit par B, soit par (nrows, ncols).

    Dans le cas (nrows, ncols), B est deduit pour que les blocs soient carres.
    """
    if B is not None and (nrows is not None or ncols is not None):
        raise ValueError("Choisir soit B, soit (nrows, ncols), pas les deux.")

    if B is not None:
        if B < 1:
            raise ValueError("B doit etre >= 1.")
        return B, n1 // B, n2 // B

    if nrows is None or ncols is None:
        raise ValueError("Indiquer B, ou bien (nrows, ncols).")
    if nrows < 1 or ncols < 1:
        raise ValueError("nrows et ncols doivent etre >= 1.")

    # Cote du carre : on ne depasse pas ce que permet chaque dimension
    B_grid = min(n1 // nrows, n2 // ncols)
    if B_grid < 1:
        raise ValueError("Grille impossible : image trop petite.")
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
    Decoupe l'image en blocs carres sans chevauchement puis aplati chaque bloc.

    Retour :
    - matrice_patchs : forme (B^2, NB) ; une colonne = un patch aplati
    - meta : (N1, N2, nrows, ncols) apres recadrage
    """
    X_arr = np.asarray(X)
    n1_o, n2_o = X_arr.shape

    B_eff, nr, nc = _resolve_b_and_grid(
        n1_o, n2_o, B=B, nrows=nrows, ncols=ncols
    )

    if nrows is not None and ncols is not None:
        # Grille imposee : on recadre exactement sur la grille
        n1c, n2c = nr * B_eff, nc * B_eff
        X_work = X_arr[:n1c, :n2c].copy()
        n1, n2 = n1c, n2c
    else:
        # B impose : on recadre a des multiples de B
        X_work, (n1, n2) = _crop_to_multiple_of_b(X_arr, B_eff)
        nr, nc = n1 // B_eff, n2 // B_eff

    NB = nr * nc
    colonnes = []

    # Parcours des blocs : ligne de blocs puis ligne suivante
    for i_bloc in range(nr):
        for j_bloc in range(nc):
            i0, j0 = i_bloc * B_eff, j_bloc * B_eff
            morceau = X_work[i0 : i0 + B_eff, j0 : j0 + B_eff]
            # Un bloc BxB -> vecteur taille B^2
            colonnes.append(morceau.reshape(B_eff * B_eff, order=order))

    matrice_patchs = np.column_stack(colonnes)
    return matrice_patchs, (n1, n2, nr, nc)


def patch(
    image_path: str,
    *,
    B: int | None = 8,
    nrows: int | None = None,
    ncols: int | None = None,
    order: str = "C",
    as_dict: bool = True,
    # Parametres pour decoder/reconstruction (si la valeur est fournie)
    ratio: float | None = None,  # ex: 20 (pourcentage) ou 0.2 (fraction)
    M: int | None = None,  # nombre de mesures par patch
    Phi: np.ndarray | None = None,  # Phi_B explicite (M x B^2)
    D: np.ndarray | None = None,  # dictionnaire (B^2 x K)
    method: str = "omp",
    dictionary_type: str = "dct",
    n_atoms: int | None = None,
    mode_phi: str = "gaussian",
    seed: int | None = 0,
    max_iter: int = 20,
    epsilon: float = 1e-6,
    t_stomp: float = 2.5,
    s_cosamp: int = 6,
    max_patches: int | None = None,
) -> Any:
    """
    Découpe une image en patchs (vecteurs colonnes).

    Si on fournit aussi une information de mesure (par ex. `ratio` ou `M` ou `Phi`) et un dictionnaire `D`,
    la fonction peut aussi reconstruire l'image (decoder BCS) et renvoyer l’image reconstruite.
    """
    # 1) Découpage : on lit l'image sous forme de matrice 2D puis on découpe en blocs.
    X = load_grayscale_matrix(image_path)
    matrice_patchs, meta = image_to_patch_vectors(
        X,
        B=B,
        nrows=nrows,
        ncols=ncols,
        order=order,
    )

    # 2) Sortie “par défaut” : uniquement les patchs.
    out: dict[str, Any] = {
        "matrice_patchs": matrice_patchs,  # forme (B^2, NB)
        "meta": meta,  # (N1, N2, nrows, ncols)
    }

    # 3) Reconstruction : déclenchée uniquement si on fournit ratio/M/Phi.
    # (Si seulement D est fourni, on ne peut pas reconstruire sans Phi.)
    if ratio is None and M is None and Phi is None:
        return out if as_dict else out["matrice_patchs"]

    # --- Reconstruction (decoder BCS) ---
    # Import local pour éviter les imports circulaires.
    from backend.utils.mesure import generate_measurement_matrix, apply_measurement
    from backend.utils.Methode import mp, omp, stomp, cosamp
    from backend.utils.Dictionnaire import build_dct_dictionary

    N, NB = matrice_patchs.shape
    n1, n2, nr, nc = meta
    B_eff = n1 // nr

    # Phi si absent : calcul à partir de ratio ou M
    if Phi is None:
        if M is None:
            if ratio is None:
                raise ValueError("Pour reconstruire, il faut fournir `Phi` ou bien `ratio` ou `M`.")
            # ratio peut être en “pourcentage” (15..75) ou en fraction (0.15..0.75)
            r = float(ratio)
            if r > 1.0:
                M = int(np.ceil((r / 100.0) * N))
            else:
                M = int(np.ceil(r * N))
        Phi = generate_measurement_matrix(M, N, mode_phi, seed=seed)

    # D si absent : dictionnaire DCT tronqué
    if D is None:
        if dictionary_type.lower() != "dct":
            raise ValueError("Pour l’instant, reconstruction : dictionnaire_type doit être 'dct' (par défaut).")
        D_full = build_dct_dictionary(N)
        K = int(n_atoms) if n_atoms is not None else N
        K = min(K, N)
        D = D_full[:, :K].astype(np.float64, copy=False)

    # Mesures de tous les patchs en colonnes : y = Phi @ x
    y = apply_measurement(Phi, matrice_patchs)  # (M, NB)

    # Dictionnaire effectif pour la résolution : A = Phi @ D
    A = Phi @ D  # (M, K)
    _, K = A.shape

    # Préparation du recollage
    X_rec = np.zeros((n1, n2), dtype=np.float64)
    NB_used = NB if max_patches is None else min(NB, int(max_patches))

    # Choix de la méthode
    method_norm = method.lower().strip()
    if method_norm == "mp":
        solver = mp
    elif method_norm == "omp":
        solver = omp
    elif method_norm == "stomp":
        solver = stomp
    elif method_norm == "cosamp":
        solver = cosamp
    else:
        raise ValueError("method doit être parmi : mp, omp, stomp, cosamp.")

    for idx in range(NB_used):
        yj = y[:, idx]

        if method_norm == "stomp":
            alpha = solver(A, yj, max_iter=max_iter, eps=epsilon, t=t_stomp)
        elif method_norm == "cosamp":
            alpha = solver(A, yj, max_iter=max_iter, epsilon=epsilon, s=s_cosamp)
        else:
            alpha = solver(A, yj, max_iter=max_iter, epsilon=epsilon)

        x_hat = D @ alpha  # (N,)
        i_bloc = idx // nc
        j_bloc = idx % nc
        i0, j0 = i_bloc * B_eff, j_bloc * B_eff
        X_rec[i0 : i0 + B_eff, j0 : j0 + B_eff] = x_hat.reshape(B_eff, B_eff, order=order)

    out["image_reconstruite"] = X_rec
    out["phi"] = Phi
    out["D"] = D
    return out if as_dict else out["image_reconstruite"]

