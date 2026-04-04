"""
Découpage d’image en patchs et reconstruction BCS (mesures simulées + solveur par patch).

Le découpage est sans recouvrement ; la reco attend ratio/M/Phi et utilise le dictionnaire
demandé (DCT, mixte, K-SVD…). CoSaMP : voir les champs cosamp_s / cosamp_s_mode en sortie.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image
import cv2

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


def vectoriser(patch: np.ndarray, order: str = "C") -> np.ndarray:
    """
    Convertit un patch (bloc B×B ou vecteur) en vecteur colonne de taille B².
    Utilisé par les méthodes qui attendent un signal 1D.
    """
    return np.asarray(patch, dtype=np.float64).reshape(-1, order=order)


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
    psnr_stop: bool = False,
    psnr_target_db: float = 45.0,
    lambda_lasso: float = 0.01,
    norm_p: float = 0.5,
    s_cosamp_auto: bool = False,
    n_iter_ksvd: int = 0,
    ksvd_train_patches: int | None = None,
    dictionary_train_image_path: str | None = None,
) -> Any:
    """
    Découpe une image en patchs (vecteurs colonnes).

    Si on fournit aussi une information de mesure (par ex. `ratio` ou `M` ou `Phi`) et un dictionnaire `D`,
    la fonction peut aussi reconstruire l'image (decoder BCS) et renvoyer l’image reconstruite.

    `dictionary_train_image_path` (sujet §7) : pour mixte / K-SVD, apprend ou initialise `D` à partir des patchs
    de cette image ; la reconstruction utilise toujours les patchs de `image_path`.
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

    # Patchs pour apprendre / initialiser D (sujet §7 : image hors entraînement)
    matrice_pour_dictionnaire = matrice_patchs
    if dictionary_train_image_path:
        Xd = load_grayscale_matrix(dictionary_train_image_path)
        matrice_pour_dictionnaire, _ = image_to_patch_vectors(
            Xd, B=B, nrows=nrows, ncols=ncols, order=order
        )
        if matrice_pour_dictionnaire.shape[0] != matrice_patchs.shape[0]:
            raise ValueError(
                "dictionary_train_image_path : les patchs doivent avoir la même dimension N que l'image à reconstruire "
                "(même B et même grille nrows×ncols)."
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
    from backend.utils.mesure import (
        apply_measurement,
        compute_coherence_cours_phi_d,
        generate_measurement_matrix,
    )
    from backend.utils.Methode import (
        basis_pursuit,
        cosamp,
        irls,
        lasso_ista,
        lp,
        mp,
        omp,
        stomp,
    )
    from backend.utils.Dictionnaire import (
        build_dct_dictionary,
        estime_ordre_parcimonie_cosamp,
        learn_ksvd_full,
    )

    N, NB = matrice_patchs.shape
    NB_d = int(matrice_pour_dictionnaire.shape[1])
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
        Phi = generate_measurement_matrix(0.0, N, mode_phi, seed=seed, M=M)

    # D si absent : DCT fixe, mélange seul, ou K-SVD (init dct / mixte / random selon le type)
    ksvd_meta: dict[str, Any] | None = None
    if D is None:
        dt = dictionary_type.lower().strip()
        K = int(n_atoms) if n_atoms is not None else N
        K = min(K, N)
        L_train = NB_d if ksvd_train_patches is None else min(int(ksvd_train_patches), NB_d)

        def _run_ksvd(init_m: str, nit: int) -> np.ndarray:
            D_loc, _ = learn_ksvd_full(
                matrice_pour_dictionnaire,
                K,
                nit,
                init=init_m,
                omp_max_iter=max_iter,
                omp_epsilon=epsilon,
                seed=seed,
                max_train_cols=L_train,
            )
            return D_loc

        if dt == "dct":
            if n_iter_ksvd > 0:
                D = _run_ksvd("dct", n_iter_ksvd)
                ksvd_meta = {
                    "mode": "ksvd",
                    "init": "dct",
                    "n_iter": n_iter_ksvd,
                    "train_cols": L_train,
                }
            else:
                D_full = build_dct_dictionary(N)
                D = D_full[:, :K].astype(np.float64, copy=False)
        elif dt in ("mixte", "mix_dct_patches", "dct_ksvd_init"):
            if n_iter_ksvd > 0:
                D = _run_ksvd("mixte", n_iter_ksvd)
                ksvd_meta = {
                    "mode": "ksvd",
                    "init": "mixte",
                    "n_iter": n_iter_ksvd,
                    "train_cols": L_train,
                }
            else:
                from backend.utils.Dictionnaire import init_dictionnaire_mixte_dct_patches

                D = init_dictionnaire_mixte_dct_patches(matrice_pour_dictionnaire, K)
        elif dt in ("ksvd", "ksvd_random"):
            nit = n_iter_ksvd if n_iter_ksvd > 0 else 10
            D = _run_ksvd("random", nit)
            ksvd_meta = {"mode": "ksvd", "init": "random", "n_iter": nit, "train_cols": L_train}
        elif dt in ("ksvd_dct", "ksvd_from_dct"):
            nit = n_iter_ksvd if n_iter_ksvd > 0 else 10
            D = _run_ksvd("dct", nit)
            ksvd_meta = {"mode": "ksvd", "init": "dct", "n_iter": nit, "train_cols": L_train}
        elif dt in ("ksvd_mixte",):
            nit = n_iter_ksvd if n_iter_ksvd > 0 else 10
            D = _run_ksvd("mixte", nit)
            ksvd_meta = {"mode": "ksvd", "init": "mixte", "n_iter": nit, "train_cols": L_train}
        else:
            raise ValueError(
                "dictionary_type inconnu : 'dct', 'mixte', 'ksvd', 'ksvd_dct', 'ksvd_mixte', "
                "'ksvd_random' (voir README). Avec 'dct' ou 'mixte', n_iter_ksvd>0 lance l'apprentissage."
            )

    # Mesures de tous les patchs en colonnes : y = Phi @ x
    y = apply_measurement(Phi, matrice_patchs)  # (M, NB)

    # Dictionnaire effectif pour la résolution : A = Phi @ D
    A = Phi @ D  # (M, K)
    _, K = A.shape

    # Recollage : une case = un patch reconstruit (les autres restent à 0 si max_patches limite le nombre).
    X_rec = np.zeros((n1, n2), dtype=np.float64)
    NB_used = NB if max_patches is None else min(NB, int(max_patches))

    meth = method.lower().strip()

    # CoSaMP a besoin d’un entier s (taille du support après rejet). Deux modes :
    # — tu fixes s_cosamp (ou method_params["cosamp"]["s"] via main) ;
    # — ou s_cosamp_auto=True : on lance OMP sur des patchs d’entraînement et on prend la médiane
    #   des nombres de coefficients actifs (même idée que dans le TD après K-SVD).
    s_eff_cosamp = int(s_cosamp)
    cosamp_s_estime = False
    if meth == "cosamp" and s_cosamp_auto:
        cosamp_s_estime = True
        s_eff_cosamp = estime_ordre_parcimonie_cosamp(
            matrice_pour_dictionnaire,
            D,
            max_iter_omp=max_iter,
            epsilon=epsilon,
            max_echantillons=min(64, NB_d),
            seed=seed,
        )

    method_norm = meth
    if method_norm == "mp":
        solver = mp
    elif method_norm == "omp":
        solver = omp
    elif method_norm == "stomp":
        solver = stomp
    elif method_norm == "cosamp":
        solver = cosamp
    elif method_norm in ("irls", "irls_lp", "irls_p"):
        solver = irls
    elif method_norm in ("bp", "basis_pursuit"):
        solver = basis_pursuit
    elif method_norm == "lp":
        solver = lp
    elif method_norm in ("lasso", "lasso_ista"):
        solver = lasso_ista
    else:
        raise ValueError(
            "method doit être parmi : mp, omp, stomp, cosamp, irls, irls_lp, bp, lp, lasso."
        )

    for idx in range(NB_used):
        yj = y[:, idx]
        # psnr_stop : à chaque itération du solveur, si le patch reconstruit dépasse psnr_target_db, on arrête.
        # Utile surtout en expérimentation (il faut le patch vrai, donc cas « simulation » où on connaît x).
        x_ref = matrice_patchs[:, idx] if psnr_stop else None
        extra_psnr: dict = {}
        if psnr_stop and x_ref is not None:
            extra_psnr = {
                "reference_for_psnr": x_ref,
                "D_recon": D,
                "psnr_target_db": psnr_target_db,
            }

        if method_norm == "stomp":
            alpha = solver(
                A, yj, max_iter=max_iter, eps=epsilon, t=t_stomp, **extra_psnr
            )
        elif method_norm == "cosamp":
            alpha = solver(
                A, yj, max_iter=max_iter, epsilon=epsilon, s=s_eff_cosamp, **extra_psnr
            )
        elif method_norm in ("irls", "irls_lp", "irls_p"):
            alpha = solver(
                A,
                yj,
                p=float(norm_p),
                max_iter=max_iter,
                epsilon=epsilon,
                **extra_psnr,
            )
        elif method_norm in ("bp", "basis_pursuit", "lp"):
            alpha = solver(A, yj, **extra_psnr)
        elif method_norm in ("lasso", "lasso_ista"):
            alpha = solver(
                A,
                yj,
                lambda_reg=lambda_lasso,
                max_iter=max_iter,
                tol=epsilon,
                **extra_psnr,
            )
        else:
            alpha = solver(A, yj, max_iter=max_iter, epsilon=epsilon, **extra_psnr)

        x_hat = D @ alpha  # (N,)
        i_bloc = idx // nc
        j_bloc = idx % nc
        i0, j0 = i_bloc * B_eff, j_bloc * B_eff
        X_rec[i0 : i0 + B_eff, j0 : j0 + B_eff] = x_hat.reshape(B_eff, B_eff, order=order)

    out["image_reconstruite"] = X_rec
    if NB_used < NB:
        out["reconstruction_partielle"] = True
        out["nb_patchs_reconstruits"] = int(NB_used)
        out["nb_patchs_total"] = int(NB)
    out["phi"] = Phi
    out["D"] = D
    M_phi, N_phi = Phi.shape
    out["nb_mesures_M"] = int(M_phi)
    out["dim_patch_N"] = int(N_phi)
    out["pourcentage_mesures"] = float(100.0 * M_phi / N_phi)
    out["coherence_mutuelle_cours"] = float(compute_coherence_cours_phi_d(Phi, D))
    if dictionary_train_image_path:
        out["dictionary_train_image_path"] = dictionary_train_image_path
    if ksvd_meta is not None:
        out["ksvd_meta"] = ksvd_meta
    if ratio is not None or M is not None or Phi is not None:
        if meth == "cosamp":
            out["s_cosamp_utilise"] = int(s_eff_cosamp)
            out["cosamp_s"] = int(s_eff_cosamp)
            out["cosamp_s_mode"] = "estime_omp" if cosamp_s_estime else "fixe"
        else:
            out["s_cosamp_utilise"] = None
    return out if as_dict else out["image_reconstruite"]

def apply_bilateral_filter(reconstructed_image: np.ndarray, d: int = 5, sigma_color: float = 75.0, sigma_space: float = 75.0) -> np.ndarray:
    """
    On applique un filtre bilatéral pour atténuer l'effet de bloc
    
    Paramètres :
    - d : Diamètre du voisinage (plus il est grand, plus le lissage est large).
    - sigma_color : Plus il est grand, plus des couleurs éloignées seront mélangées.
    - sigma_space : Plus il est grand, plus les pixels éloignés influenceront le calcul.
    """
    img_float32 = np.clip(reconstructed_image, 0, 255).astype(np.float32)
    
    # Application du filtre
    smoothed_img = cv2.bilateralFilter(img_float32, d, sigma_color, sigma_space)

    return smoothed_img.astype(np.float64)