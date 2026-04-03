"""
Méthodes parcimonieuses et relaxations convexes pour le problème y ≈ A α
(avec A = Φ D en compressive sensing sur les patchs).

On regroupe : MP, OMP, StOMP, CoSaMP, IRLS (régression pondérée itérée),
Basis Pursuit / LP (même formulation linéaire), LASSO via ISTA.
Les critères d’arrêt classiques (résidu, max itérations) peuvent être complétés
par un PSNR cible si on fournit le patch original (cas expérimental / debug).
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.optimize import linprog

from backend.utils.Metrics import compute_psnr


def _soft_threshold(v: np.ndarray, tau: float) -> np.ndarray:
    """Seuillage doux (proximal de la norme L1)."""
    return np.sign(v) * np.maximum(np.abs(v) - tau, 0.0)


def _psnr_stop(
    reference_for_psnr: np.ndarray | None,
    D_recon: np.ndarray | None,
    alpha: np.ndarray,
    psnr_target_db: float | None,
) -> bool:
    """True si le PSNR dépasse le seuil (arrêt anticipé)."""
    if reference_for_psnr is None or D_recon is None or psnr_target_db is None:
        return False
    x_hat = D_recon @ alpha
    p = compute_psnr(reference_for_psnr, x_hat)
    return not math.isinf(p) and p >= float(psnr_target_db)


def mp(
    D: np.ndarray,
    x: np.ndarray,
    *,
    max_iter: int = 1000,
    epsilon: float = 1e-6,
    reference_for_psnr: np.ndarray | None = None,
    D_recon: np.ndarray | None = None,
    psnr_target_db: float | None = None,
    **kwargs: object,
) -> np.ndarray:
    """
    Matching Pursuit (MP).

    D : matrice (souvent A = ΦD), x : mesures y.
    Si reference_for_psnr, D_recon et psnr_target_db sont fournis, on peut
    arrêter dès que le PSNR (entre patch vrai et D_recon @ α) dépasse le seuil.
    """
    _ = kwargs
    D = np.asarray(D, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    _, K = D.shape

    alpha = np.zeros(K, dtype=np.float64)
    residuel = x.copy()

    denom = float(np.linalg.norm(D))
    k = 0
    while k < max_iter and np.linalg.norm(residuel) > epsilon:
        corr = D.T @ residuel
        if denom > 0:
            corr = np.abs(corr) / denom
        mk = int(np.argmax(corr))

        dj = D[:, mk]
        dj_norm2 = float(np.linalg.norm(dj) ** 2)
        if dj_norm2 < 1e-15:
            break

        zmk = float((dj.T @ residuel) / dj_norm2)
        alpha[mk] += zmk
        residuel = residuel - zmk * dj

        if _psnr_stop(reference_for_psnr, D_recon, alpha, psnr_target_db):
            break
        k += 1

    return alpha


def omp(
    D: np.ndarray,
    x: np.ndarray,
    *,
    max_iter: int = 1000,
    epsilon: float = 1e-6,
    reference_for_psnr: np.ndarray | None = None,
    D_recon: np.ndarray | None = None,
    psnr_target_db: float | None = None,
    **kwargs: object,
) -> np.ndarray:
    """Orthogonal Matching Pursuit (OMP)."""
    _ = kwargs
    D = np.asarray(D, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    _, K = D.shape

    alpha = np.zeros(K, dtype=np.float64)
    residuel = x.copy()

    P: list[int] = []
    denom = float(np.linalg.norm(D))

    k = 0
    alpha_k = np.zeros(0, dtype=np.float64)
    while k < max_iter and np.linalg.norm(residuel) > epsilon:
        corr = D.T @ residuel
        if denom > 0:
            corr = np.abs(corr) / denom
        mk = int(np.argmax(corr))

        P.append(mk)
        Dk = D[:, P]

        alpha_k, *_ = np.linalg.lstsq(Dk, x, rcond=None)
        residuel = x - Dk @ alpha_k

        alpha_full = np.zeros(K, dtype=np.float64)
        alpha_full[P] = alpha_k
        if _psnr_stop(reference_for_psnr, D_recon, alpha_full, psnr_target_db):
            break
        k += 1

    alpha = np.zeros(K, dtype=np.float64)
    if P:
        alpha[P] = alpha_k
    return alpha


def stomp(
    D: np.ndarray,
    x: np.ndarray,
    *,
    max_iter: int = 1000,
    eps: float = 1e-6,
    t: float = 2.5,
    reference_for_psnr: np.ndarray | None = None,
    D_recon: np.ndarray | None = None,
    psnr_target_db: float | None = None,
    **kwargs: object,
) -> np.ndarray:
    """StOMP (sélection par seuil)."""
    _ = kwargs
    D = np.asarray(D, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    _, K = D.shape

    alpha = np.zeros(K, dtype=np.float64)
    P: list[int] = []
    residuel = x.copy()

    k = 0
    while k < max_iter and np.linalg.norm(residuel) > eps:
        C = np.zeros(K, dtype=np.float64)
        for j in range(K):
            dj = D[:, j]
            norme_dj = np.linalg.norm(dj)
            if norme_dj > 1e-15:
                C[j] = np.abs(dj.T @ residuel) / norme_dj

        seuil = float(t * np.linalg.norm(residuel) / np.sqrt(K))

        Lambda = [j for j in range(K) if C[j] > seuil]
        if not Lambda:
            break

        for j in Lambda:
            if j not in P:
                P.append(j)
        P.sort()

        DS = D[:, P]
        alphak, *_ = np.linalg.lstsq(DS, x, rcond=None)
        residuel = x - DS @ alphak

        alpha = np.zeros(K, dtype=np.float64)
        alpha[P] = alphak

        if _psnr_stop(reference_for_psnr, D_recon, alpha, psnr_target_db):
            break
        k += 1

    return alpha


def cosamp(
    D: np.ndarray,
    x: np.ndarray,
    *,
    max_iter: int = 100,
    epsilon: float = 1e-6,
    s: int = 6,
    reference_for_psnr: np.ndarray | None = None,
    D_recon: np.ndarray | None = None,
    psnr_target_db: float | None = None,
    **kwargs: object,
) -> np.ndarray:
    """CoSaMP."""
    _ = kwargs
    D = np.asarray(D, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    _, K = D.shape
    if s < 1:
        raise ValueError("s doit être >= 1.")

    alpha = np.zeros(K, dtype=np.float64)
    residuel = x.copy()
    supp: set[int] = set()

    s_eff = min(s, K)

    k = 1
    while k <= max_iter and np.linalg.norm(residuel) > epsilon:
        denom = float(np.linalg.norm(D))
        c = D.T @ residuel
        if denom > 0:
            c = np.abs(c) / denom
        else:
            c = np.abs(c)

        top2 = min(2 * s_eff, K)
        supp1 = set(np.argsort(c)[-top2:])

        merged = list(supp | supp1)
        if not merged:
            break

        As = D[:, merged]
        z = np.linalg.pinv(As) @ x

        top_local = min(s_eff, len(merged))
        local_best = np.argsort(np.abs(z))[-top_local:]
        supp = set(merged[i] for i in local_best)

        final_idx = sorted(list(supp))
        Af = D[:, final_idx]
        alpha = np.zeros(K, dtype=np.float64)
        alpha[final_idx] = np.linalg.pinv(Af) @ x

        residuel = x - D @ alpha

        if _psnr_stop(reference_for_psnr, D_recon, alpha, psnr_target_db):
            break
        k += 1

    return alpha


def irls(
    A: np.ndarray,
    y: np.ndarray,
    *,
    max_iter: int = 100,
    epsilon: float = 1e-6,
    delta: float = 1e-4,
    reference_for_psnr: np.ndarray | None = None,
    D_recon: np.ndarray | None = None,
    psnr_target_db: float | None = None,
    **kwargs: object,
) -> np.ndarray:
    """
    IRLS pour approcher la norme L1 sous contrainte Aα = y (régression pondérée).

    À chaque pas : poids w_i = 1/(|α_i|+δ), puis α = W^{-1} A^T (A W^{-1} A^T)^{-1} y
    avec W = diag(w_i). C’est le schéma classique « reweighted least squares » pour BP.
    """
    _ = kwargs
    A = np.asarray(A, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    M, _ = A.shape

    # Initialisation : solution aux moindres carrés (norme minimale si sous-déterminé)
    if M <= A.shape[1]:
        try:
            alpha = A.T @ np.linalg.solve(A @ A.T + 1e-10 * np.eye(M), y)
        except np.linalg.LinAlgError:
            alpha = np.linalg.pinv(A) @ y
    else:
        alpha = np.linalg.pinv(A) @ y

    eye_m = np.eye(M, dtype=np.float64)
    for _ in range(max_iter):
        # Pondération type L1 : Q^{-1} = diag(|α|+δ) dans min α^T Q α s.t. Aα = y
        q_inv = np.abs(alpha) + delta
        q_inv_mat = np.diag(q_inv)
        mmat = A @ q_inv_mat @ A.T
        try:
            alpha_new = q_inv_mat @ A.T @ np.linalg.solve(mmat + 1e-12 * eye_m, y)
        except np.linalg.LinAlgError:
            alpha_new = q_inv_mat @ A.T @ np.linalg.pinv(mmat) @ y

        if np.linalg.norm(alpha_new - alpha) < epsilon * max(1.0, np.linalg.norm(alpha)):
            alpha = alpha_new
            break
        alpha = alpha_new

        if _psnr_stop(reference_for_psnr, D_recon, alpha, psnr_target_db):
            break

    return alpha


def basis_pursuit(
    A: np.ndarray,
    y: np.ndarray,
    *,
    tol_feas: float = 1e-9,
    **kwargs: object,
) -> np.ndarray:
    """
    Basis Pursuit : min ||α||_1 sous A α = y (bruit nul).

    Reformulation LP : α = u - v avec u,v ≥ 0, coût Σ(u_i+v_i).
    """
    _ = kwargs
    A = np.asarray(A, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    M, K = A.shape

    c = np.ones(2 * K, dtype=np.float64)
    A_eq = np.hstack([A, -A], dtype=np.float64)
    bounds = [(0.0, None)] * (2 * K)

    res = linprog(
        c,
        A_eq=A_eq,
        b_eq=y,
        bounds=bounds,
        method="highs",
        options={"presolve": True},
    )
    if res.success and res.x is not None:
        z = res.x
        u, v = z[:K], z[K:]
        alpha = u - v
        if np.linalg.norm(A @ alpha - y) <= tol_feas * max(1.0, np.linalg.norm(y)):
            return alpha.astype(np.float64)

    # repli si le solveur LP échoue (matrices mal conditionnées)
    return np.linalg.lstsq(A, y, rcond=None)[0]


def lp(
    A: np.ndarray,
    y: np.ndarray,
    **kwargs: object,
) -> np.ndarray:
    """
    Même problème que Basis Pursuit (programme linéaire pour la norme L1).
    Le nom « lp » rappelle la formulation en variables LP du cours.
    """
    return basis_pursuit(A, y, **kwargs)  # type: ignore[arg-type]


def lasso_ista(
    A: np.ndarray,
    y: np.ndarray,
    *,
    lambda_reg: float = 0.01,
    max_iter: int = 2000,
    tol: float = 1e-6,
    reference_for_psnr: np.ndarray | None = None,
    D_recon: np.ndarray | None = None,
    psnr_target_db: float | None = None,
    **kwargs: object,
) -> np.ndarray:
    """
    LASSO (relaxation convexe) : min ½||y - Aα||² + λ||α||_1.

    Résolu par ISTA (gradient + seuillage doux). Utile quand on préfère un
    terme quadratique de fidélité plutôt que l’égalité stricte Aα = y.
    """
    _ = kwargs
    A = np.asarray(A, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    _, K = A.shape

    # Pas de gradient (Lipschitz de ∇f)
    L = float(np.linalg.norm(A, ord=2) ** 2)
    if L < 1e-15:
        L = 1.0
    step = 1.0 / L

    alpha = np.zeros(K, dtype=np.float64)
    for _ in range(max_iter):
        grad = A.T @ (A @ alpha - y)
        alpha_new = _soft_threshold(alpha - step * grad, lambda_reg * step)
        if np.linalg.norm(alpha_new - alpha) < tol * max(1.0, np.linalg.norm(alpha)):
            alpha = alpha_new
            break
        alpha = alpha_new
        if _psnr_stop(reference_for_psnr, D_recon, alpha, psnr_target_db):
            break

    return alpha


def main_methode(
    patches: list[np.ndarray],
    Phi: np.ndarray,
    D: np.ndarray,
    methodes: str | list[str],
    method_params: dict[str, dict] | None = None,
    *,
    reference_patch_vectors: np.ndarray | None = None,
    psnr_target_db: float | None = None,
) -> dict:
    """
    Applique plusieurs méthodes sur une liste de patchs (matrices 2D ou vecteurs).

    Si `reference_patch_vectors` a les colonnes des patchs originaux (N × Nb) et
    `psnr_target_db` est fixé, on transmet au solveur le critère PSNR (avec D).
    """
    from backend.Tratement_Image import vectoriser
    from backend.utils.mesure import apply_measurement

    if isinstance(methodes, str):
        methodes = [methodes]

    method_params = method_params or {}
    A = Phi @ D

    mesures = []
    alphas_by_method = {m.lower(): [] for m in methodes}
    reconstructed_by_method = {m.lower(): [] for m in methodes}

    dispatch: dict[str, Callable[..., np.ndarray]] = {
        "mp": mp,
        "omp": omp,
        "stomp": stomp,
        "cosamp": cosamp,
        "irls": irls,
        "bp": basis_pursuit,
        "basis_pursuit": basis_pursuit,
        "lp": lp,
        "lasso": lasso_ista,
        "lasso_ista": lasso_ista,
    }

    for idx, patch in enumerate(patches):
        x = vectoriser(patch)
        y = apply_measurement(Phi, x)
        mesures.append(y)

        ref_col = None
        if reference_patch_vectors is not None:
            ref_col = np.asarray(reference_patch_vectors[:, idx], dtype=np.float64)

        for methode in methodes:
            nom = methode.lower()
            params = dict(method_params.get(nom, {}))

            if ref_col is not None and psnr_target_db is not None:
                params.setdefault("reference_for_psnr", ref_col)
                params.setdefault("D_recon", D)
                params.setdefault("psnr_target_db", psnr_target_db)

            if nom not in dispatch:
                raise ValueError(f"Méthode inconnue : {methode}")

            alpha = dispatch[nom](A, y, **params)
            x_hat = D @ alpha

            alphas_by_method[nom].append(alpha)
            reconstructed_by_method[nom].append(x_hat)

    return {
        "mesures": mesures,
        "alphas_by_method": alphas_by_method,
        "reconstructed_by_method": reconstructed_by_method,
    }
