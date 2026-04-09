"""
Méthodes parcimonieuses et relaxations convexes pour le problème y ≈ A α
(avec A = Φ D en compressive sensing sur les patchs).

On regroupe : MP, OMP, StOMP, CoSaMP, **IRLS** (pseudo-norme ℓp, 0<p<1, §5.2 du sujet),
Basis Pursuit / LP (relaxation **L1** convexe), LASSO via ISTA.
Les critères d’arrêt classiques (résidu, max itérations) peuvent être complétés
par un PSNR cible si on fournit le patch original (cas expérimental / debug).
"""

from __future__ import annotations

import math
import time
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
    """Arrêt si le patch reconstruit atteint au moins psnr_target_db (expérimental)."""
    if reference_for_psnr is None or D_recon is None or psnr_target_db is None:
        return False
    x_hat = D_recon @ alpha
    p = compute_psnr(reference_for_psnr, x_hat)
    return not math.isinf(p) and p >= float(psnr_target_db)


def _deadline_reached(deadline_s: float | None) -> bool:
    """Retourne True si l'échéance d'exécution est dépassée."""
    return deadline_s is not None and time.perf_counter() >= float(deadline_s)


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
    Différence clé vs OMP / StOMP : un seul atome par tour ; pas de moindres carrés
    sur tout le support (étapes 2–3 = coeff. 1D puis résiduel non réorthogonalisé au sous-espace).
    PSNR d’arrêt optionnel si reference_for_psnr + D_recon + seuil.
    """
    deadline_s = kwargs.get("deadline_s")
    D = np.asarray(D, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    _, K = D.shape

    alpha = np.zeros(K, dtype=np.float64)
    residuel = x.copy()

    denom = float(np.linalg.norm(D))
    k = 0
    while k < max_iter and np.linalg.norm(residuel) > epsilon:
        if _deadline_reached(deadline_s):
            break
        corr = D.T @ residuel
        if denom > 0:
            corr = np.abs(corr) / denom
        mk = int(np.argmax(corr))

        dj = D[:, mk]
        dj_norm2 = float(np.linalg.norm(dj) ** 2)
        if dj_norm2 < 1e-15:
            break

        zmk = float((dj.T @ residuel) / dj_norm2)
        # MP : étape 2 = coeff. le long d’un seul atome (pas de MC sur tout le support).
        alpha[mk] += zmk
        # MP : étape 3 = on retranche seulement cette contribution ; le résiduel n’est
        # pas rendu orthogonal à tout le sous-espace (contrairement à OMP).
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
    """
    OMP : un atome par tour, mais à chaque fois moindres carrés sur **tout** le support courant
    (résiduel orthogonal à Vect(D_P)) — plus coûteux que MP, meilleure qualité en général.
    """
    deadline_s = kwargs.get("deadline_s")
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
        if _deadline_reached(deadline_s):
            break
        corr = D.T @ residuel
        if denom > 0:
            corr = np.abs(corr) / denom
        mk = int(np.argmax(corr))

        P.append(mk)
        Dk = D[:, P]

        # OMP : étape 2–3 = moindres carrés sur **tout** le support P (tous les atomes
        # choisis), donc résiduel orthogonal au sous-espace Vect(D_P) ; pas seulement
        # une mise à jour scalaire comme en MP.
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
    """
    StOMP : même idée de MC sur le support qu’OMP après sélection, mais plusieurs atomes
    peuvent entrer **en même temps** (seuillage du cours) au lieu d’un seul par itération.
    """
    deadline_s = kwargs.get("deadline_s")
    D = np.asarray(D, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    _, K = D.shape

    alpha = np.zeros(K, dtype=np.float64)
    P: list[int] = []
    residuel = x.copy()

    k = 0
    while k < max_iter and np.linalg.norm(residuel) > eps:
        if _deadline_reached(deadline_s):
            break
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

        # StOMP vs OMP : ici on peut ajouter **plusieurs** indices d’un coup (seuil),
        # alors qu’OMP n’en ajoute qu’**un** par tour.
        for j in Lambda:
            if j not in P:
                P.append(j)
        P.sort()

        DS = D[:, P]
        # Même principe qu’OMP après coup : MC sur le support élargi (étape type 2–3 du sujet).
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
    """
    CoSaMP : besoin d’un entier s (support final). Arrêt classique : ||r|| < epsilon ou max_iter.
    Si reference_for_psnr est fourni avec D_recon et psnr_target_db, arrêt dès PSNR atteint.
    """
    deadline_s = kwargs.get("deadline_s")
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
        if _deadline_reached(deadline_s):
            break
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
    p: float = 0.5,
    max_iter: int = 100,
    epsilon: float = 1e-6,
    delta: float = 1e-6,
    reference_for_psnr: np.ndarray | None = None,
    D_recon: np.ndarray | None = None,
    psnr_target_db: float | None = None,
    **kwargs: object,
) -> np.ndarray:
    """
    IRLS pour le problème (Pp) du sujet (§5.2) : minimiser ∑ᵢ |αᵢ|^p sous **Aα = y**, avec **0 < p < 1**.

    Le PDF ne demande pas une variante IRLS « uniquement L1 » : il relie IRLS à la pseudo-norme ℓp
    (p entre 0 et 1) et au problème (MCP2). Pour la **norme L1** sous contrainte, utiliser **BP/LP**
    (`basis_pursuit` / `lp`).

    Poids : wᵢ ∝ (|αᵢ|+δ)^{p-2} à chaque itération (W dépend de l’itéré précédent).
    """
    deadline_s = kwargs.get("deadline_s")
    if not (0.0 < p < 1.0):
        raise ValueError(
            "irls : p doit être dans ]0, 1[ (comme au §5.2). Pour la norme L1, utiliser bp ou lp."
        )

    A = np.asarray(A, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    M, _ = A.shape

    if M <= A.shape[1]:
        try:
            alpha = A.T @ np.linalg.solve(A @ A.T + 1e-10 * np.eye(M), y)
        except np.linalg.LinAlgError:
            alpha = np.linalg.pinv(A) @ y
    else:
        alpha = np.linalg.pinv(A) @ y

    eye_m = np.eye(M, dtype=np.float64)
    for _ in range(max_iter):
        if _deadline_reached(deadline_s):
            break
        t = np.abs(alpha) + delta
        # Poids pour la quadratisation de ∑ |α_i|^p : en pratique w_i ∝ t^{p-2}
        w_diag = (0.5 * p) * np.power(t, p - 2.0)
        w_diag = np.maximum(w_diag, 1e-20)
        w_inv = 1.0 / w_diag
        q_inv_mat = np.diag(w_inv)
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


# Alias : ancien nom, même implémentation (sujet PDF + §5.2)
irls_lp = irls


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
    deadline_s = kwargs.get("deadline_s")
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
        if _deadline_reached(deadline_s):
            break
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
        "irls_lp": irls,
        "irls_p": irls,
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
