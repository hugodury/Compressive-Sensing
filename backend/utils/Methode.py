"""
Méthodes de calcul (algos parcimonieux).

Ici on fournit des fonctions directement appelables pour MP, OMP, StOMP et CoSaMP.
"""

from __future__ import annotations

import numpy as np


def mp(D: np.ndarray, x: np.ndarray, *, max_iter: int = 1000, epsilon: float = 1e-6) -> np.ndarray:
    """
    Matching Pursuit (MP).

    D : matrice de dictionnaire (N, K)
    x : signal (N,)
    """
    D = np.asarray(D, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    N, K = D.shape

    alpha = np.zeros(K, dtype=np.float64)
    residuel = x.copy()

    # Dans tes scripts d'origine, la “normalisation” utilisée est la norme globale de D.
    denom = float(np.linalg.norm(D))
    k = 0
    while k < max_iter and np.linalg.norm(residuel) > epsilon:
        # Choix de l'atome le plus corrélé au résiduel
        corr = D.T @ residuel
        if denom > 0:
            corr = np.abs(corr) / denom
        mk = int(np.argmax(corr))

        dj = D[:, mk]
        dj_norm2 = float(np.linalg.norm(dj) ** 2)
        if dj_norm2 < 1e-15:
            break

        # Mise à jour du coefficient et du résiduel
        zmk = float((dj.T @ residuel) / dj_norm2)
        alpha[mk] += zmk
        residuel = residuel - zmk * dj

        k += 1

    return alpha


def omp(D: np.ndarray, x: np.ndarray, *, max_iter: int = 1000, epsilon: float = 1e-6) -> np.ndarray:
    """
    Orthogonal Matching Pursuit (OMP).

    D : matrice de dictionnaire (N, K)
    x : signal (N,)
    """
    D = np.asarray(D, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    N, K = D.shape

    alpha = np.zeros(K, dtype=np.float64)
    residuel = x.copy()

    P: list[int] = []  # indices des atomes sélectionnés
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

        # Moindres carrés sur les atomes actifs pour annuler la composante “orthogonale”
        alpha_k, *_ = np.linalg.lstsq(Dk, x, rcond=None)
        residuel = x - Dk @ alpha_k

        k += 1

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
) -> np.ndarray:
    """
    StOMP (Stagewise OMP).

    Sélection de plusieurs atomes au même tour via un seuil calculé.
    """
    D = np.asarray(D, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    N, K = D.shape

    alpha = np.zeros(K, dtype=np.float64)
    P: list[int] = []
    residuel = x.copy()

    k = 0
    while k < max_iter and np.linalg.norm(residuel) > eps:
        # Contributions de chaque atome au résiduel courant
        C = np.zeros(K, dtype=np.float64)
        for j in range(K):
            dj = D[:, j]
            norme_dj = np.linalg.norm(dj)
            if norme_dj > 1e-15:
                C[j] = np.abs(dj.T @ residuel) / norme_dj

        # Seuil de sélection
        seuil = float(t * np.linalg.norm(residuel) / np.sqrt(K))

        # Tous les atomes dont la contribution dépasse le seuil
        Lambda = [j for j in range(K) if C[j] > seuil]
        if not Lambda:
            break

        # Mise à jour du support (on évite les doublons)
        for j in Lambda:
            if j not in P:
                P.append(j)
        P.sort()

        DS = D[:, P]
        # Moindres carrés sur les atomes actifs
        alphak, *_ = np.linalg.lstsq(DS, x, rcond=None)
        residuel = x - DS @ alphak

        # Reconstruction de alpha “complet”
        alpha = np.zeros(K, dtype=np.float64)
        alpha[P] = alphak

        k += 1

    return alpha


def cosamp(
    D: np.ndarray,
    x: np.ndarray,
    *,
    max_iter: int = 100,
    epsilon: float = 1e-6,
    s: int = 6,
) -> np.ndarray:
    """
    CoSaMP (Compressive Sampling Matching Pursuit).

    D : dictionnaire (N, K), x : signal (N,)
    s : parcimonie supposée (ordre)
    """
    D = np.asarray(D, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    N, K = D.shape
    if s < 1:
        raise ValueError("s doit être >= 1.")

    alpha = np.zeros(K, dtype=np.float64)
    residuel = x.copy()
    supp: set[int] = set()

    # On adapte automatiquement si s dépasse la taille du dictionnaire
    s_eff = min(s, K)

    k = 1
    while k <= max_iter and np.linalg.norm(residuel) > epsilon:
        # 1) Sélection : 2s atomes les plus corrélés
        denom = float(np.linalg.norm(D))
        c = D.T @ residuel
        if denom > 0:
            c = np.abs(c) / denom
        else:
            c = np.abs(c)

        top2 = min(2 * s_eff, K)
        supp1 = set(np.argsort(c)[-top2:])

        # 2) Fusion du support
        merged = list(supp | supp1)
        if not merged:
            break

        # 3) Estimation sur le support candidat
        As = D[:, merged]
        z = np.linalg.pinv(As) @ x

        # 4) Rejet : garder les s meilleurs coefficients
        top_local = min(s_eff, len(merged))
        local_best = np.argsort(np.abs(z))[-top_local:]
        supp = set(merged[i] for i in local_best)

        # 5) Estimation sur le support final
        final_idx = sorted(list(supp))
        Af = D[:, final_idx]
        alpha = np.zeros(K, dtype=np.float64)
        alpha[final_idx] = np.linalg.pinv(Af) @ x

        # 6) Mise à jour du résiduel
        residuel = x - D @ alpha

        k += 1

    return alpha



def main_methode(
    patches: list[np.ndarray],
    Phi: np.ndarray,
    D: np.ndarray,
    methodes: str | list[str],
    method_params: dict[str, dict] | None = None,
) -> dict:
    if isinstance(methodes, str):
        methodes = [methodes]

    method_params = method_params or {}
    A = Phi @ D

    mesures = []
    alphas_by_method = {m.lower(): [] for m in methodes}
    reconstructed_by_method = {m.lower(): [] for m in methodes}

    for patch in patches:
        x = vectoriser(patch)
        y = apply_measurement(Phi, x)
        mesures.append(y)

        for methode in methodes:
            nom = methode.lower()
            params = method_params.get(nom, {})

            if nom == "mp":
                alpha = mp(A, y, **params)
            elif nom == "omp":
                alpha = omp(A, y, **params)
            elif nom == "stomp":
                alpha = stomp(A, y, **params)
            elif nom == "cosamp":
                alpha = cosamp(A, y, **params)
            elif nom == "irls":
                alpha = irls(A, y, **params)
            else:
                raise ValueError(f"Méthode inconnue : {methode}")

            x_hat = D @ alpha

            alphas_by_method[nom].append(alpha)
            reconstructed_by_method[nom].append(x_hat)

    return {
        "mesures": mesures,
        "alphas_by_method": alphas_by_method,
        "reconstructed_by_method": reconstructed_by_method,
    }