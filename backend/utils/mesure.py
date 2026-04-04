#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Génération de matrices de mesure (Phi) et application y = Phi x.
"""

from __future__ import annotations

import numpy as np

_MODES_VALIDES = frozenset({"gaussian", "uniform", "bernoulli_1", "bernoulli_01"})

# Cours 4 (réduction de dimension) — section « Exemples de matrices de mesure »
# Φ1 : uniforme ; Φ2 : Bernoulli {-1,+1} ; Φ3 : Bernoulli {0,1} ; Φ4 : gaussienne N(0,1/M)
_MODES_MESURE_COURS: dict[str, str] = {
    "phi1": "uniform",
    "φ1": "uniform",
    "phi2": "bernoulli_1",
    "φ2": "bernoulli_1",
    "phi3": "bernoulli_01",
    "φ3": "bernoulli_01",
    "phi4": "gaussian",
    "φ4": "gaussian",
}

# Projet PDF section 6 — pourcentages P de mesures (M = ceil(P*N/100))
POURCENTAGES_MESURES_PROJET: tuple[int, ...] = (15, 20, 25, 30, 50, 75)


def resolve_measurement_mode(mode: str) -> str:
    """
    Résout un libellé du cours / du projet vers un mode interne.
    Ex. phi1, Φ1 → uniform ; phi4 → gaussian.
    """
    key = mode.strip().lower().replace("φ", "phi")
    if key in _MODES_MESURE_COURS:
        return _MODES_MESURE_COURS[key]
    return mode.lower().strip()


def pourcentage_vers_M(P: float, N: int) -> int:
    """
    Nombre de mesures M pour un pourcentage P (comme au projet) :
    M = ceil(P * N / 100).
    """
    if N < 1:
        raise ValueError("N doit être >= 1.")
    p = float(P)
    if p < 0.0 or p > 100.0:
        raise ValueError("P doit être dans [0, 100] (pourcentage de mesures).")
    return int(np.ceil(p * N / 100.0))


def liste_M_pour_pourcentages_projet(N: int) -> dict[int, int]:
    """Pour N fixé, M pour chaque P ∈ {15,20,25,30,50,75}."""
    return {int(p): pourcentage_vers_M(p, N) for p in POURCENTAGES_MESURES_PROJET}


def generate_measurement_matrix(
    ratio: float,
    N: int,
    mode: str,
    *,
    p: float = 0.5,
    seed: int | None = None,
    M: int | None = None,
) -> np.ndarray:
    """
    Génère Phi ∈ R^{M×N}.

    Si `M` est fourni, on l’utilise directement (nombre de mesures).
    Sinon `ratio` (fraction ou pourcentage) détermine M via `compute_ratio`.

    modes (ou alias du cours / sujet) :
    - gaussian / phi4 : N(0,1)/sqrt(M)
    - uniform / phi1 : U(0,1)/sqrt(M)
    - bernoulli_1 / phi2 : {-1,+1} (probabilité p de -1) / sqrt(M)
    - bernoulli_01 / phi3 : {0,1} / sqrt(M)
    """
    if N < 1:
        raise ValueError("N doit être un entier >= 1.")

    mode_norm = resolve_measurement_mode(mode)
    if mode_norm not in _MODES_VALIDES:
        raise ValueError(f"mode doit être parmi {sorted(_MODES_VALIDES)} ou alias phi1…phi4, reçu : {mode!r}.")

    if M is not None:
        M_val = int(M)
        if M_val < 1 or M_val > N:
            raise ValueError(f"M doit être dans [1, N], reçu M={M_val}, N={N}.")
    else:
        M_val = compute_ratio(ratio, N)
    if M_val < 1:
        raise ValueError("M doit être un entier >= 1.")
    if M_val > N:
        raise ValueError("Pour une acquisition compressée, il faut M <= N.")

    if not (0.0 <= p <= 1.0):
        raise ValueError("p doit être dans [0, 1].")

    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(M_val)

    if mode_norm == "gaussian":
        return rng.standard_normal(size=(M_val, N), dtype=np.float64) * scale

    if mode_norm == "uniform":
        return rng.uniform(0.0, 1.0, size=(M_val, N)).astype(np.float64, copy=False) * scale

    if mode_norm == "bernoulli_1":
        u = rng.random(size=(M_val, N))
        phi = np.where(u < p, -1.0, 1.0)
        return (phi * scale).astype(np.float64, copy=False)

    # bernoulli_01
    bits = rng.integers(0, 2, size=(M_val, N), dtype=np.int8)
    return bits.astype(np.float64, copy=False) * scale


def apply_measurement(Phi: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Calcule y = Phi x.

    x peut être :
    - un vecteur (N,)
    - une matrice (N, K) (K patchs en colonnes)
    """
    Phi = np.asarray(Phi, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    if Phi.ndim != 2:
        raise ValueError("Phi doit être une matrice 2D.")

    M, N = Phi.shape

    if x.ndim == 1:
        if x.shape[0] != N:
            raise ValueError(f"x doit avoir une taille N={N}, obtenu {x.shape[0]}.")
        return Phi @ x

    if x.ndim == 2:
        if x.shape[0] != N:
            raise ValueError(f"x doit avoir N lignes (= {N}), obtenu {x.shape[0]}.")
        return Phi @ x

    raise ValueError("x doit être un vecteur (N,) ou une matrice (N, K).")


def compute_ratio(ratio: float, N: int) -> int:
    """
    Convertit un ratio (fraction ou pourcentage) en nombre de mesures M.

    - Si ratio <= 1 : interprété comme fraction (ex: 0.75)
    - Si ratio  > 1 : interprété comme pourcentage (ex: 75)
    """
    if N < 1:
        raise ValueError("N doit être >= 1.")
    r = float(ratio)
    if r < 0.0:
        raise ValueError("ratio doit être >= 0.")

    if r <= 1.0:
        M = int(np.ceil(r * N))
    else:
        if r > 100.0:
            raise ValueError("ratio en pourcentage doit être dans [0, 100].")
        M = int(np.ceil((r / 100.0) * N))

    return M


def compute_coherence_cours_phi_d(Phi: np.ndarray, D: np.ndarray) -> float:
    """
    Cohérence mutuelle μ(Φ, D) du cours (ch. 4) :

        μ(Φ, D) = max_{i,j} |⟨φ_i, d_j⟩| / (√N ‖φ_i‖ ‖d_j‖)

    où φ_i est la i-ème ligne de Φ et d_j la j-ème colonne de D (vecteurs de R^N).
    Tableau du sujet PDF section 6 : utiliser cette définition.
    """
    Phi = np.asarray(Phi, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)

    if Phi.ndim != 2 or D.ndim != 2:
        raise ValueError("Phi et D doivent être des matrices 2D.")
    M, N = Phi.shape
    if D.shape[0] != N:
        raise ValueError("Dimensions incompatibles : Phi.shape[1] doit égaler D.shape[0].")

    P = Phi @ D
    row_norms = np.linalg.norm(Phi, axis=1)
    col_norms = np.linalg.norm(D, axis=0)
    if np.any(row_norms < 1e-15) or np.any(col_norms < 1e-15):
        raise ValueError("Une ligne de Φ ou une colonne de D a une norme quasi nulle.")

    S = np.abs(P) / (row_norms[:, np.newaxis] * col_norms[np.newaxis, :])
    return float(np.max(S) / np.sqrt(float(N)))


def compute_coherence(Phi: np.ndarray, D: np.ndarray) -> float:
    """
    Cohérence des colonnes de A = ΦD (normalisées) : max_{i≠j} |⟨a_i, a_j⟩| / (‖a_i‖‖a_j‖).
    Utile pour analyser le dictionnaire équivalent en domaine des mesures ; ce n’est pas
    la même quantité que μ(Φ, D) du cours — voir `compute_coherence_cours_phi_d`.
    """
    Phi = np.asarray(Phi, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)

    if Phi.ndim != 2 or D.ndim != 2:
        raise ValueError("Phi et D doivent être des matrices 2D.")
    if Phi.shape[1] != D.shape[0]:
        raise ValueError("Dimensions incompatibles : Phi.shape[1] doit = D.shape[0].")

    A = Phi @ D  # (M, K)
    _, K = A.shape
    if K < 2:
        return 0.0

    norms = np.linalg.norm(A, axis=0, keepdims=True)
    if np.any(norms < 1e-15):
        raise ValueError("Une colonne de Phi@D est quasi nulle.")

    A_n = A / norms
    G = A_n.T @ A_n
    np.fill_diagonal(G, 0.0)
    return float(np.max(np.abs(G)))


def main(
    ratio: float,
    N: int,
    mode: str,
    seed: int | None,
    xi: np.ndarray,
) -> np.ndarray:
    """
    Crée la matrice de mesure Phi avec (ratio, N, mode, seed),
    applique y = Phi x aux patchs vectorisés xi, et renvoie yi.
    """
    Phi = generate_measurement_matrix(ratio=ratio, N=N, mode=mode, seed=seed)
    yi = apply_measurement(Phi, xi)
    return yi
