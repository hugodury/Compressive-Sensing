#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Génération de matrices de mesure (Phi) et application y = Phi x.
"""

from __future__ import annotations

import numpy as np

_MODES_VALIDES = frozenset({"gaussian", "uniform", "bernoulli_1", "bernoulli_01"})


def generate_measurement_matrix(
    M: int,
    N: int,
    mode: str,
    *,
    p: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """
    Génère Phi ∈ R^{M×N}.

    modes :
    - gaussian : N(0,1)/sqrt(M)
    - uniform : U(0,1)/sqrt(M)
    - bernoulli_1 : {-1,+1} (probabilité p de -1) / sqrt(M)
    - bernoulli_01 : {0,1} / sqrt(M)
    """
    if M < 1:
        raise ValueError("M doit être un entier >= 1.")
    if N < 1:
        raise ValueError("N doit être un entier >= 1.")
    if M > N:
        raise ValueError("Pour une acquisition compressée, il faut M <= N.")

    mode_norm = mode.lower().strip()
    if mode_norm not in _MODES_VALIDES:
        raise ValueError(f"mode doit être parmi {sorted(_MODES_VALIDES)}, reçu : {mode!r}.")
    if not (0.0 <= p <= 1.0):
        raise ValueError("p doit être dans [0, 1].")

    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(M)

    if mode_norm == "gaussian":
        return rng.standard_normal(size=(M, N), dtype=np.float64) * scale

    if mode_norm == "uniform":
        return rng.uniform(0.0, 1.0, size=(M, N)).astype(np.float64, copy=False) * scale

    if mode_norm == "bernoulli_1":
        u = rng.random(size=(M, N))
        phi = np.where(u < p, -1.0, 1.0)
        return (phi * scale).astype(np.float64, copy=False)

    # bernoulli_01
    bits = rng.integers(0, 2, size=(M, N), dtype=np.int8)
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


def compute_ratio(ratio: float, N: int) -> float: #ratio ex : 75 pour 75%
    """r = M / N."""
    if N < 1:
        raise ValueError("N doit être >= 1.")
    
    M = (ratio*N)/100

    # Calcul
    return M


def compute_coherence(Phi: np.ndarray, D: np.ndarray) -> float:
    """
    Cohérence mutuelle entre colonnes de A = Phi @ D (normalisées).
    Retour : max_{i!=j} |<a_i, a_j>|.
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

