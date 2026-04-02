#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np

from Traitement_Image.py import image_to_patch_vectors, load_grayscale_matrix

_MODES_VALIDES = frozenset(
    {
        "uniform",
        "bernoulli_1",
        "bernoulli_01",
        "gaussian",}
)


def generate_measurement_matrix(
    ratio : float
    N: int,
    mode: str,
    *,
    p: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """
    Génère Φ ∈ R^{M×N} (gaussienne,bernouilli(0,1), bernouilli(-1,1)=1, uniforme) avec M determiner par compute ratio
    `seed` optionnel : reproductibilité.
    """

    # Verfication des dims

    M = compute_ratio(ratio, N)

    if M < 1:
        raise ValueError("M doit être un entier >= 1.")
    if N < 1:
        raise ValueError("N doit être un entier >= 1 (taille du signal / patch aplati).")
    if M > N:
        raise ValueError("Pour une acquisition compressée, il faut M <= N.")

    # Verfication du mode

    mode_norm = mode.lower().strip()
    if mode_norm not in _MODES_VALIDES:
        raise ValueError(f"mode doit être parmi {sorted(_MODES_VALIDES)}, reçu : {mode!r}.")

    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(M)

    if mode_norm == "gaussian":
        return rng.standard_normal(size=(M, N), dtype=np.float64) * scale # gaussienne N(0,1)

    if mode_norm == "uniform":
        return rng.uniform(0.0, 1.0, size=(M, N)).astype(np.float64, copy=False) * scale # uniforme [0,1]

    if mode_norm in {"bernoulli_1"}: # berouilli [-1,1]
            raise ValueError("p doit être dans [0, 1].")
        u = rng.random(size=(M, N))
        phi = np.where(u < p, -1.0, 1.0)
        return (phi * scale).astype(np.float64, copy=False)

    # bernoulli_01
    bits = rng.integers(0, 2, size=(M, N), dtype=np.int8)
    return bits.astype(np.float64, copy=False) * scale


def apply_measurement(Phi: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Calcule y = Φ x (acquisition compressée).
    x peut être un vecteur (N,) ou (N, 1), ou une matrice (N, K) (K patchs en colonnes).
    """

    # Conversion type

    Phi = np.asarray(Phi, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    # Verfication des dims

    if Phi.ndim != 2:
        raise ValueError("Phi n'est pas de dim 2")
    M, N = Phi.shape
    if x.ndim == 1 or x.ndim == 2:
        if x.shape[0] != N:
            raise ValueError(f"Ligne diff de N.")
        return Phi @ x

    raise ValueError("x doit être de dim 1 vecteur ou 2 (matrice N*K).")


def compute_ratio(ratio: float, N: int) -> float:
    """
    Renvoie la taille M
    """
    # Verfication des dims
    if N < 1:
        raise ValueError("N doit être >= 1.")
    
    M = (ratio*N)/100

    # Calcul
    return M


def compute_coherence(Phi: np.ndarray, D: np.ndarray) -> float:
    """
    Cohérence mutuelle des colonnes de A = Φ D (normalisées), max |⟨a_i, a_j⟩| pour i ≠ j.
    Φ : (M, N), D : (N, K). Exige Φ.shape[1] == D.shape[0].
    """

    #Convertion type


    Phi = np.asarray(Phi, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)

    ## Verfication


    if Phi.ndim != 2 or D.ndim != 2:
        raise ValueError("Phi et D doivent être des matrices 2D.")
    if Phi.shape[1] != D.shape[0]:
        raise ValueError(
            f"Dimensions incompatibles : Phi a N={Phi.shape[1]} colonnes, "
            f"D doit avoir {Phi.shape[1]} lignes, obtenu D.shape[0]={D.shape[0]}."
        )

    ## Calcul

    A = Phi @ D
    _, K = A.shape
    if K < 2:
        return 0.0


    norms = np.linalg.norm(A, axis=0, keepdims=True) #normalisation colonne par colonne
    if np.any(norms < 1e-15):d
        raise ValueError("Une colonne de ΦD est quasi nulle (norme nulle).")
    A_n = A / norms #normalisation colonne par colonne

    G = A_n.T @ A_n #produit scalaire colonne par colonne
    np.fill_diagonal(G, 0.0) #on met les diag a 0
    return float(np.max(np.abs(G))) #retourne la valeur max de la matrice G



