#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import numpy as np

from image_blocking import image_to_patch_vectors, load_grayscale_matrix

def matrice_mesure(M: int,N: int,*,type_phi: str,rng: np.random.Generator):
    if M < 1:
        raise ValueError("M doit être >= 1.")
    if N < 1:
        raise ValueError("N doit être >= 1.")
    if M > N:
        # possible mathématiquement mais plus pb de reduction
        raise ValueError("M doit être <= B² (sinon on ne compresse plus).")

    type_phi = type_phi.lower().strip()
    if type_phi == "gaussian":
        phi = rng.standard_normal(size=(M, N), dtype=np.float64) / np.sqrt(M)
        return phi
    if type_phi == "bernoulli":
        signes = rng.integers(0, 2, size=(M, N), dtype=np.int8)
        phi = (2.0 * signes - 1.0) / np.sqrt(M)
        return phi.astype(np.float64, copy=False)

    raise ValueError("type_phi doit être 'gaussian' ou 'bernoulli'.")


def encoder_bcs(matrice_patchs: np.ndarray,*,M: int,type_phi: str = "gaussian",seed: int | None = None):

    if matrice_patchs.ndim != 2:
        raise ValueError("matrice_patchs doit être une matrice 2D de taille (B², NB).")

    N, nb_blocs = matrice_patchs.shape
    rng = np.random.default_rng(seed)
    phi = matrice_mesure(M, N, type_phi=type_phi, rng=rng)
    mesures = phi @ matrice_patchs
    assert mesures.shape == (M, nb_blocs)
    return mesures, phi


def _ratio_vers_M(ratio_mesure: float, B: int) -> int:
    if not (0.0 < ratio_mesure <= 1.0):
        raise ValueError("ratio_mesure doit être dans (0, 1].")
    N = B * B
    M = int(np.ceil(ratio_mesure * N))
    return max(1, min(M, N))


