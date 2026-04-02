"""
Calcul des métriques.
"""


import math
from typing import Any, Dict, Optional

import numpy as np


def _validate_same_shape(original: np.ndarray, reconstructed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Vérifie que `original` et `reconstructed` ont la même forme puis convertit en float.
    """
    original_arr = np.asarray(original)
    reconstructed_arr = np.asarray(reconstructed)
    if original_arr.shape != reconstructed_arr.shape:
        raise ValueError(
            f"Les formes ne correspondent pas : original={original_arr.shape}, reconstruit={reconstructed_arr.shape}."
        )
    return original_arr.astype(np.float64, copy=False), reconstructed_arr.astype(np.float64, copy=False)


def _infer_peak_value(original_arr: np.ndarray, reconstructed_arr: np.ndarray) -> float:
    """
    Valeur de pic utilisée pour PSNR.

    - Si les tableaux ressemblent à des images uint8 : peak = 255
    - Sinon : peak = max(|valeurs|) sur original et reconstruit
    """
    if np.issubdtype(original_arr.dtype, np.integer) or np.issubdtype(reconstructed_arr.dtype, np.integer):
        # Cas typique images 8 bits
        if original_arr.dtype == np.uint8 or reconstructed_arr.dtype == np.uint8:
            return 255.0
        # Sinon on prend la borne “pratique” via max
    peak = float(np.max(np.abs(original_arr)))
    peak = max(peak, float(np.max(np.abs(reconstructed_arr))))
    return peak


def compute_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Calcule l'erreur quadratique moyenne (Mean Squared Error).

    Formule :
        MSE = (1/N) * ||x - x_hat||²

    Parameters
    ----------
    original : np.ndarray
        Image originale.
    reconstructed : np.ndarray
        Image reconstruite.

    Returns
    -------
    float
        Valeur du MSE.
    """
    original_arr, reconstructed_arr = _validate_same_shape(original, reconstructed)
    error = original_arr - reconstructed_arr
    mse = np.mean(error ** 2)
    return float(mse)


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Calcule le PSNR (Peak Signal-to-Noise Ratio).

    Formule :
        PSNR = 10 * log10((MAX²) / MSE)

    Si MSE = 0, on retourne +inf.

    Parameters
    ----------
    original : np.ndarray
        Image originale.
    reconstructed : np.ndarray
        Image reconstruite.

    Returns
    -------
    float
        Valeur du PSNR en dB.
    """
    original_arr, reconstructed_arr = _validate_same_shape(original, reconstructed)
    mse = compute_mse(original_arr, reconstructed_arr)

    if mse == 0.0:
        return float("inf")

    peak = _infer_peak_value(original_arr, reconstructed_arr)
    psnr = 10.0 * math.log10((peak ** 2) / mse)
    return float(psnr)


def compute_relative_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Calcule l'erreur relative.

    Formule :
        ||x - x_hat|| / ||x||

    Si ||x|| = 0 :
    - retourne 0 si les deux images sont identiques
    - sinon retourne +inf

    Parameters
    ----------
    original : np.ndarray
        Image originale.
    reconstructed : np.ndarray
        Image reconstruite.

    Returns
    -------
    float
        Erreur relative.
    """
    original_arr, reconstructed_arr = _validate_same_shape(original, reconstructed)

    numerator = np.linalg.norm(original_arr - reconstructed_arr)
    denominator = np.linalg.norm(original_arr)

    if denominator == 0.0:
        if numerator == 0.0:
            return 0.0
        return float("inf")

    rel_error = numerator / denominator
    return float(rel_error)


def compute_execution_time(start: float,end: float) -> float:
    """
    Calcule le temps d'exécution en secondes.

    Parameters
    ----------
    start : float
        Temps de départ.
    end : float
        Temps de fin.

    Returns
    -------
    float
        Durée d'exécution en secondes.
    """
    if start is None or end is None:
        raise ValueError("start et end ne peuvent pas être None.")

    execution_time = float(end) - float(start)

    if execution_time < 0:
        raise ValueError("Le temps de fin doit être supérieur ou égal au temps de départ.")

    return execution_time


def compute_parcimony(alpha: np.ndarray, *, eps: float = 1e-8) -> Dict[str, float]:
    """
    Mesure la parcimonie à partir des coefficients `alpha`.

    On renvoie :
    - `l0_approx` : nombre de coefficients dont |alpha_i| > eps
    - `sparsity_ratio` : l0_approx / nombre total de coefficients
    """
    a = np.asarray(alpha, dtype=np.float64).ravel()
    if a.size == 0:
        return {"l0_approx": 0.0, "sparsity_ratio": 0.0}

    nz = int(np.sum(np.abs(a) > float(eps)))
    return {
        "l0_approx": float(nz),
        "sparsity_ratio": float(nz) / float(a.size),
    }


def compute_all_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
    start: Optional[float] = None,
    end: Optional[float] = None,
    *,
    alpha: Optional[np.ndarray] = None,
    alpha_eps: float = 1e-8,
) -> Dict[str, Any]:
    """
    Calcule toutes les métriques utiles du projet.

    Parameters
    ----------
    original : np.ndarray
        Image originale.
    reconstructed : np.ndarray
        Image reconstruite.
    start : float, optional
        Temps de départ.
    end : float, optional
        Temps de fin.

    Returns
    -------
    dict
        Dictionnaire contenant les métriques calculées.
    """
    metrics = {
        "mse": compute_mse(original, reconstructed),
        "psnr": compute_psnr(original, reconstructed),
        "relative_error": compute_relative_error(original, reconstructed),
        "execution_time": compute_execution_time(start, end)
    }

    # Ajout optionnel : parcimonie à partir de `alpha` (si disponible).
    if alpha is not None:
        metrics.update(compute_parcimony(alpha, eps=alpha_eps))

    return metrics