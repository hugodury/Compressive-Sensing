"""
Point d'entrée du backend.
"""

from __future__ import annotations

import time
from typing import Any

from backend.Tratement_Image import patch
from backend.utils.Metrics import compute_all_metrics
from backend.Tratement_Image import apply_bilateral_filter

def main_backend(params: dict[str, Any]) -> dict[str, Any]:
    patch_params = params.get("patch_params") or {}
    patch_B = patch_params.get("B", params["B"])
    patch_nrows = patch_params.get("nrows")
    patch_ncols = patch_params.get("ncols")
    patch_order = patch_params.get("order", "C")
    patch_max_patches = patch_params.get("max_patches")
    patch_ratio = patch_params.get("ratio", params.get("ratio"))
    patch_mode_phi = patch_params.get("mode_phi", params.get("measurement_mode", "gaussian"))
    patch_seed = patch_params.get("seed", params.get("seed"))
    patch_psnr_stop = patch_params.get("psnr_stop", False)
    patch_psnr_target = patch_params.get("psnr_target_db", 45.0)
    patch_lambda_lasso = patch_params.get("lambda_lasso", 0.01)
    patch_norm_p = patch_params.get("norm_p", 0.5)
    patch_s_cosamp_auto = patch_params.get("s_cosamp_auto", False)
    patch_n_iter_ksvd = int(patch_params.get("n_iter_ksvd", params.get("n_iter_ksvd", 0)))
    patch_ksvd_train = patch_params.get("ksvd_train_patches")
    patch_dict_train = patch_params.get("dictionary_train_image_path", params.get("dictionary_train_image_path"))

    # 1) Découpage seul (référence de base + image originale recadrée)
    base = patch(
        image_path=params["image_path"],
        B=patch_B,
        nrows=patch_nrows,
        ncols=patch_ncols,
        order=patch_order,
        as_dict=True,
    )
    original = base["matrice_patchs"]  # (N, NB), utile pour diagnostics

    # Image originale (recadrée) pour les métriques
    from backend.Tratement_Image import load_grayscale_matrix  # import local pour éviter effets de bord

    image_full = load_grayscale_matrix(params["image_path"])
    n1, n2, _, _ = base["meta"]
    image_originale = image_full[:n1, :n2]

    methodes = params["methodes"]
    if isinstance(methodes, str):
        methodes = [methodes]
    method_params = params.get("method_params") or {}

    images_by_method: dict[str, Any] = {}
    metrics_by_method: dict[str, Any] = {}

    # 2) Reconstruction pour chaque méthode demandée
    for methode in methodes:
        nom = str(methode).lower()
        mparams = method_params.get(nom, {})

        t0 = time.perf_counter()
        out = patch(
            image_path=params["image_path"],
            B=patch_B,
            nrows=patch_nrows,
            ncols=patch_ncols,
            order=patch_order,
            as_dict=True,
            M=patch_params.get("M", params.get("M")),
            ratio=patch_ratio,
            method=nom,
            dictionary_type=params.get("dictionary_type", "dct"),
            n_atoms=params.get("n_atoms"),
            mode_phi=patch_mode_phi,
            seed=patch_seed,
            max_iter=mparams.get("max_iter", 20),
            epsilon=mparams.get("epsilon", 1e-6),
            t_stomp=mparams.get("t", 2.5),
            s_cosamp=mparams.get("s", 6),
            max_patches=patch_max_patches,
            psnr_stop=mparams.get("psnr_stop", patch_psnr_stop),
            psnr_target_db=mparams.get("psnr_target_db", patch_psnr_target),
            lambda_lasso=mparams.get("lambda_lasso", patch_lambda_lasso),
            norm_p=mparams.get("norm_p", patch_norm_p),
            s_cosamp_auto=mparams.get("s_cosamp_auto", patch_s_cosamp_auto),
            n_iter_ksvd=patch_n_iter_ksvd,
            ksvd_train_patches=patch_ksvd_train,
            dictionary_train_image_path=patch_dict_train,
        )
        t1 = time.perf_counter()

        reconstructed = out["image_reconstruite"]
        #reconstructed = apply_bilateral_filter(reconstructed_brut, d=5, sigma_color=50.0, sigma_space=50.0)
        images_by_method[nom] = reconstructed

        # On ne dispose pas encore d'un alpha "global" exposé par patch(),
        # donc on calcule les métriques image + temps ici.
        metrics = compute_all_metrics(
            image_originale,
            reconstructed,
            start=t0,
            end=t1,
        )
        if "coherence_mutuelle_cours" in out:
            metrics["coherence_mutuelle_cours"] = out["coherence_mutuelle_cours"]
            metrics["pourcentage_mesures"] = out["pourcentage_mesures"]
            metrics["nb_mesures_M"] = out["nb_mesures_M"]
        if out.get("s_cosamp_utilise") is not None:
            metrics["s_cosamp_utilise"] = out["s_cosamp_utilise"]
        metrics_by_method[nom] = metrics

    return {
        "params": params,
        "patch": base,
        "original": image_originale,
        "images_by_method": images_by_method,
        "metrics": metrics_by_method,
        "n_patches": int(original.shape[1]),
    }
