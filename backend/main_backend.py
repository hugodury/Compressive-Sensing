"""
Enchaîne les appels à patch() pour chaque méthode demandée, puis calcule PSNR/MSE/temps.

Les réglages passent par params (setupParam) : patch_params pour Φ, max_patches, s_cosamp_auto,
psnr_stop, etc. ; method_params pour max_iter, epsilon, s (CoSaMP), t (StOMP).
"""

from __future__ import annotations

import resource
import time
from typing import Any

from backend.Tratement_Image import patch
from backend.utils.Metrics import compute_all_metrics
from backend.utils.empreinte import fusionner_empreinte_dans_resultat
from backend.utils.stockage_compressif import estimer_stockage_bcs


def _normaliser_methodes(methodes: str | list[str]) -> list[str]:
    """Accepte une méthode seule (str) ou une liste, et renvoie toujours une liste."""
    if isinstance(methodes, str):
        return [methodes]
    return list(methodes)


def _ajouter_infos_patch_dans_metrics(metrics: dict[str, Any], out_patch: dict[str, Any]) -> None:
    """Copie dans metrics les infos utiles remontées par patch()."""
    if "coherence_mutuelle_cours" in out_patch:
        metrics["coherence_mutuelle_cours"] = out_patch["coherence_mutuelle_cours"]
        metrics["pourcentage_mesures"] = out_patch["pourcentage_mesures"]
        metrics["nb_mesures_M"] = out_patch["nb_mesures_M"]
    if out_patch.get("s_cosamp_utilise") is not None:
        metrics["s_cosamp_utilise"] = out_patch["s_cosamp_utilise"]
    if out_patch.get("cosamp_s_mode") is not None:
        metrics["cosamp_s_mode"] = out_patch["cosamp_s_mode"]
    if out_patch.get("time_limit_reached") is not None:
        metrics["time_limit_reached"] = bool(out_patch["time_limit_reached"])
        metrics["max_time_s"] = out_patch.get("max_time_s")
        metrics["nb_patchs_reconstruits"] = out_patch.get("nb_patchs_reconstruits")
        metrics["nb_patchs_total"] = out_patch.get("nb_patchs_total")


def main_backend(params: dict[str, Any]) -> dict[str, Any]:
    """Traitement backend complet pour une image : patchs, reconstruction, métriques, empreinte."""
    t_wall_debut = time.perf_counter()
    rusage_debut = None
    try:
        rusage_debut = resource.getrusage(resource.RUSAGE_SELF)
    except (OSError, AttributeError):
        pass

    # Paramètres liés à patch() (sinon on reprend les paramètres globaux).
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
    patch_max_time_s = patch_params.get("max_time_s")
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

    methodes = _normaliser_methodes(params["methodes"])
    method_params = params.get("method_params") or {}

    images_by_method: dict[str, Any] = {}
    metrics_by_method: dict[str, Any] = {}
    alphas_by_method: dict[str, Any] = {}

    # 2) Reconstruction pour chaque méthode demandée
    for methode in methodes:
        nom = str(methode).lower()
        mparams = method_params.get(nom, {})

        t0 = time.perf_counter()
        out_patch = patch(
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
            max_time_s=patch_max_time_s,
        )
        t1 = time.perf_counter()

        reconstructed = out_patch["image_reconstruite"]
        images_by_method[nom] = reconstructed
        if "alphas" in out_patch:
            alphas_by_method[nom] = out_patch["alphas"]

        # On ne dispose pas encore d'un alpha "global" exposé par patch(),
        # donc on calcule les métriques image + temps ici.
        metrics = compute_all_metrics(
            image_originale,
            reconstructed,
            start=t0,
            end=t1,
        )
        _ajouter_infos_patch_dans_metrics(metrics, out_patch)
        metrics_by_method[nom] = metrics

    resultat: dict[str, Any] = {
        "params": params,
        "patch": base,
        "original": image_originale,
        "images_by_method": images_by_method,
        "metrics": metrics_by_method,
        "alphas_by_method": alphas_by_method,
        "n_patches": int(original.shape[1]),
    }

    # Estimation de stockage théorique (fichier source vs données compressées y+Phi).
    if methodes:
        m0 = metrics_by_method.get(methodes[0], {})
        M_val = m0.get("nb_mesures_M")
        if M_val is not None:
            hr, wr = image_originale.shape
            img_path = str(params.get("image_path") or "").strip() or None
            resultat["stockage_bcs"] = estimer_stockage_bcs(
                int(hr),
                int(wr),
                int(resultat["n_patches"]),
                int(patch_B),
                int(M_val),
                chemin_fichier_source=img_path,
            )

    # Ajout de l'estimation d'empreinte carbone (si activée).
    fusionner_empreinte_dans_resultat(
        resultat,
        params,
        t_wall_debut=t_wall_debut,
        rusage_debut=rusage_debut,
        contexte="main_backend",
    )
    return resultat