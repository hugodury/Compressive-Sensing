"""
Point d'entrée du backend.
"""

from __future__ import annotations

from typing import Any

from Tratement_Image import main_patch, main_reconstruct_patch
from utils.Dictionnaire import main_dictionnaire
from utils.Methode import main_methode
from utils.Metrics import main_metrics
from utils.Save import main_save
from utils.mesure import main_mesure


def main_backend(params: dict[str, Any]) -> dict[str, Any]:
    patch_data = main_patch(
        image_path=params["image_path"],
        B=params["B"],
    )

    dictionnaire_data = main_dictionnaire(
        patches=patch_data["patches"],
        dictionary_type=params["dictionary_type"],
        n_atoms=params["n_atoms"],
        n_iter=params["n_iter_ksvd"],
        B=params["B"],
    )

    mesure_data = main_mesure(
        patches=patch_data["patches"],
        D=dictionnaire_data["D"],
        M=params["M"],
        N=params["N"],
        mode=params["measurement_mode"],
        seed=params.get("seed"),
    )

    methode_data = main_methode(
        patches=mesure_data["mesures"],
        A=mesure_data["A"],
        methodes=params["methodes"],
        method_params=params.get("method_params"),
    )

    reconstruct_data = main_reconstruct_patch(
        alphas_by_method=methode_data["alphas_by_method"],
        D=dictionnaire_data["D"],
        image_shape=patch_data["image_shape"],
        B=params["B"],
        patch_meta=patch_data.get("patch_meta"),
    )

    metrics_data = main_metrics(
        original=patch_data["image"],
        reconstructed_by_method=reconstruct_data["images_by_method"],
    )

    save_data = main_save(
        output_path=params["output_path"],
        original=patch_data["image"],
        reconstructed_by_method=reconstruct_data["images_by_method"],
        metrics_by_method=metrics_data,
        params=params,
    )

    return {
        "params": params,
        "patch": patch_data,
        "dictionnaire": dictionnaire_data,
        "mesure": mesure_data,
        "methode": methode_data,
        "reconstruction": reconstruct_data,
        "metrics": metrics_data,
        "save": save_data,
    }
