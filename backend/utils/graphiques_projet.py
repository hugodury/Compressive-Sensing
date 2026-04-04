"""
Courbes PSNR (ou MSE) en fonction du ratio de mesures — sujet §6–§7, « graphiques ».

Nécessite matplotlib (optionnel dans le README).
"""

from __future__ import annotations

import copy
import os
import time
from typing import Any, Sequence

import numpy as np

from backend.main_backend import main_backend


def sweep_ratios_psnr(
    params_template: dict[str, Any],
    ratios: Sequence[float],
) -> dict[str, Any]:
    """
    Pour chaque valeur de ``ratio`` (fraction ]0,1] ou pourcentage ]0,100]), relance ``main_backend``
    avec les mêmes paramètres ailleurs.

    Retourne ``{"ratios": [...], "psnr_by_method": {methode: [..]}, "mse_by_method": {...}}``.
    """
    ratios_list = [float(r) for r in ratios]
    methodes = params_template["methodes"]
    if isinstance(methodes, str):
        methodes = [methodes]
    methodes = [str(m).lower() for m in methodes]

    psnr_by_method: dict[str, list[float]] = {m: [] for m in methodes}
    mse_by_method: dict[str, list[float]] = {m: [] for m in methodes}

    for r in ratios_list:
        p = copy.deepcopy(params_template)
        p["ratio"] = r
        n = int(p["B"]) * int(p["B"])
        if r <= 1.0:
            p["M"] = max(1, int(np.ceil(r * n)))
        else:
            p["M"] = max(1, int(np.ceil((r / 100.0) * n)))

        out = main_backend(p)
        for m in methodes:
            met = out["metrics"][m]
            psnr_by_method[m].append(float(met.get("psnr", 0.0)))
            mse_by_method[m].append(float(met.get("mse", 0.0)))

    return {
        "ratios": ratios_list,
        "psnr_by_method": psnr_by_method,
        "mse_by_method": mse_by_method,
    }


def sauvegarder_courbe_psnr_ratios(
    sweep: dict[str, Any],
    dossier_sortie: str,
    *,
    titre: str = "PSNR vs ratio de mesures",
    xlabel: str = "Ratio (fraction si ≤1, sinon %)",
) -> str:
    """
    Trace une courbe par méthode et enregistre un PNG dans ``dossier_sortie/Graph/``.
    Retourne le chemin du fichier PNG.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "Installe matplotlib pour générer les graphiques : pip install matplotlib"
        ) from e

    os.makedirs(os.path.join(dossier_sortie, "Graph"), exist_ok=True)
    horodatage = time.strftime("%d.%m.%H.%M")
    base = os.path.join(dossier_sortie, horodatage, "Graph")
    os.makedirs(base, exist_ok=True)
    path_png = os.path.join(base, "courbe_psnr_vs_ratio.png")

    ratios = sweep["ratios"]
    plt.figure(figsize=(8, 5))
    for methode, ys in sweep["psnr_by_method"].items():
        plt.plot(ratios, ys, marker="o", linewidth=1.5, label=methode.upper())
    plt.xlabel(xlabel)
    plt.ylabel("PSNR (dB)")
    plt.title(titre)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close()
    return path_png


def exporter_sweep_graphique(
    params_template: dict[str, Any],
    ratios: Sequence[float],
    output_path: str = "Data/Result",
) -> dict[str, Any]:
    """Enchaîne ``sweep_ratios_psnr`` + ``sauvegarder_courbe_psnr_ratios``."""
    sw = sweep_ratios_psnr(params_template, ratios)
    png = sauvegarder_courbe_psnr_ratios(sw, output_path)
    sw["graphique_psnr_png"] = png
    return sw
