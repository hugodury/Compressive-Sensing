"""
Sauvegarde des résultats du Compressive Sensing.
"""

import os
import time
import csv
from typing import Any

import numpy as np
from PIL import Image

from backend.utils.stockage_compressif import enrichir_stockage_apres_export, stockage_dict_pour_sauvegarde


def _taille_totale_dossier(dossier: str) -> int:
    total = 0
    for racine, _, fichiers in os.walk(dossier):
        for fn in fichiers:
            chemin = os.path.join(racine, fn)
            try:
                total += os.path.getsize(chemin)
            except OSError:
                pass
    return total


def save_results(resultats: dict[str, Any], output_path: str) -> None:
    """
    Sauvegarde l'image originale, les images reconstruites et les métriques 
    dans un dossier horodaté selon l'arborescence du projet.
    """
    # 1. Création du dossier horodaté (format jj.mm.hh.mm)
    horodatage = time.strftime("%d.%m.%H.%M")
    dossier_sauvegarde = os.path.join(output_path, horodatage)
    
    # Création des sous-dossiers
    os.makedirs(dossier_sauvegarde, exist_ok=True)
    os.makedirs(os.path.join(dossier_sauvegarde, "Graph"), exist_ok=True)

    # 2. Sauvegarde de l'image originale (pour comparaison visuelle)
    img_originale = np.clip(resultats["original"], 0, 255).astype(np.uint8)
    Image.fromarray(img_originale).save(os.path.join(dossier_sauvegarde, "Image_Originale.png"))

    # 3. Sauvegarde des images reconstruites
    for methode, image_array in resultats["images_by_method"].items():
        img_reconstruite = np.clip(image_array, 0, 255).astype(np.uint8)
        nom_fichier = f"Image_Reconstruite_{methode.upper()}.png"
        Image.fromarray(img_reconstruite).save(os.path.join(dossier_sauvegarde, nom_fichier))

    # 4. Sauvegarde des métriques dans un fichier CSV
    chemin_csv = os.path.join(dossier_sauvegarde, "metrics.csv")
    extra_cols = [
        "pourcentage_mesures",
        "coherence_mutuelle_cours",
        "nb_mesures_M",
        "s_cosamp_utilise",
        "cosamp_s_mode",
    ]
    sample = next(iter(resultats["metrics"].values()), {})
    optionnelles = [c for c in extra_cols if c in sample]
    with open(chemin_csv, mode="w", newline="", encoding="utf-8") as fichier_csv:
        writer = csv.writer(fichier_csv)
        entetes = ["Methode", "PSNR (dB)", "MSE", "Erreur Relative", "Temps (s)"]
        entetes += [c.replace("_", " ").title() for c in optionnelles]
        writer.writerow(entetes)
        for methode, metrics in resultats["metrics"].items():
            ligne = [
                methode.upper(),
                round(metrics.get("psnr", 0), 2),
                round(metrics.get("mse", 0), 4),
                round(metrics.get("relative_error", 0), 4),
                round(metrics.get("execution_time", 0), 2),
            ]
            for c in optionnelles:
                v = metrics.get(c, "")
                ligne.append(round(v, 6) if isinstance(v, float) else v)
            writer.writerow(ligne)

    emp = resultats.get("empreinte")
    if isinstance(emp, dict) and emp.get("message"):
        chemin_emp = os.path.join(dossier_sauvegarde, "empreinte_estimation.txt")
        with open(chemin_emp, mode="w", encoding="utf-8") as f:
            f.write(str(emp.get("message", "")).strip() + "\n")
            for k in (
                "duree_wall_s",
                "duree_cpu_process_s",
                "energie_estimee_wh",
                "co2e_g_estime",
                "energie_wh_temps_cpu",
                "co2e_g_estime_temps_cpu",
                "hypothese_puissance_w",
                "hypothese_g_co2_par_kwh",
            ):
                if k in emp and emp[k] is not None:
                    f.write(f"{k}: {emp[k]}\n")

    taille_export = _taille_totale_dossier(dossier_sauvegarde)
    stk = resultats.get("stockage_bcs")
    if isinstance(stk, dict):
        resultats["stockage_bcs"] = enrichir_stockage_apres_export(stk, dossier_sauvegarde, taille_export)
    stk_final = resultats.get("stockage_bcs")
    if isinstance(stk_final, dict) and stk_final.get("message"):
        chemin_stk = os.path.join(dossier_sauvegarde, "stockage_compression.txt")
        with open(chemin_stk, mode="w", encoding="utf-8") as f:
            f.write(stockage_dict_pour_sauvegarde(stk_final))

    print(f"\nRésultats (Images et CSV) sauvegardés dans :\n -> {dossier_sauvegarde}")