"""
Sauvegarde des résultats du Compressive Sensing.
"""

import os
import time
import csv
from typing import Any
import numpy as np
from PIL import Image

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
    extra_cols = ["pourcentage_mesures", "coherence_mutuelle_cours", "nb_mesures_M", "s_cosamp_utilise"]
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

    print(f"\nRésultats (Images et CSV) sauvegardés dans :\n -> {dossier_sauvegarde}")