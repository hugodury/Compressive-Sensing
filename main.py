"""
Point d'entrée du projet.
"""

from __future__ import annotations

import math
from typing import Any

from backend.main_backend import main_backend
from backend.utils.save import save_results

def setupParam(
    image_path: str,
    block_size: int,
    ratio: float,
    methodes: str | list[str],
    dictionary_type: str,
    measurement_mode: str = "gaussian",
    output_path: str = "Data/Result",
    n_atoms: int | None = None,
    n_iter_ksvd: int = 0,
    method_params: dict[str, dict[str, Any]] | None = None,
    patch_params: dict[str, Any] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    if block_size <= 0:
        raise ValueError("block_size doit être > 0.")
    if not (0 < ratio <= 1):
        raise ValueError("ratio doit être dans ]0, 1].")

    n = block_size * block_size
    m = max(1, math.ceil(ratio * n))

    if isinstance(methodes, str):
        methodes = [methodes]

    if n_atoms is None:
        n_atoms = n

    return {
        "image_path": image_path,
        "B": block_size,
        "ratio": ratio,
        "N": n,
        "M": m,
        "methodes": methodes,
        "dictionary_type": dictionary_type,
        "measurement_mode": measurement_mode,
        "output_path": output_path,
        "n_atoms": n_atoms,
        "n_iter_ksvd": n_iter_ksvd,
        "method_params": method_params or {},
        "patch_params": patch_params or {},
        "seed": seed,
    }


def main(
    image_path: str,
    block_size: int,
    ratio: float,
    methodes: str | list[str],
    dictionary_type: str,
    measurement_mode: str = "gaussian",
    output_path: str = "Data/Result",
    n_atoms: int | None = None,
    n_iter_ksvd: int = 0,
    method_params: dict[str, dict[str, Any]] | None = None,
    patch_params: dict[str, Any] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    params = setupParam(
        image_path=image_path,
        block_size=block_size,
        ratio=ratio,
        methodes=methodes,
        dictionary_type=dictionary_type,
        measurement_mode=measurement_mode,
        output_path=output_path,
        n_atoms=n_atoms,
        n_iter_ksvd=n_iter_ksvd,
        method_params=method_params,
        patch_params=patch_params,
        seed=seed,
    )
    return main_backend(params)


if __name__ == "__main__":
    IMAGE_TEST = "lena.jpg"
    METHODES_A_TESTER = ["mp", "omp", "stomp", "cosamp"]
    DOSSIER_SORTIE = "Data/Result"
    
    print(f"--- DÉMARRAGE DU TEST SUR {IMAGE_TEST} ---")
    
    try:
        # 1. On lance le calcul
        resultats = main(
            image_path=IMAGE_TEST,
            block_size=8,
            ratio=0.25,
            methodes=METHODES_A_TESTER,
            dictionary_type="dct",
            measurement_mode="gaussian",
            output_path=DOSSIER_SORTIE,
            method_params={
                "mp": {"max_iter": 50, "epsilon": 1e-6},
                "omp": {"max_iter": 50, "epsilon": 1e-6},
                "stomp": {"max_iter": 50, "epsilon": 1e-6, "t": 2.5},
                "cosamp": {"max_iter": 30, "epsilon": 1e-6, "s": 6}
            },
            seed=42
        )
        
        # 2. On affiche les résultats dans la console
        print("\n✅ Métriques :\n")
        for methode, metrics in resultats["metrics"].items():
            print(f"🔹 {methode.upper()} : PSNR = {metrics['psnr']:.2f} dB | Temps = {metrics['execution_time']:.2f} s")
            
        # 3. Appel de ta nouvelle fonction de sauvegarde
        save_results(resultats, DOSSIER_SORTIE)
            
    except Exception as e:
        print(f"\nUne erreur s'est produite : {e}")