"""
Point d'entrée du projet.
"""

from __future__ import annotations

import math
from typing import Any

from backend.main_backend import main_backend


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
    resultats = main(
        image_path="lena.jpg",
        block_size=8,
        ratio=0.25,
        methodes=["omp", "cosamp"],
        dictionary_type="dct",
        measurement_mode="gaussian",
        output_path="Data/Result",
        method_params={
            "omp": {"max_iter": 50, "epsilon": 1e-6},
            "cosamp": {"max_iter": 30, "epsilon": 1e-6, "s": 8},
        },
    )
    print(resultats["metrics"])