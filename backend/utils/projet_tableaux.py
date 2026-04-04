"""
Tableaux du sujet PDF (section 6) : pourcentages P, cohérence mutuelle μ(Φ,D),
erreurs relatives sur des vecteurs de validation.

Aligné sur `fonction.md` (utils) et le cours 4 (Φ1…Φ4, définition de μ).
"""

from __future__ import annotations

import csv
import os
import time
from typing import Any, Sequence

import numpy as np

from backend.utils.Dictionnaire import build_dct_dictionary
from backend.utils.Methode import cosamp, irls, mp, omp, stomp
from backend.utils.mesure import (
    POURCENTAGES_MESURES_PROJET,
    compute_coherence_cours_phi_d,
    generate_measurement_matrix,
    liste_M_pour_pourcentages_projet,
    pourcentage_vers_M,
    resolve_measurement_mode,
)

# Ordre du sujet / cours
LIGNES_PHI_PROJET: tuple[tuple[str, str], ...] = (
    ("phi1", "uniform"),
    ("phi2", "bernoulli_1"),
    ("phi3", "bernoulli_01"),
    ("phi4", "gaussian"),
)

METHODES_SECTION_6: tuple[str, ...] = ("mp", "omp", "stomp", "cosamp", "irls")


def _signal_dct_parsimonieux(D0: np.ndarray, N: int, coeffs: dict[int, float]) -> np.ndarray:
    a = np.zeros(N, dtype=np.float64)
    for k, v in coeffs.items():
        idx = (N - 1) if k < 0 else k
        if 0 <= idx < N:
            a[idx] = v
    if not np.any(a):
        a[0] = 1.0
    return D0 @ a


def vecteurs_validation_projet(N: int) -> list[np.ndarray]:
    """
    Trois signaux de ℝ^N, parcimonieux en base DCT, reproductibles (sujet : 3 vecteurs de validation).
    """
    if N < 1:
        raise ValueError("N doit être >= 1.")
    D0 = build_dct_dictionary(N)
    specs: tuple[dict[int, float], ...] = (
        {0: 1.0, 2: 0.5, 5: -0.25},
        {1: 0.85, 7: 0.55, 13: -0.33},
        {3: -0.6, 11: 0.28, -1: 0.45},
    )
    return [_signal_dct_parsimonieux(D0, N, c) for c in specs]


def vecteur_validation_reference(N: int) -> np.ndarray:
    """Premier vecteur de :func:`vecteurs_validation_projet` (compatibilité)."""
    return vecteurs_validation_projet(N)[0]


def tableau_M_pour_pourcentages(N: int) -> dict[int, int]:
    """P ∈ {15,…,75} → M = ceil(P N / 100) (énoncé projet)."""
    return liste_M_pour_pourcentages_projet(N)


def tableau_coherence_mutuelle(
    D: np.ndarray,
    N: int,
    *,
    seed: int = 0,
    pourcentages: Sequence[int] | None = None,
) -> dict[str, Any]:
    """
    Remplit le tableau « Cohérence mutuelle » : lignes Φ1…Φ4, colonnes p = 15,…,75.

    Retourne un dict avec clés ``lignes`` (liste de dicts pour CSV), ``matrice`` (phi_label -> {P: μ}).
    """
    D = np.asarray(D, dtype=np.float64)
    if D.shape[0] != N:
        raise ValueError(f"D doit être (N, K) avec N={N}, obtenu {D.shape}.")

    ps = list(pourcentages) if pourcentages is not None else list(POURCENTAGES_MESURES_PROJET)
    matrice: dict[str, dict[int, float]] = {}
    lignes_csv: list[dict[str, Any]] = []

    for phi_idx, (phi_label, mode_interne) in enumerate(LIGNES_PHI_PROJET):
        matrice[phi_label] = {}
        row: dict[str, Any] = {"Phi": phi_label, "mode": resolve_measurement_mode(phi_label)}
        for P in ps:
            M = pourcentage_vers_M(P, N)
            subseed = int(seed) + phi_idx * 97_000 + int(P)
            Phi = generate_measurement_matrix(0.0, N, mode_interne, seed=subseed, M=M)
            mu = compute_coherence_cours_phi_d(Phi, D)
            matrice[phi_label][int(P)] = mu
            row[f"P_{P}"] = round(mu, 6)
        lignes_csv.append(row)

    return {"pourcentages": ps, "matrice": matrice, "lignes": lignes_csv}


def _erreur_relative(x: np.ndarray, x_hat: np.ndarray) -> float:
    n = float(np.linalg.norm(x))
    if n < 1e-15:
        return float("inf")
    return float(np.linalg.norm(x - x_hat) / n)


def _resoudre_alpha(
    A: np.ndarray,
    y: np.ndarray,
    methode: str,
    *,
    max_iter: int,
    epsilon: float,
    stomp_t: float,
    cosamp_s: int,
    irls_p: float,
) -> np.ndarray:
    if methode == "mp":
        return mp(A, y, max_iter=max_iter, epsilon=epsilon)
    if methode == "omp":
        return omp(A, y, max_iter=max_iter, epsilon=epsilon)
    if methode == "stomp":
        return stomp(A, y, max_iter=max_iter, eps=epsilon, t=stomp_t)
    if methode == "cosamp":
        return cosamp(A, y, max_iter=max_iter, epsilon=epsilon, s=cosamp_s)
    if methode == "irls":
        return irls(A, y, p=irls_p, max_iter=max_iter, epsilon=epsilon)
    raise ValueError(f"méthode inconnue : {methode}")


def tableau_erreurs_relatives_vecteurs(
    vecteurs: Sequence[np.ndarray],
    D: np.ndarray,
    *,
    seed: int = 0,
    pourcentages: Sequence[int] | None = None,
    max_iter: int = 80,
    epsilon: float = 1e-6,
    stomp_t: float = 2.5,
    cosamp_s: int | None = None,
    irls_p: float = 0.5,
) -> dict[str, Any]:
    """
    Pour chaque vecteur x, chaque Φi, chaque P, chaque méthode : erreur relative ‖x-x̂‖/‖x‖
    avec x̂ = Dα et α solution de (ΦD)α ≈ Φx.
    """
    D = np.asarray(D, dtype=np.float64)
    N, K = D.shape
    ps = list(pourcentages) if pourcentages is not None else list(POURCENTAGES_MESURES_PROJET)
    s_cos = int(cosamp_s) if cosamp_s is not None else max(2, min(8, K // 4))

    par_vecteur: list[dict[str, Any]] = []

    for v_idx, x in enumerate(vecteurs):
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.shape[0] != N:
            raise ValueError(f"Vecteur {v_idx} : taille {x.shape[0]} != N={N}.")

        blocs: dict[str, dict[str, dict[int, float]]] = {}
        for phi_idx, (phi_label, mode_interne) in enumerate(LIGNES_PHI_PROJET):
            blocs[phi_label] = {}
            for meth in METHODES_SECTION_6:
                blocs[phi_label][meth] = {}
            for P in ps:
                M = pourcentage_vers_M(P, N)
                subseed = int(seed) + v_idx * 50_000 + phi_idx * 3_000 + int(P)
                Phi = generate_measurement_matrix(0.0, N, mode_interne, seed=subseed, M=M)
                A = Phi @ D
                y = Phi @ x
                for meth in METHODES_SECTION_6:
                    alpha = _resoudre_alpha(
                        A,
                        y,
                        meth,
                        max_iter=max_iter,
                        epsilon=epsilon,
                        stomp_t=stomp_t,
                        cosamp_s=s_cos,
                        irls_p=irls_p,
                    )
                    x_hat = D @ alpha
                    blocs[phi_label][meth][int(P)] = _erreur_relative(x, x_hat)

        par_vecteur.append({"indice": v_idx + 1, "details": blocs})

    return {"pourcentages": ps, "par_vecteur": par_vecteur}


def _ecrire_csv_coherence(chemin: str, data: dict[str, Any]) -> None:
    ps: list[int] = data["pourcentages"]
    fieldnames = ["Phi", "mode"] + [f"P_{p}" for p in ps]
    with open(chemin, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in data["lignes"]:
            w.writerow(row)


def _ecrire_csv_erreurs(chemin_csv: str, erreurs: dict[str, Any]) -> None:
    """Une ligne par (vecteur de test, Φ, méthode) ; colonnes P_15 …"""
    ps: list[int] = erreurs["pourcentages"]
    fieldnames = ["vecteur", "Phi", "methode"] + [f"P_{p}" for p in ps]
    with open(chemin_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for bloc in erreurs["par_vecteur"]:
            vid = int(bloc["indice"])
            for phi_label, par_meth in bloc["details"].items():
                for meth, vals in par_meth.items():
                    row: dict[str, Any] = {"vecteur": vid, "Phi": phi_label, "methode": meth}
                    for p in ps:
                        row[f"P_{p}"] = round(vals[int(p)], 6)
                    w.writerow(row)


def exporter_tableaux_section6(
    D: np.ndarray,
    N: int,
    output_dir: str = "Data/Result",
    *,
    seed: int = 0,
    avec_erreurs_relatives: bool = True,
    max_iter: int = 80,
) -> str:
    """
    Crée un dossier horodaté (jj.mm.hh.mm) sous ``output_dir`` avec un sous-dossier ``Graph`` contenant :
    - ``M_pour_P.csv`` : P → M
    - ``coherence_mutuelle.csv``
    - ``erreurs_relatives.csv`` si demandé (une ligne par vecteur de test 1…3, Φ et méthode)

    Retourne le chemin du dossier ``Graph`` (là où sont les CSV).
    """
    horodatage = time.strftime("%d.%m.%H.%M")
    dossier = os.path.join(output_dir, horodatage, "Graph")
    os.makedirs(dossier, exist_ok=True)

    dico_M = tableau_M_pour_pourcentages(N)
    with open(os.path.join(dossier, "M_pour_P.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["P_pourcent", "M"])
        for p in sorted(dico_M.keys()):
            w.writerow([p, dico_M[p]])

    coh = tableau_coherence_mutuelle(D, N, seed=seed)
    _ecrire_csv_coherence(os.path.join(dossier, "coherence_mutuelle.csv"), coh)

    if avec_erreurs_relatives:
        vecs = vecteurs_validation_projet(N)
        err = tableau_erreurs_relatives_vecteurs(
            vecs, D, seed=seed, max_iter=max_iter,
        )
        _ecrire_csv_erreurs(os.path.join(dossier, "erreurs_relatives.csv"), err)

    print(f"Tableaux exportés dans :\n -> {dossier}")
    # Dossier contenant directement les CSV (sous-dossier « Graph » horodaté).
    return dossier


if __name__ == "__main__":
    # Exemple : dictionnaire DCT complet, N = B² (ex. 8×8)
    N_ex = 64
    D_ex = build_dct_dictionary(N_ex)
    exporter_tableaux_section6(D_ex, N_ex, seed=42)
