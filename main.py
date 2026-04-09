"""
Fichier d'entrée du projet.

- `main()` : lance une reconstruction unique.
- `run_pipeline()` : enchaîne des étapes (reconstruct/save/tableaux/graphique).

Objectif de cette version : garder les mêmes fonctionnalités,
mais avec une structure plus simple à lire.
"""

from __future__ import annotations

import argparse
import copy
import math
import resource
import sys
import time
from collections.abc import Sequence
from typing import Any

from backend.utils.empreinte import (
    afficher_si_demande,
    estimation_dict,
    estimer_empreinte,
    cpu_process_delta_depuis,
)

from backend.main_backend import main_backend
from backend.utils.save import save_results


def _ratio_vers_nb_mesures_par_patch(ratio: float, n: int) -> int:
    """Convertit un ratio (fraction ou %) en M (nombre de mesures)."""
    rf = float(ratio)
    if rf <= 0.0:
        raise ValueError("ratio doit être > 0.")
    if rf > 100.0:
        raise ValueError(
            "ratio : utiliser une fraction dans ]0, 1] (ex. 0.25) ou un pourcentage dans ]0, 100] (ex. 25)."
        )
    if rf <= 1.0:
        return max(1, math.ceil(rf * n))
    return max(1, math.ceil((rf / 100.0) * n))


def _normaliser_methodes(methodes: str | list[str]) -> list[str]:
    """Accepte une méthode str ou une liste, et renvoie toujours une liste."""
    if isinstance(methodes, str):
        return [methodes]
    return list(methodes)


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
    dictionary_train_image_path: str | None = None,
    method_params: dict[str, dict[str, Any]] | None = None,
    patch_params: dict[str, Any] | None = None,
    seed: int | None = None,
    empreinte_carbone: bool = True,
    empreinte_afficher_console: bool = True,
    empreinte_puissance_w: float = 45.0,
    empreinte_g_co2_par_kwh: float = 85.0,
    post_filter: bool = False,
    post_filter_d: int = 5,
    post_filter_sigma_color: float = 75.0,
    post_filter_sigma_space: float = 75.0,
) -> dict[str, Any]:
    """Prépare le dictionnaire de paramètres pour le backend."""
    if block_size <= 0:
        raise ValueError("block_size doit être > 0.")

    n = block_size * block_size
    m = _ratio_vers_nb_mesures_par_patch(ratio, n)
    methodes_norm = _normaliser_methodes(methodes)

    if n_atoms is None:
        n_atoms = n

    return {
        "image_path": image_path,
        "B": block_size,
        "ratio": ratio,
        "N": n,
        "M": m,
        "methodes": methodes_norm,
        "dictionary_type": dictionary_type,
        "measurement_mode": measurement_mode,
        "output_path": output_path,
        "n_atoms": n_atoms,
        "n_iter_ksvd": n_iter_ksvd,
        "dictionary_train_image_path": dictionary_train_image_path,
        "method_params": method_params or {},
        "patch_params": patch_params or {},
        "seed": seed,
        "empreinte_carbone": empreinte_carbone,
        "empreinte_afficher_console": empreinte_afficher_console,
        "empreinte_puissance_w": empreinte_puissance_w,
        "empreinte_g_co2_par_kwh": empreinte_g_co2_par_kwh,
        "post_filter": post_filter,
        "post_filter_d": post_filter_d,
        "post_filter_sigma_color": post_filter_sigma_color,
        "post_filter_sigma_space": post_filter_sigma_space,
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
    dictionary_train_image_path: str | None = None,
    method_params: dict[str, dict[str, Any]] | None = None,
    patch_params: dict[str, Any] | None = None,
    seed: int | None = None,
    empreinte_carbone: bool = True,
    empreinte_afficher_console: bool = True,
    empreinte_puissance_w: float = 45.0,
    empreinte_g_co2_par_kwh: float = 85.0,
    post_filter: bool = False,
    post_filter_d: int = 5,
    post_filter_sigma_color: float = 75.0,
    post_filter_sigma_space: float = 75.0,
) -> dict[str, Any]:
    """Fonction principale : prépare les paramètres puis lance le backend."""
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
        dictionary_train_image_path=dictionary_train_image_path,
        method_params=method_params,
        patch_params=patch_params,
        seed=seed,
        empreinte_carbone=empreinte_carbone,
        empreinte_afficher_console=empreinte_afficher_console,
        empreinte_puissance_w=empreinte_puissance_w,
        empreinte_g_co2_par_kwh=empreinte_g_co2_par_kwh,
        post_filter=post_filter,
        post_filter_d=post_filter_d,
        post_filter_sigma_color=post_filter_sigma_color,
        post_filter_sigma_space=post_filter_sigma_space,
    )
    return main_backend(params)


ALL_BACKEND_METHODS: tuple[str, ...] = ("mp", "omp", "stomp", "cosamp", "irls", "bp", "lp", "lasso")


def run_coarse_best_search(
    base_params: dict[str, Any],
    *,
    ratios: Sequence[float] | None = None,
    measurement_modes: Sequence[str] | None = None,
    max_patches_cap: int = 72,
) -> dict[str, Any]:
    """
    Explore plusieurs couples (ratio de mesures, Φ) avec toutes les méthodes, en limitant le nombre de patchs
    pour garder un temps raisonnable. Retourne la meilleure combinaison au sens du PSNR (sur les patchs traités).

    Exige un découpage classique par B (pas de grille nrows×ncols dans ``patch_params``).
    """
    if ratios is None:
        ratios = (20.0, 35.0, 50.0)
    if measurement_modes is None:
        measurement_modes = ("phi1", "phi2", "phi3", "phi4")

    pp_base = copy.deepcopy(base_params.get("patch_params") or {})
    if pp_base.get("nrows") is not None or pp_base.get("ncols") is not None:
        raise ValueError("Balayage automatique : désactivez la grille nrows×ncols et utilisez la taille de bloc B.")

    trials: list[dict[str, Any]] = []
    best_psnr = float("-inf")
    best: dict[str, Any] | None = None

    for phi in measurement_modes:
        for ratio in ratios:
            # Paramètres de base pour un essai (phi, ratio), avec toutes les méthodes.
            p = setupParam(
                image_path=str(base_params["image_path"]),
                block_size=int(base_params["B"]),
                ratio=float(ratio),
                methodes=list(ALL_BACKEND_METHODS),
                dictionary_type=str(base_params["dictionary_type"]),
                measurement_mode=str(phi),
                output_path=str(base_params.get("output_path") or "Data/Result"),
                n_atoms=base_params.get("n_atoms"),
                n_iter_ksvd=int(base_params.get("n_iter_ksvd", 0) or 0),
                dictionary_train_image_path=base_params.get("dictionary_train_image_path"),
                method_params=copy.deepcopy(base_params.get("method_params") or {}),
                patch_params={},
                seed=base_params.get("seed"),
                empreinte_carbone=False,
                empreinte_afficher_console=False,
                empreinte_puissance_w=float(base_params.get("empreinte_puissance_w", 45.0)),
                empreinte_g_co2_par_kwh=float(base_params.get("empreinte_g_co2_par_kwh", 85.0)),
            )
            # On limite le nombre de patchs pour garder un balayage raisonnable en temps.
            pp = copy.deepcopy(pp_base)
            pp["max_patches"] = int(max_patches_cap)
            pp["mode_phi"] = phi
            pp.pop("M", None)
            p["patch_params"] = pp

            out = main_backend(p)
            for meth, met in out.get("metrics", {}).items():
                ps = float(met.get("psnr", float("-inf")))
                row = {
                    "measurement_mode": phi,
                    "ratio": float(ratio),
                    "method": meth,
                    "psnr": ps,
                    "mse": float(met.get("mse", 0.0)),
                }
                trials.append(row)
                if ps > best_psnr:
                    best_psnr = ps
                    best = row.copy()

    return {"best": best, "trials": trials, "nb_evaluations": len(trials)}


def run_pipeline(
    params: dict[str, Any],
    *,
    etapes: Sequence[str] | str = ("reconstruct",),
    sweep_ratios: Sequence[float] | None = None,
    tableaux_avec_erreurs: bool = True,
    tableaux_max_iter: int = 64,
) -> dict[str, Any]:
    """
    Enchaîne des étapes sans IHM : reconstruct → save_results → CSV section 6 → courbe PSNR.

    etapes : tuple ou chaîne "reconstruct,save,tableaux_s6,sweep_graph".
    sweep_ratios : pourcentages ou fractions comme ailleurs (ex. 15, 25 ou 0.25, 0.5).
    """
    # L'utilisateur peut passer "reconstruct,save" ou ["reconstruct", "save"].
    if isinstance(etapes, str):
        etapes_list = [e.strip() for e in etapes.split(",") if e.strip()]
    else:
        etapes_list = [str(e).strip() for e in etapes]
    etapes_set = frozenset(etapes_list)
    sortie: dict[str, Any] = {}
    out_path = str(params.get("output_path") or "Data/Result")

    t_pipe = time.perf_counter()
    ru_pipe = None
    try:
        ru_pipe = resource.getrusage(resource.RUSAGE_SELF)
    except (OSError, AttributeError):
        pass

    if "reconstruct" in etapes_set:
        p_run = copy.deepcopy(params)
        p_run["empreinte_afficher_console"] = False
        sortie["reconstruction"] = main_backend(p_run)

    if "save" in etapes_set:
        if "reconstruction" not in sortie:
            raise ValueError("Étape save sans reconstruct : lance d’abord reconstruct.")
        save_results(sortie["reconstruction"], out_path)

    if "tableaux_s6" in etapes_set:
        from backend.utils.Dictionnaire import build_dct_dictionary
        from backend.utils.projet_tableaux import exporter_tableaux_section6

        B = int(params["B"])
        N = B * B
        na = params.get("n_atoms")
        K = min(int(na) if na is not None else N, N)
        D_tab = build_dct_dictionary(N)[:, :K]
        sd = params.get("seed")
        seed_i = int(sd) if sd is not None else 0
        dossier = exporter_tableaux_section6(
            D_tab,
            N,
            output_dir=out_path,
            seed=seed_i,
            avec_erreurs_relatives=tableaux_avec_erreurs,
            max_iter=tableaux_max_iter,
        )
        sortie["dossier_tableaux_section6"] = dossier

    if "sweep_graph" in etapes_set:
        if not sweep_ratios:
            raise ValueError("sweep_graph : fournir sweep_ratios=[...] (ex. [15, 25, 50]).")
        from backend.utils.graphiques_projet import exporter_sweep_graphique

        sortie["sweep_graphique"] = exporter_sweep_graphique(
            params, list(sweep_ratios), output_path=out_path
        )

    # Estimation d'empreinte sur toute la session.
    if params.get("empreinte_carbone", True):
        wall = time.perf_counter() - t_pipe
        cpu = cpu_process_delta_depuis(ru_pipe)
        est = estimer_empreinte(
            wall,
            puissance_w=float(params.get("empreinte_puissance_w", 45.0)),
            intensite_g_co2_par_kwh=float(params.get("empreinte_g_co2_par_kwh", 85.0)),
            duree_cpu_process_s=cpu,
            contexte="run_pipeline (session)",
        )
        sortie["empreinte_session"] = estimation_dict(est)
        afficher_si_demande(
            est,
            actif=bool(params.get("empreinte_afficher_console", True)),
        )

    return sortie


def _parse_sweep_ratios(s: str) -> list[float]:
    """Parse une chaîne CSV de ratios (ex. '15,25,50') vers une liste de float."""
    out: list[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reconstruction BCS — une reco ou un enchaînement d’étapes.",
    )
    parser.add_argument("-i", "--image", default="lena.jpg", help="Chemin image niveaux de gris")
    parser.add_argument(
        "--etapes",
        default="reconstruct,save",
        help="reconstruct, save, tableaux_s6, sweep_graph (virgules)",
    )
    parser.add_argument("--sweep-ratios", default="15,25,50", help="Pour sweep_graph uniquement")
    parser.add_argument("--no-tableaux-erreurs", action="store_true", help="CSV §6 sans erreurs relatives (plus rapide)")
    parser.add_argument("--max-patches", type=int, default=None, help="Limite de patchs (tests rapides)")
    args, _unknown = parser.parse_known_args()

    # Réglages CLI par défaut (identiques à avant).
    IMAGE_TEST = args.image
    DOSSIER_SORTIE = "Data/Result"
    METHODES_A_TESTER = ["mp", "omp", "stomp", "cosamp"]

    patch_extra: dict[str, Any] = {}
    if args.max_patches is not None:
        patch_extra["max_patches"] = args.max_patches

    params = setupParam(
        image_path=IMAGE_TEST,
        block_size= 8,
        ratio=0.25,
        methodes=METHODES_A_TESTER,
        dictionary_type="dct",
        measurement_mode="gaussian",
        output_path=DOSSIER_SORTIE,
        method_params={
            "mp": {"max_iter": 50, "epsilon": 1e-6},
            "omp": {"max_iter": 50, "epsilon": 1e-6},
            "stomp": {"max_iter": 50, "epsilon": 1e-6, "t": 2.5},
            "cosamp": {"max_iter": 30, "epsilon": 1e-6, "s": 6},
        },
        patch_params=patch_extra or {},
        seed=42,
    )

    try:
        etapes_tokens = [x.strip() for x in args.etapes.split(",") if x.strip()]
        ratios = _parse_sweep_ratios(args.sweep_ratios)
        tout = run_pipeline(
            params,
            etapes=args.etapes,
            sweep_ratios=ratios if "sweep_graph" in etapes_tokens else None,
            tableaux_avec_erreurs=not args.no_tableaux_erreurs,
        )
        # Affichage console compact des résultats de reconstruction.
        if "reconstruction" in tout:
            r = tout["reconstruction"]
            print(f"--- OK : {IMAGE_TEST} ---")
            for methode, metrics in r["metrics"].items():
                tps = metrics.get("execution_time", 0.0)
                ps = metrics.get("psnr", 0.0)
                mse = metrics.get("mse", 0.0)
                rel = metrics.get("relative_error", 0.0)
                ps_str = f"{ps:.2f}" if math.isfinite(float(ps)) else "inf"
                ligne = (
                    f"{methode.upper()}  PSNR={ps_str} dB  MSE={float(mse):.6g}  "
                    f"err_rel={float(rel):.4f}  temps={tps:.2f}s"
                )
                if metrics.get("cosamp_s_mode"):
                    ligne += f"  CoSaMP s={metrics.get('s_cosamp_utilise')} ({metrics['cosamp_s_mode']})"
                print(ligne)
        if "dossier_tableaux_section6" in tout:
            print("Tableaux §6 ->", tout["dossier_tableaux_section6"])
        if "sweep_graphique" in tout:
            print("Courbe PSNR ->", tout["sweep_graphique"].get("graphique_psnr_png"))
    except FileNotFoundError as e:
        print("Image introuvable :", e, file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print("Erreur :", e, file=sys.stderr)
        sys.exit(1)