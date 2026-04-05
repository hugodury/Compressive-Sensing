"""
Estimation grossière de l’empreinte carbone d’une exécution locale (projet PDF : sensibilisation).

Ce n’est pas une mesure au watt-près : on part du temps CPU/machine et d’hypothèses documentées
(puissance moyenne du PC, intensité carbone du réseau électrique). L’objectif est l’ordre de grandeur
et la comparaison entre runs (plus court / moins de méthodes = moins de consommation probable).
"""

from __future__ import annotations

import resource
import sys
import time
from dataclasses import dataclass
from typing import Any


# Ordres de grandeur indicatifs (à ajuster dans le rapport selon ta machine / pays)
# Intensité : g CO2eq par kWh — mix France ~ 60–100 ; Europe ~ 300 ; monde ~ 450
INTENSITE_FRANCE_INDICATIVE_G_KWH = 85.0
# PC portable sous charge numérique modérée (W) — très variable selon modèle
PUISSANCE_MOYENNE_PC_W = 45.0


@dataclass(frozen=True)
class EstimationEmpreinte:
    duree_wall_s: float
    duree_cpu_process_s: float | None
    energie_estimee_wh: float
    co2e_g_estime: float
    hypothese_puissance_w: float
    hypothese_g_co2_par_kwh: float
    message: str
    # Même P et g/kWh, mais durée = temps CPU processus (borne basse si la machine ne consomme qu’au travail utile)
    energie_wh_temps_cpu: float | None = None
    co2e_g_estime_temps_cpu: float | None = None


def estimer_empreinte(
    duree_secondes: float,
    *,
    puissance_w: float = PUISSANCE_MOYENNE_PC_W,
    intensite_g_co2_par_kwh: float = INTENSITE_FRANCE_INDICATIVE_G_KWH,
    duree_cpu_process_s: float | None = None,
    contexte: str = "",
) -> EstimationEmpreinte:
    """
    Énergie (Wh) ≈ (puissance_w / 1000) * (heures), puis CO2eq (g) ≈ kWh * intensité.
    """
    h = max(0.0, float(duree_secondes)) / 3600.0
    energie_wh = float(puissance_w) * h
    energie_kwh = energie_wh / 1000.0
    co2_g = energie_kwh * float(intensite_g_co2_par_kwh)

    energie_wh_cpu: float | None = None
    co2_cpu: float | None = None
    if duree_cpu_process_s is not None and float(duree_cpu_process_s) > 0:
        h_cpu = float(duree_cpu_process_s) / 3600.0
        energie_wh_cpu = float(puissance_w) * h_cpu
        co2_cpu = (energie_wh_cpu / 1000.0) * float(intensite_g_co2_par_kwh)

    ctx = f" ({contexte})" if contexte else ""
    cpu_txt = ""
    if duree_cpu_process_s is not None:
        cpu_txt = f" Temps CPU processus ~ {duree_cpu_process_s:.2f} s."

    msg = (
        f"Empreinte estimée{ctx} : ~ {co2_g:.4f} g CO2eq "
        f"(≈ {energie_kwh:.6f} kWh, hypothèses {puissance_w:.0f} W, {intensite_g_co2_par_kwh:.0f} g/kWh) "
        f"— basé sur le temps mural (souvent un majorant si le CPU n’est pas à pleine charge tout du long)."
        f"{cpu_txt} "
    )
    if co2_cpu is not None:
        msg += (
            f" Borne basse indicative (mêmes W et g/kWh × temps CPU seulement) : ~ {co2_cpu:.4f} g CO2eq. "
            f"La réalité est souvent entre ces deux ordres de grandeur. "
        )
    msg += "Valeur indicative — voir EMPREINTE.md."

    return EstimationEmpreinte(
        duree_wall_s=float(duree_secondes),
        duree_cpu_process_s=duree_cpu_process_s,
        energie_estimee_wh=energie_wh,
        co2e_g_estime=co2_g,
        hypothese_puissance_w=puissance_w,
        hypothese_g_co2_par_kwh=intensite_g_co2_par_kwh,
        message=msg,
        energie_wh_temps_cpu=energie_wh_cpu,
        co2e_g_estime_temps_cpu=co2_cpu,
    )


def cpu_process_delta_depuis(start_rusage: Any) -> float | None:
    """Delta utime+stime depuis un snapshot getrusage (Unix)."""
    if start_rusage is None:
        return None
    try:
        cur = resource.getrusage(resource.RUSAGE_SELF)
        return float(cur.ru_utime + cur.ru_stime - start_rusage.ru_utime - start_rusage.ru_stime)
    except (ValueError, OSError, AttributeError):
        return None


def fusionner_empreinte_dans_resultat(
    resultat: dict[str, Any],
    params: dict[str, Any],
    *,
    t_wall_debut: float,
    rusage_debut: Any,
    contexte: str = "main_backend",
) -> None:
    """Ajoute la clé ``empreinte`` et affiche sur stderr si demandé."""
    if not params.get("empreinte_carbone", True):
        return
    wall = time.perf_counter() - t_wall_debut
    cpu = cpu_process_delta_depuis(rusage_debut)
    puissance = float(params.get("empreinte_puissance_w", PUISSANCE_MOYENNE_PC_W))
    intensite = float(params.get("empreinte_g_co2_par_kwh", INTENSITE_FRANCE_INDICATIVE_G_KWH))
    est = estimer_empreinte(
        wall,
        puissance_w=puissance,
        intensite_g_co2_par_kwh=intensite,
        duree_cpu_process_s=cpu,
        contexte=contexte,
    )
    resultat["empreinte"] = estimation_dict(est)
    afficher_si_demande(est, actif=bool(params.get("empreinte_afficher_console", True)))


class ChronoEmpreinte:
    """Mesure le mur + le CPU processus entre __enter__ et __exit__."""

    def __init__(self) -> None:
        self._t0 = 0.0
        self._ru0: Any = None

    def __enter__(self) -> ChronoEmpreinte:
        self._t0 = time.perf_counter()
        try:
            self._ru0 = resource.getrusage(resource.RUSAGE_SELF)
        except (OSError, AttributeError):
            self._ru0 = None
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def duree_wall_s(self) -> float:
        return float(time.perf_counter() - self._t0)

    def rapport(
        self,
        *,
        puissance_w: float = PUISSANCE_MOYENNE_PC_W,
        intensite_g_co2_par_kwh: float = INTENSITE_FRANCE_INDICATIVE_G_KWH,
        contexte: str = "",
    ) -> EstimationEmpreinte:
        wall = self.duree_wall_s()
        cpu = cpu_process_delta_depuis(self._ru0)
        return estimer_empreinte(
            wall,
            puissance_w=puissance_w,
            intensite_g_co2_par_kwh=intensite_g_co2_par_kwh,
            duree_cpu_process_s=cpu,
            contexte=contexte,
        )


def estimation_dict(est: EstimationEmpreinte) -> dict[str, Any]:
    d: dict[str, Any] = {
        "duree_wall_s": est.duree_wall_s,
        "duree_cpu_process_s": est.duree_cpu_process_s,
        "energie_estimee_wh": est.energie_estimee_wh,
        "co2e_g_estime": est.co2e_g_estime,
        "hypothese_puissance_w": est.hypothese_puissance_w,
        "hypothese_g_co2_par_kwh": est.hypothese_g_co2_par_kwh,
        "message": est.message,
    }
    if est.energie_wh_temps_cpu is not None:
        d["energie_wh_temps_cpu"] = est.energie_wh_temps_cpu
    if est.co2e_g_estime_temps_cpu is not None:
        d["co2e_g_estime_temps_cpu"] = est.co2e_g_estime_temps_cpu
    return d


def afficher_si_demande(est: EstimationEmpreinte, *, actif: bool) -> None:
    if actif:
        print(est.message, file=sys.stderr)
