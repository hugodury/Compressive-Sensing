"""
Estimation de l’espace disque « côté acquisition compressive » vs image raster.

Compare la taille d’une image niveaux de gris 8 bits (raster recadrée) à un modèle de
stockage minimal des **mesures y** (float32 par coefficient) + **une matrice Φ** M×N
partagée par tous les patchs (N = B²). C’est un ordre de grandeur pédagogique : en
pratique on ajouterait en-têtes, dictionnaire D, métadonnées, etc.
"""

from __future__ import annotations

import os
from typing import Any


def estimer_stockage_bcs(
    h: int,
    w: int,
    nb_patches: int,
    B: int,
    M: int,
    *,
    chemin_fichier_source: str | None = None,
    bytes_par_pixel_raster: int = 1,
    bytes_par_float_mesure: int = 4,
    bytes_par_float_phi: int = 4,
) -> dict[str, Any]:
    """
    :param h, w: dimensions de l’image recadrée utilisée pour la reco.
    :param nb_patches: nombre de blocs B×B traités.
    :param B: côté du patch.
    :param M: nombre de mesures par patch (lignes de Φ).
    """
    N = int(B) * int(B)
    h_i, w_i = int(h), int(w)
    nb = int(nb_patches)
    M_i = int(M)

    octets_raster = max(0, h_i * w_i * bytes_par_pixel_raster)
    octets_y = nb * M_i * bytes_par_float_mesure
    octets_phi = M_i * N * bytes_par_float_phi
    octets_modele = octets_y + octets_phi

    fichier_octets: int | None = None
    if chemin_fichier_source:
        try:
            fichier_octets = int(os.path.getsize(chemin_fichier_source))
        except OSError:
            fichier_octets = None

    # « Avant » = taille réelle du fichier sur disque si on en a un ; sinon raster recadré 8 bits
    avant_fichier_sur_disque = bool(fichier_octets is not None and fichier_octets > 0)
    if avant_fichier_sur_disque:
        octets_avant_principal = int(fichier_octets)  # type: ignore[arg-type]
    else:
        octets_avant_principal = octets_raster

    gain = octets_avant_principal - octets_modele
    taux_reduction_pct = (
        100.0 * (1.0 - octets_modele / octets_avant_principal) if octets_avant_principal > 0 else 0.0
    )

    mib = lambda x: x / (1024.0 * 1024.0)

    if octets_modele <= octets_avant_principal:
        comparaison = (
            f"Modèle mesures+Φ ≈ {mib(octets_modele):.4f} MiB vs avant (fichier ou raster) ≈ {mib(octets_avant_principal):.4f} MiB "
            f"→ réduction indicative ~ {taux_reduction_pct:.1f} %."
        )
    else:
        comparaison = (
            f"Modèle mesures+Φ ≈ {mib(octets_modele):.4f} MiB dépasse l’image avant ≈ {mib(octets_avant_principal):.4f} MiB "
            f"(M ou float32 élevés) : le gain stockage n’apparaît pas avec ce décompte simplifié."
        )

    msg = (
        f"1) Avant compression — "
        + (
            f"fichier image sur disque : ≈ {mib(octets_avant_principal):.4f} MiB ({octets_avant_principal} o)."
            if avant_fichier_sur_disque
            else f"pas de fichier — raster recadré {h_i}×{w_i} uint8 : ≈ {mib(octets_raster):.4f} MiB ({octets_raster} o)."
        )
        + f" 2) Après compression (théorique, données à stocker : y float32 + Φ) : ≈ {mib(octets_modele):.4f} MiB. "
        f"Détail : {nb} patchs × M={M_i} → y ≈ {mib(octets_y):.4f} MiB ; Φ {M_i}×{N} ≈ {mib(octets_phi):.4f} MiB. {comparaison}"
    )
    if avant_fichier_sur_disque and octets_raster != octets_avant_principal:
        msg += f" Raster recadré en mémoire : ≈ {mib(octets_raster):.4f} MiB (peut différer du fichier si JPEG/PNG)."

    # Synthèse avant (fichier/raster) → après (compressé) en MiB / ko
    kio = 1024.0
    avant_mib = mib(octets_avant_principal)
    apres_mib = mib(octets_modele)
    gain_mib = gain / (1024.0 * 1024.0)
    avant_ko = octets_avant_principal / kio
    apres_ko = octets_modele / kio
    gain_ko = gain / kio

    msg += (
        f" Synthèse fichier→compressé : {avant_mib:.4f} MiB ({avant_ko:.1f} ko) → {apres_mib:.4f} MiB ({apres_ko:.1f} ko) "
        f"= gain {gain_mib:+.6f} MiB ({gain_ko:+.1f} ko)."
    )

    return {
        "h": h_i,
        "w": w_i,
        "nb_patches": nb,
        "B": int(B),
        "N_patch": N,
        "M": M_i,
        "octets_raster_u8": octets_raster,
        "octets_fichier_source": fichier_octets,
        "avant_est_taille_fichier_disque": avant_fichier_sur_disque,
        "octets_avant_fichier_ou_raster": octets_avant_principal,
        "octets_reference_pour_gain": octets_avant_principal,
        "octets_mesures_y_float32": octets_y,
        "octets_phi_float32": octets_phi,
        "octets_modele_mesures_plus_phi": octets_modele,
        "gain_octets": gain,
        "taux_reduction_vs_reference_pct": round(taux_reduction_pct, 3),
        "avant_compression_mib": round(avant_mib, 6),
        "apres_compression_mib": round(apres_mib, 6),
        "gain_mib": round(gain_mib, 6),
        "avant_compression_ko": round(avant_ko, 2),
        "apres_compression_ko": round(apres_ko, 2),
        "gain_ko": round(gain_ko, 2),
        "octets_apres_export_total": None,
        "chemin_dossier_export": None,
        "apres_export_mib": None,
        "apres_export_ko": None,
        "message": msg,
    }


def enrichir_stockage_apres_export(
    stk: dict[str, Any],
    dossier_export: str,
    octets_total_dossier: int,
) -> dict[str, Any]:
    """
    Après ``save_results`` : ajoute la taille réelle de tout le dossier d’export (PNG, CSV, etc.).
    """
    out = dict(stk)
    kio = 1024.0
    mib = octets_total_dossier / (kio * kio)
    ko = octets_total_dossier / kio
    out["octets_apres_export_total"] = int(octets_total_dossier)
    out["chemin_dossier_export"] = dossier_export
    out["apres_export_mib"] = round(mib, 6)
    out["apres_export_ko"] = round(ko, 2)
    base = str(out.get("message", "")).rstrip()
    out["message"] = (
        base
        + "\n\n— Annexe (hors comparaison compression) : dossier d’export après reconstruction « "
        + dossier_export
        + f" » ≈ {mib:.4f} MiB ({ko:.1f} ko) au total (PNG pleins, CSV, textes). "
        "Ne pas confondre avec « après compression » : ce volume inclut les images reconstruites, pas seulement les mesures y."
    )
    return out


def stockage_dict_pour_sauvegarde(d: dict[str, Any]) -> str:
    """Texte multi-lignes pour fichier côté disque."""
    lines = [
        d.get("message", ""),
        "",
        "Détail (octets) :",
        f"  raster_u8: {d.get('octets_raster_u8')}",
    ]
    if d.get("octets_fichier_source") is not None:
        lines.append(f"  fichier_source: {d.get('octets_fichier_source')}")
    lines += [
        f"  mesures_y_float32: {d.get('octets_mesures_y_float32')}",
        f"  phi_float32: {d.get('octets_phi_float32')}",
        f"  total_modele: {d.get('octets_modele_mesures_plus_phi')}",
        f"  reference_pour_gain: {d.get('octets_reference_pour_gain')}",
        f"  gain_octets: {d.get('gain_octets')}",
        f"  taux_reduction_vs_reference_pct: {d.get('taux_reduction_vs_reference_pct')}",
        "",
        "Synthèse — avant = fichier sur disque (sinon raster recadré), après = données compressées théoriques :",
        f"  avant_est_fichier_disque: {d.get('avant_est_taille_fichier_disque')}",
        f"  octets_avant_fichier_ou_raster: {d.get('octets_avant_fichier_ou_raster')}",
        f"  avant_compression_MiB: {d.get('avant_compression_mib')}",
        f"  avant_compression_ko: {d.get('avant_compression_ko')}",
        f"  apres_compression_MiB (y+Phi): {d.get('apres_compression_mib')}",
        f"  apres_compression_ko: {d.get('apres_compression_ko')}",
        f"  gain_MiB (avant fichier - compressé théorique): {d.get('gain_mib')}",
        f"  gain_ko: {d.get('gain_ko')}",
    ]
    if d.get("octets_apres_export_total") is not None:
        lines += [
            "",
            "Annexe — dossier export (reconstruction + PNG ; pas la taille « après compression seule ») :",
            f"  chemin_dossier_export: {d.get('chemin_dossier_export')}",
            f"  octets_apres_export_total: {d.get('octets_apres_export_total')}",
            f"  apres_export_MiB: {d.get('apres_export_mib')}",
            f"  apres_export_ko: {d.get('apres_export_ko')}",
        ]
    return "\n".join(lines) + "\n"
