from __future__ import annotations

import io
import os
import platform
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import numpy as np
from PIL import Image, ImageTk
from matplotlib.figure import Figure

# --- Quatre familles de matrices Φ (alias internes uniform, bernoulli_*, gaussian) ---
PHI_COURS_KEYS: tuple[str, ...] = ("phi1", "phi2", "phi3", "phi4")

PHI_COURS_DETAIL: dict[str, str] = {
    "phi1": "Uniforme U(0,1), colonnes normalisées 1/√M — Φ₁",
    "phi2": "Bernoulli {−1, +1}, 1/√M — Φ₂",
    "phi3": "Bernoulli {0, 1}, 1/√M — Φ₃",
    "phi4": "Gaussienne N(0,1), 1/√M — Φ₄",
}

# Libellés courts pour boutons radio
PHI_COURS_RADIO_LABELS: dict[str, str] = {
    "phi1": "Φ₁ — U(0,1)/√M",
    "phi2": "Φ₂ — Bernoulli ±1 / √M",
    "phi3": "Φ₃ — Bernoulli {0,1} / √M",
    "phi4": "Φ₄ — Gaussienne / √M",
}

# Libellés courts pour légendes (M, μ, Φ)
PHI_SHORT_LABEL: dict[str, str] = {"phi1": "Φ₁", "phi2": "Φ₂", "phi3": "Φ₃", "phi4": "Φ₄"}

PHI_MEASUREMENT_MODES: tuple[str, ...] = PHI_COURS_KEYS

# Toutes les clés reconnues par le backend (saisie manuelle / anciens projets)
DICTIONARY_TYPES_ALL: tuple[str, ...] = ("dct", "mixte", "ksvd", "ksvd_dct", "ksvd_mixte", "ksvd_random")

DICTIONARY_UI_LABELS: dict[str, str] = {
    "dct": "DCT seule — base 2D fixe (aucun apprentissage)",
    "mixte": "Mixte — moitié DCT + moitié colonnes tirées des patchs de l’image",
    "ksvd": "K-SVD — init. aléatoire (colonnes patchs), puis itérations",
    "ksvd_dct": "K-SVD — initialisation DCT tronquée puis itérations sur les patchs",
    "ksvd_mixte": "K-SVD — initialisation mixte (DCT+patchs) puis itérations sur les patchs",
    "ksvd_random": "K-SVD (init. aléatoire, alias)",
}

# Choix proposés dans l’IHM (alignés sur ``Tratement_Image`` : dct, mixte, ksvd, ksvd_dct, ksvd_mixte)
DICTIONARY_TYPES_UI: tuple[str, ...] = ("dct", "mixte", "ksvd", "ksvd_dct", "ksvd_mixte")

DICTIONARY_COMBO_TEXT: list[str] = [f"{k} — {DICTIONARY_UI_LABELS[k]}" for k in DICTIONARY_TYPES_UI]
DICTIONARY_COMBO_TO_KEY: dict[str, str] = {f"{k} — {DICTIONARY_UI_LABELS[k]}": k for k in DICTIONARY_TYPES_UI}


def dictionary_key_from_combo_selection(text: str) -> str:
    t = text.strip()
    return DICTIONARY_COMBO_TO_KEY.get(t, t.split(" — ", 1)[0].strip() if " — " in t else t)

# Huit méthodes du sujet — « IRLS » uniquement (le code accepte aussi irls_lp / irls_p comme alias identiques)
SOLVER_UI_CHOICES: tuple[tuple[str, str], ...] = (
    ("mp", "MP"),
    ("omp", "OMP"),
    ("stomp", "StOMP"),
    ("cosamp", "CoSaMP"),
    ("irls", "IRLS"),
    ("bp", "BP"),
    ("lp", "LP"),
    ("lasso", "LASSO"),
)

SOLVER_METHOD_IDS: tuple[str, ...] = tuple(k for k, _ in SOLVER_UI_CHOICES)
SOLVER_DISPLAY: dict[str, str] = dict(SOLVER_UI_CHOICES)


def solver_checkbox_caption(method_id: str) -> str:
    return SOLVER_DISPLAY.get(method_id, method_id.upper())


# Textes d’aide courts (IHM — pas de fichier supplémentaire)
UI_HELP_PHI_BLOC = (
    "Φ₁ à Φ₄ sont les quatre familles de matrices de mesure prévues dans ce projet. "
    "Chaque bouton correspond à une seule famille."
)

UI_HELP_RATIO_M = (
    "Ratio : part de mesures par patch (ex. 0,25 ou 25 pour 25 %). "
    "Si vous remplissez M, ce nombre de lignes de Φ prime sur le ratio."
)

UI_HELP_METHODS_BLOC = (
    "MP / OMP / StOMP / CoSaMP : poursuites parcimonieuses classiques. "
    "IRLS = repondérage itéré (ℓp). "
    "Ce n’est pas « LP » : LP = programmation linéaire (autre algorithme). "
    "BP = Basis Pursuit ; LASSO = pénalisation ℓ1."
)

UI_HELP_DICT_BLOC = (
    "Les 5 choix sont tous utiles selon ce que vous voulez montrer : "
    "(1) DCT seule — référence rapide, pas d’apprentissage, idéal baseline. "
    "(2) Mixte — DCT + vrais patchs sans itérer K-SVD : bon compromis « structure + données ». "
    "(3) K-SVD init. aléatoire — dico appris from scratch sur les patchs (plus lent, très démo apprentissage). "
    "(4) K-SVD depuis DCT — même apprentissage mais départ stable (souvent meilleur que l’aléatoire). "
    "(5) K-SVD depuis mixte — départ riche ; utile pour comparer initialisations dans le rapport. "
    "K-SVD = `learn_ksvd_full` ; itérations > 0 indispensables pour un vrai apprentissage (sinon défaut interne si mode ksvd*)."
)

UI_HELP_DICT_COMBO_LINES = (
    "1. DCT seule — baseline cours, aucun coût K-SVD.\n"
    "2. Mixte — moitié DCT + moitié colonnes de patchs (fixe, sans boucle K-SVD).\n"
    "3. K-SVD (aléatoire) — apprentissage complet, init. par patchs tirés.\n"
    "4. K-SVD (depuis DCT) — apprentissage avec init. DCT tronquée.\n"
    "5. K-SVD (depuis mixte) — apprentissage avec init. mixte DCT+patchs."
)

UI_HELP_EMPREINTE_RESULTS = (
    "Estimation à partir du temps d’exécution, d’une puissance PC supposée et d’un facteur g CO₂eq/kWh. "
    "Utile pour comparer deux essais et pour le volet « sensibilisation » du rapport — pas un bilan carbone audité."
)


def format_empreinte_pour_ui(emp: dict[str, Any] | None) -> tuple[str, str]:
    """Ligne principale (chiffres) + ligne explicative pour l’onglet Résultats."""
    if not isinstance(emp, dict) or emp.get("co2e_g_estime") is None:
        return (
            "—",
            "Lancez une reconstruction avec « Empreinte carbone » activée (onglet Reconstruction) pour afficher une estimation.",
        )
    co2 = float(emp.get("co2e_g_estime", 0.0))
    wh = float(emp.get("energie_estimee_wh", 0.0))
    kwh = wh / 1000.0
    wall = float(emp.get("duree_wall_s", 0.0))
    cpu = emp.get("duree_cpu_process_s")
    pw = float(emp.get("hypothese_puissance_w", 0.0))
    gkwh = float(emp.get("hypothese_g_co2_par_kwh", 0.0))
    main = f"≈ {co2:.4f} g CO₂eq   ·   ≈ {kwh:.6f} kWh   ·   {wall:.1f} s (temps mural)"
    if cpu is not None:
        main += f"   ·   CPU processus ~ {float(cpu):.1f} s"
    co2_cpu = emp.get("co2e_g_estime_temps_cpu")
    if co2_cpu is not None:
        lo, hi = min(co2, float(co2_cpu)), max(co2, float(co2_cpu))
        if abs(hi - lo) > 1e-9:
            main += f"   ·   fourchette indicative {lo:.4f}–{hi:.4f} g (CPU vs mural, mêmes hypothèses W)"
    detail = (
        f"Hypothèses : ≈ {pw:.0f} W (machine), ≈ {gkwh:.0f} g CO₂eq/kWh (mix électrique). "
        f"Le temps mural donne souvent un ordre de grandeur haut ; le temps CPU une borne basse si l’énergie suivait "
        f"uniquement le travail processeur (voir message complet dans l’export). "
        f"{UI_HELP_EMPREINTE_RESULTS} Voir aussi EMPREINTE.md à la racine du projet."
    )
    return main, detail


def format_stockage_bcs_pour_ui(stk: dict[str, Any] | None) -> str:
    """Résumé IHM : uniquement avant (fichier) → après compression (mesures y + Φ), sans reconstruction."""
    if not isinstance(stk, dict):
        return ""
    ref = int(stk.get("octets_avant_fichier_ou_raster") or stk.get("octets_reference_pour_gain") or 0)
    mod = int(stk.get("octets_modele_mesures_plus_phi") or 0)
    if ref <= 0:
        return ""
    pct = float(stk.get("taux_reduction_vs_reference_pct") or 0.0)
    av_mib = float(stk.get("avant_compression_mib") or ref / (1024.0**2))
    ap_mib = float(stk.get("apres_compression_mib") or mod / (1024.0**2))
    g_mib = float(stk.get("gain_mib") or (ref - mod) / (1024.0**2))
    av_ko = float(stk.get("avant_compression_ko") or ref / 1024.0)
    ap_ko = float(stk.get("apres_compression_ko") or mod / 1024.0)
    g_ko = float(stk.get("gain_ko") or (ref - mod) / 1024.0)
    est_fichier = bool(stk.get("avant_est_taille_fichier_disque"))
    avant_lbl = "Avant (fichier image)" if est_fichier else "Avant (raster recadré)"
    if g_mib >= 0:
        gain_txt = f"Gain (compression seule) : {g_mib:.4f} MiB ({g_ko:.1f} ko), ~{pct:.1f} % de moins qu’avant"
    else:
        gain_txt = f"Pas de gain : mesures+Φ dépassent l’avant de {-g_mib:.4f} MiB ({-g_ko:.1f} ko)"
    return (
        f"{avant_lbl} : {av_mib:.4f} MiB ({av_ko:.1f} ko) · "
        f"Après compression (mesures y + Φ uniquement) : {ap_mib:.4f} MiB ({ap_ko:.1f} ko). {gain_txt}. "
        f"Détails : stockage_compression.txt."
    )


def clear_ttk_label_image(label: tk.Widget, text: str) -> None:
    """
    Enlève l’image d’un ttk.Label et fixe le texte. Sur certaines versions Tcl / OS,
    ``configure(image=\"\")`` peut lever TclError ; on l’ignore.
    """
    label.configure(text=text)
    try:
        label.configure(image="")
    except tk.TclError:
        pass


def ensure_project_root() -> Path:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def array_to_photo(array: np.ndarray, max_size: tuple[int, int] = (320, 320)) -> ImageTk.PhotoImage:
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("L'image doit être une matrice 2D.")
    arr = np.nan_to_num(arr)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    image = Image.fromarray(arr, mode="L")
    image.thumbnail(max_size)
    return ImageTk.PhotoImage(image)


def path_to_photo(path: str | os.PathLike[str], max_size: tuple[int, int] = (320, 320)) -> ImageTk.PhotoImage:
    image = Image.open(path)
    # RGB/RGBA/grayscale → RGB pour PhotoImage (évite affichage vide avec certains thèmes ttk)
    if image.mode == "RGBA":
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        image = bg
    elif image.mode == "L":
        image = image.convert("RGB")
    elif image.mode not in ("RGB",):
        image = image.convert("RGB")
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(image)


def figure_to_photo(fig: Figure, *, dpi: int = 120, max_width_px: int = 920) -> ImageTk.PhotoImage:
    buf = io.BytesIO()
    fig.patch.set_facecolor("#ffffff")
    for ax in fig.axes:
        ax.set_facecolor("#fafbfc")
        for spine in ax.spines.values():
            spine.set_color("#d1d5db")
        ax.tick_params(colors="#374151", labelsize=9)
        ax.title.set_color("#111827")
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
    buf.seek(0)
    image = Image.open(buf)
    if image.width > max_width_px:
        ratio = max_width_px / float(image.width)
        image = image.resize((max_width_px, max(1, int(image.height * ratio))), Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(image)


def parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def parse_int(value: str, default: int | None = None) -> int | None:
    value = str(value).strip()
    if not value:
        return default
    return int(value)


def parse_float(value: str, default: float | None = None) -> float | None:
    value = str(value).strip()
    if not value:
        return default
    return float(value)


def latest_subdir(base_dir: str | os.PathLike[str]) -> str:
    base = Path(base_dir)
    if not base.exists():
        return ""
    dirs = [p for p in base.iterdir() if p.is_dir()]
    if not dirs:
        return ""
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(dirs[0])


def open_path(path: str) -> None:
    if not path:
        return
    system = platform.system().lower()
    try:
        if system == "windows":
            os.startfile(path)
        elif system == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception:
        pass


def metrics_rows(metrics_by_method: dict[str, dict[str, Any]]) -> list[tuple[str, float, float, float, float]]:
    rows: list[tuple[str, float, float, float, float]] = []
    for method, metrics in metrics_by_method.items():
        rows.append(
            (
                method.upper(),
                float(metrics.get("psnr", 0.0)),
                float(metrics.get("mse", 0.0)),
                float(metrics.get("relative_error", 0.0)),
                float(metrics.get("execution_time", 0.0)),
            )
        )
    return rows


def co2eq_par_methode_prorata_temps(
    metrics_by_method: dict[str, dict[str, Any]],
    empreinte: dict[str, Any] | None,
) -> dict[str, str]:
    """
    Répartit l’estimation d’empreinte totale de l’exécution entre les méthodes au prorata du temps CPU indiqué par ligne.
    """
    keys = list(metrics_by_method.keys())
    if not keys:
        return {}
    if not isinstance(empreinte, dict) or empreinte.get("co2e_g_estime") is None:
        return {k: "—" for k in keys}
    total_co2 = float(empreinte.get("co2e_g_estime", 0.0))
    times = {k: float(metrics_by_method[k].get("execution_time", 0.0)) for k in keys}
    t_sum = sum(times.values())
    if t_sum > 0:
        return {k: f"{total_co2 * times[k] / t_sum:.4f}" for k in keys}
    n = len(keys)
    return {k: f"{total_co2 / n:.4f}" for k in keys}


def build_pipeline_diagram_figure() -> Figure:
    """
    Schéma du flux réellement exécuté (découpage → mesure → solveur → recomposition).
    Affiché sur l’accueil pour compléter la lecture de ``main.py`` / ``main_backend``.
    """
    fig = Figure(figsize=(11.4, 3.15), facecolor="#f8fafc")
    ax = fig.add_subplot(111, facecolor="#f8fafc")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.11)

    labels = (
        "1 · Image I\n(niveaux de gris)",
        "2 · Patchs\nx ∈ ℝᴺ, N=B²",
        "3 · Mesure\ny = Φ x",
        "4 · D + solveur\nα̂ parcimonieux",
        "5 · Patch estimé\nx̂ ≈ D α̂",
        "6 · Image Î\nrecollement",
    )
    face_alt = ("#eef4fc", "#e4edf7", "#eef4fc", "#e4edf7", "#eef4fc", "#e4edf7")
    n = len(labels)
    centers = [(i + 0.5) / n for i in range(n)]
    half = 0.34 / n
    for xc, lab, fc in zip(centers, labels, face_alt):
        ax.text(
            xc,
            0.48,
            lab,
            ha="center",
            va="center",
            fontsize=8.4,
            linespacing=1.18,
            bbox={
                "boxstyle": "round,pad=0.34",
                "facecolor": fc,
                "edgecolor": "#1e5a8e",
                "linewidth": 1.15,
            },
        )
    for i in range(n - 1):
        x0, x1 = centers[i] + half, centers[i + 1] - half
        ax.annotate(
            "",
            xy=(x1, 0.48),
            xytext=(x0, 0.48),
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#334155",
                "lw": 1.45,
                "mutation_scale": 12,
                "shrinkA": 2,
                "shrinkB": 2,
            },
        )
    ax.text(
        0.5,
        0.93,
        "Pipeline logiciel (chaîne patch par patch — main_backend / Tratement_Image)",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        0.5,
        0.055,
        "Φ : Φ₁…Φ₄  ·  D : DCT fixe, mixte DCT+patchs, ou K-SVD (init. aléatoire / DCT / mixte)",
        ha="center",
        va="center",
        fontsize=8,
        color="#475569",
    )
    return fig


def build_metrics_figure(metrics_by_method: dict[str, dict[str, Any]]) -> Figure:
    keys = list(metrics_by_method.keys())
    methods_lbl = [str(m).upper() for m in keys]
    psnr = [float(metrics_by_method[m].get("psnr", 0.0)) for m in keys]
    mse = [float(metrics_by_method[m].get("mse", 0.0)) for m in keys]

    fig = Figure(figsize=(12.5, 5.2), dpi=110)
    fig.patch.set_facecolor("#ffffff")
    fig.suptitle(
        "Qualité de reconstruction par méthode",
        fontsize=14,
        fontweight="bold",
        color="#0f172a",
        y=0.97,
    )

    color_psnr = "#1d4ed8"
    color_mse = "#0f766e"
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    for ax in (ax1, ax2):
        ax.set_facecolor("#f8fafc")
        ax.tick_params(axis="both", labelsize=11, colors="#334155")
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
            spine.set_color("#cbd5e1")

    ax1.bar(methods_lbl, psnr, color=color_psnr, edgecolor="#1e40af", linewidth=0.9, alpha=0.92, zorder=2)
    ax1.set_title("PSNR (plus haut = mieux)", fontsize=12, fontweight="bold", color="#1e293b", pad=10)
    ax1.set_ylabel("dB", fontsize=11)
    ax1.tick_params(axis="x", rotation=28)
    ax1.grid(axis="y", linestyle="--", alpha=0.45, color="#94a3b8", zorder=0)
    ax1.set_axisbelow(True)

    ax2.bar(methods_lbl, mse, color=color_mse, edgecolor="#115e59", linewidth=0.9, alpha=0.92, zorder=2)
    ax2.set_title("MSE (plus bas = mieux)", fontsize=12, fontweight="bold", color="#1e293b", pad=10)
    ax2.set_ylabel("MSE", fontsize=11)
    ax2.tick_params(axis="x", rotation=28)
    ax2.grid(axis="y", linestyle="--", alpha=0.45, color="#94a3b8", zorder=0)
    ax2.set_axisbelow(True)

    fig.subplots_adjust(top=0.88, bottom=0.20, left=0.09, right=0.97, wspace=0.42)
    return fig


def build_sweep_figure(ratios: Iterable[float], psnr_by_method: dict[str, list[float]]) -> Figure:
    fig = Figure(figsize=(8.2, 4.4), dpi=100)
    fig.patch.set_facecolor("#ffffff")
    fig.suptitle(
        "PSNR en fonction du taux de mesures",
        fontsize=12,
        fontweight="bold",
        color="#0f172a",
        y=0.97,
    )
    ax = fig.add_subplot(111)
    ax.set_facecolor("#f8fafc")
    ratios_list = list(ratios)
    palette = ("#1d4ed8", "#0d9488", "#b45309", "#7c3aed", "#be123c", "#047857", "#4338ca", "#0e7490")
    for i, (method, ys) in enumerate(psnr_by_method.items()):
        c = palette[i % len(palette)]
        ax.plot(
            ratios_list,
            ys,
            marker="o",
            markersize=6,
            linewidth=2.2,
            label=method.upper(),
            color=c,
            markeredgecolor="white",
            markeredgewidth=0.6,
        )
    ax.set_xlabel("Ratio de mesures (fraction si ≤ 1, sinon %)", fontsize=9, color="#334155")
    ax.set_ylabel("PSNR (dB)", fontsize=9, color="#334155")
    ax.tick_params(labelsize=9, colors="#475569")
    ax.grid(True, linestyle="--", alpha=0.5, color="#94a3b8")
    ax.set_axisbelow(True)
    leg = ax.legend(
        title="Méthode",
        framealpha=1.0,
        fontsize=9,
        title_fontsize=9,
        loc="best",
        edgecolor="#e2e8f0",
        fancybox=False,
    )
    leg.get_frame().set_facecolor("#ffffff")
    leg.get_frame().set_linewidth(0.8)
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
        spine.set_linewidth(0.8)
    fig.subplots_adjust(top=0.88, bottom=0.14, left=0.1, right=0.98)
    return fig


def build_section6_mp_coherence_figure(
    mp_rows: list[list[str]] | None,
    coherence_rows: list[list[str]] | None,
) -> Figure:
    """
    Deux graphiques : M(P) depuis ``M_pour_P.csv`` ;
    cohérence mutuelle μ(Φ,D) pour les quatre Φ depuis ``coherence_mutuelle.csv``.
    """
    fig = Figure(figsize=(9.2, 7.2), dpi=100)
    fig.patch.set_facecolor("#ffffff")
    fig.suptitle(
        "Mesures M(P) et cohérence mutuelle μ(Φ, D)",
        fontsize=12,
        fontweight="bold",
        color="#0f172a",
        y=0.98,
    )
    ax0 = fig.add_subplot(2, 1, 1)
    ax1 = fig.add_subplot(2, 1, 2)
    for ax in (ax0, ax1):
        ax.set_facecolor("#f8fafc")
        for spine in ax.spines.values():
            spine.set_color("#cbd5e1")
        ax.grid(True, linestyle="--", alpha=0.5, color="#94a3b8")
        ax.set_axisbelow(True)

    # --- M(P)
    if mp_rows and len(mp_rows) >= 2:
        try:
            ps_m: list[int] = []
            ms: list[float] = []
            for r in mp_rows[1:]:
                if len(r) >= 2 and r[0].strip():
                    ps_m.append(int(float(r[0].replace(",", "."))))
                    ms.append(float(r[1].replace(",", ".")))
            if ps_m:
                ax0.plot(ps_m, ms, marker="s", color="#1d4ed8", linewidth=2, markersize=7, markeredgecolor="white")
                ax0.set_xticks(ps_m)
        except (ValueError, IndexError):
            ax0.text(0.5, 0.5, "M_pour_P.csv : format inattendu", ha="center", va="center", transform=ax0.transAxes)
    else:
        ax0.text(
            0.5,
            0.5,
            "Générez les tableaux pour obtenir M_pour_P.csv",
            ha="center",
            va="center",
            transform=ax0.transAxes,
            color="#64748b",
        )
    ax0.set_title("M = ⌈P·N/100⌉ en fonction du pourcentage P (N = B²)", fontsize=10, fontweight="bold", color="#334155")
    ax0.set_xlabel("P (%)", fontsize=9, color="#475569")
    ax0.set_ylabel("M", fontsize=9, color="#475569")

    # --- Cohérence μ : 4 courbes (Φ₁…Φ₄), comme le tableau du PDF
    palette = ("#1d4ed8", "#0d9488", "#b45309", "#7c3aed")
    if coherence_rows and len(coherence_rows) >= 2:
        headers = coherence_rows[0]
        p_cols: list[tuple[int, int]] = []  # (index, P)
        for i, h in enumerate(headers):
            hs = h.strip()
            if hs.startswith("P_"):
                try:
                    p_cols.append((i, int(hs.split("_", 1)[1])))
                except ValueError:
                    continue
        p_cols.sort(key=lambda x: x[1])
        if not p_cols:
            ax1.text(0.5, 0.5, "Pas de colonnes P_* dans le CSV", ha="center", va="center", transform=ax1.transAxes)
        else:
            percents = [p for _, p in p_cols]
            for ri, row in enumerate(coherence_rows[1:]):
                if len(row) < 1:
                    continue
                phi_key = row[0].strip().lower()
                leg = PHI_SHORT_LABEL.get(phi_key, row[0].strip().upper())
                ci = list(PHI_COURS_KEYS).index(phi_key) if phi_key in PHI_COURS_KEYS else ri % len(palette)
                c = palette[ci % len(palette)]
                ys: list[float] = []
                ok = True
                for idx, _p in p_cols:
                    if idx >= len(row):
                        ok = False
                        break
                    try:
                        ys.append(float(row[idx].replace(",", ".")))
                    except ValueError:
                        ok = False
                        break
                if ok and ys:
                    ax1.plot(
                        percents,
                        ys,
                        marker="o",
                        linewidth=2,
                        markersize=6,
                        label=leg,
                        color=c,
                        markeredgecolor="white",
                        markeredgewidth=0.5,
                    )
            handles, labels = ax1.get_legend_handles_labels()
            if labels:
                ax1.legend(handles, labels, title="Matrice Φ", fontsize=8, title_fontsize=8, loc="best", framealpha=1.0)
            ax1.set_xticks(percents)
    else:
        ax1.text(
            0.5,
            0.5,
            "Générez les tableaux pour obtenir coherence_mutuelle.csv",
            ha="center",
            va="center",
            transform=ax1.transAxes,
            color="#64748b",
        )
    ax1.set_title("Cohérence mutuelle μ(Φ, D) — une courbe par Φ (cours)", fontsize=10, fontweight="bold", color="#334155")
    ax1.set_xlabel("Pourcentage de mesures P (%)", fontsize=9, color="#475569")
    ax1.set_ylabel("μ", fontsize=9, color="#475569")

    fig.subplots_adjust(top=0.91, bottom=0.08, left=0.11, right=0.97, hspace=0.35)
    return fig


def build_sparsity_figure(alphas_by_method: dict[str, Any], *, eps: float = 1e-8) -> Figure:
    """
    Bar chart du nombre moyen de coefficients non nuls (‖α‖₀ approché)
    par méthode, moyenné sur tous les patchs reconstruits.

    alphas_by_method : {methode: np.ndarray de forme (K, NB_used)}
    """
    import numpy as np

    methods = list(alphas_by_method.keys())
    nnz_means = []
    nnz_stds = []
    for m in methods:
        A = np.asarray(alphas_by_method[m])  # (K, NB)
        nnz_per_patch = np.sum(np.abs(A) > eps, axis=0).astype(float)  # (NB,)
        nnz_means.append(float(np.mean(nnz_per_patch)))
        nnz_stds.append(float(np.std(nnz_per_patch)))

    fig = Figure(figsize=(max(5, len(methods) * 1.2), 4), dpi=110)
    fig.patch.set_facecolor("#f7f8fb")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#ffffff")

    colors = ["#1e5a8e", "#2e86c1", "#117a65", "#b7950b", "#884ea0", "#cb4335", "#1a5276", "#117864"]
    x = range(len(methods))
    bars = ax.bar(
        x,
        nnz_means,
        yerr=nnz_stds,
        capsize=4,
        color=[colors[i % len(colors)] for i in x],
        edgecolor="#334155",
        linewidth=0.8,
        error_kw={"elinewidth": 1.2, "ecolor": "#64748b"},
    )

    # Valeur au-dessus de chaque barre
    for bar, mean, std in zip(bars, nnz_means, nnz_stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.3,
            f"{mean:.1f}",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#1a2332",
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels([m.upper() for m in methods], fontsize=10)
    ax.set_ylabel("Nombre moyen de coefficients non nuls (‖α‖₀)", fontsize=9, color="#475569")
    ax.set_title("Parcimonie par méthode — coefficients non nuls moyens sur tous les patchs", fontsize=10, fontweight="bold", color="#334155")
    ax.yaxis.grid(True, alpha=0.4, linestyle="--")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    return fig


def build_dico_comparison_table(
    result_a: dict[str, Any],
    result_b: dict[str, Any],
    *,
    eps: float = 1e-8,
) -> tuple[list[str], list[list[str]]]:
    """
    Construit les données du tableau de comparaison entre deux reconstructions.
    Retourne (colonnes, lignes) sous forme de listes de chaînes.
    Chaque ligne : [Méthode, PSNR_A, PSNR_B, ΔPSNR, MSE_A, MSE_B, ErrRel_A, ErrRel_B, Temps_A, Temps_B, Parcimonie_A, Parcimonie_B]
    """
    import numpy as np

    def _label(res: dict) -> str:
        p = res.get("params", {})
        dt = str(p.get("dictionary_type", "?")).upper()
        phi = str(p.get("measurement_mode", "?"))
        ratio = p.get("ratio", "?")
        ratio_str = f"{int(float(ratio)*100) if float(ratio) <= 1 else int(float(ratio))}%"
        return f"{dt} · {phi} · {ratio_str}"

    label_a = _label(result_a)
    label_b = _label(result_b)

    metrics_a = result_a.get("metrics", {})
    metrics_b = result_b.get("metrics", {})
    alphas_a = result_a.get("alphas_by_method", {})
    alphas_b = result_b.get("alphas_by_method", {})

    # Union des méthodes présentes dans les deux résultats
    methods = list(dict.fromkeys(list(metrics_a.keys()) + list(metrics_b.keys())))

    def _nnz(alphas: dict, meth: str) -> str:
        if meth not in alphas:
            return "—"
        A = np.asarray(alphas[meth])
        return f"{float(np.mean(np.sum(np.abs(A) > eps, axis=0))):.1f}"

    def _get(metrics: dict, meth: str, key: str, fmt: str) -> str:
        if meth not in metrics:
            return "—"
        v = metrics[meth].get(key)
        if v is None:
            return "—"
        try:
            return fmt.format(float(v))
        except (ValueError, TypeError):
            return str(v)

    columns = [
        "Méthode",
        f"PSNR A\n{label_a}",
        f"PSNR B\n{label_b}",
        "ΔPSNR (B−A)",
        f"MSE A",
        f"MSE B",
        f"Err.rel A",
        f"Err.rel B",
        f"Temps A (s)",
        f"Temps B (s)",
        f"‖α‖₀ moy A",
        f"‖α‖₀ moy B",
    ]

    rows = []
    for m in methods:
        psnr_a_str = _get(metrics_a, m, "psnr", "{:.2f}")
        psnr_b_str = _get(metrics_b, m, "psnr", "{:.2f}")
        try:
            delta = float(metrics_b[m]["psnr"]) - float(metrics_a[m]["psnr"])
            delta_str = f"{delta:+.2f}"
        except (KeyError, TypeError, ValueError):
            delta_str = "—"

        rows.append([
            m.upper(),
            psnr_a_str,
            psnr_b_str,
            delta_str,
            _get(metrics_a, m, "mse", "{:.2f}"),
            _get(metrics_b, m, "mse", "{:.2f}"),
            _get(metrics_a, m, "relative_error", "{:.4f}"),
            _get(metrics_b, m, "relative_error", "{:.4f}"),
            _get(metrics_a, m, "execution_time", "{:.2f}"),
            _get(metrics_b, m, "execution_time", "{:.2f}"),
            _nnz(alphas_a, m),
            _nnz(alphas_b, m),
        ])

    return columns, rows, label_a, label_b