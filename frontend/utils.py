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

PHI_MEASUREMENT_MODES: tuple[str, ...] = PHI_COURS_KEYS

# Toutes les clés reconnues par le backend (saisie manuelle / anciens projets)
DICTIONARY_TYPES_ALL: tuple[str, ...] = ("dct", "mixte", "ksvd", "ksvd_dct", "ksvd_mixte", "ksvd_random")

DICTIONARY_UI_LABELS: dict[str, str] = {
    "dct": "DCT",
    "mixte": "Mixte (DCT + patchs)",
    "ksvd": "K-SVD (init. aléatoire)",
    "ksvd_dct": "K-SVD (init. DCT)",
    "ksvd_mixte": "K-SVD (init. mixte)",
    "ksvd_random": "K-SVD (init. aléatoire, alias)",
}

# Choix proposés dans l’IHM (les plus utilisés)
DICTIONARY_TYPES_UI: tuple[str, ...] = ("dct", "mixte", "ksvd_dct", "ksvd_mixte")

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
    "DCT : base fixe. Mixte : DCT complété par des patchs. K-SVD : dictionnaire appris "
    "(n_iter_K-SVD > 0 lance l’optimisation). L’image d’entraînement sert à apprendre D sur une autre image que la cible si vous la renseignez."
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
    main = f"≈ {co2:.4f} g CO₂eq   ·   ≈ {kwh:.6f} kWh   ·   {wall:.1f} s (temps total)"
    if cpu is not None:
        main += f"   ·   CPU processus ~ {float(cpu):.1f} s"
    detail = (
        f"Hypothèses : ≈ {pw:.0f} W (machine), ≈ {gkwh:.0f} g CO₂eq/kWh (mix électrique). "
        f"{UI_HELP_EMPREINTE_RESULTS} Voir aussi EMPREINTE.md à la racine du projet."
    )
    return main, detail


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


def build_metrics_figure(metrics_by_method: dict[str, dict[str, Any]]) -> Figure:
    keys = list(metrics_by_method.keys())
    methods_lbl = [str(m).upper() for m in keys]
    psnr = [float(metrics_by_method[m].get("psnr", 0.0)) for m in keys]
    mse = [float(metrics_by_method[m].get("mse", 0.0)) for m in keys]

    fig = Figure(figsize=(9.2, 4.0), dpi=100)
    fig.patch.set_facecolor("#ffffff")
    fig.suptitle(
        "Qualité de reconstruction par méthode",
        fontsize=12,
        fontweight="bold",
        color="#0f172a",
        y=0.98,
    )

    color_psnr = "#1d4ed8"
    color_mse = "#0f766e"
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    for ax in (ax1, ax2):
        ax.set_facecolor("#f8fafc")
        ax.tick_params(axis="both", labelsize=9, colors="#334155")
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#cbd5e1")

    ax1.bar(methods_lbl, psnr, color=color_psnr, edgecolor="#1e40af", linewidth=0.8, alpha=0.92, zorder=2)
    ax1.set_title("PSNR", fontsize=10, fontweight="bold", color="#1e293b", pad=8)
    ax1.set_ylabel("dB", fontsize=9)
    ax1.tick_params(axis="x", rotation=22)
    ax1.grid(axis="y", linestyle="--", alpha=0.45, color="#94a3b8", zorder=0)
    ax1.set_axisbelow(True)

    ax2.bar(methods_lbl, mse, color=color_mse, edgecolor="#115e59", linewidth=0.8, alpha=0.92, zorder=2)
    ax2.set_title("MSE", fontsize=10, fontweight="bold", color="#1e293b", pad=8)
    ax2.set_ylabel("Erreur quadratique moyenne", fontsize=9)
    ax2.tick_params(axis="x", rotation=22)
    ax2.grid(axis="y", linestyle="--", alpha=0.45, color="#94a3b8", zorder=0)
    ax2.set_axisbelow(True)

    fig.subplots_adjust(top=0.86, bottom=0.18, left=0.08, right=0.98, wspace=0.28)
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
