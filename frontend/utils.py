from __future__ import annotations

import io
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageTk
from matplotlib.figure import Figure


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
    image.thumbnail(max_size)
    return ImageTk.PhotoImage(image)


def figure_to_photo(fig: Figure) -> ImageTk.PhotoImage:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    buf.seek(0)
    image = Image.open(buf)
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


def build_metrics_figure(metrics_by_method: dict[str, dict[str, Any]]) -> Figure:
    methods = list(metrics_by_method.keys())
    psnr = [float(metrics_by_method[m].get("psnr", 0.0)) for m in methods]
    mse = [float(metrics_by_method[m].get("mse", 0.0)) for m in methods]

    fig = Figure(figsize=(8, 3.8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.bar(methods, psnr)
    ax1.set_title("PSNR par méthode")
    ax1.set_ylabel("PSNR (dB)")
    ax1.tick_params(axis="x", rotation=25)

    ax2.bar(methods, mse)
    ax2.set_title("MSE par méthode")
    ax2.set_ylabel("MSE")
    ax2.tick_params(axis="x", rotation=25)

    fig.tight_layout()
    return fig


def build_sweep_figure(ratios: Iterable[float], psnr_by_method: dict[str, list[float]]) -> Figure:
    fig = Figure(figsize=(7.5, 4.2))
    ax = fig.add_subplot(111)
    ratios_list = list(ratios)
    for method, ys in psnr_by_method.items():
        ax.plot(ratios_list, ys, marker="o", label=method.upper())
    ax.set_title("PSNR selon le ratio de mesures")
    ax.set_xlabel("Ratio / pourcentage")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig
