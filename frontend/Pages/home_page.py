from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

from PIL import Image, ImageTk

from .base_page import BasePage

POSTER_BG = "#0b1220"
TEXT = "#9fb0c7"
POSTER_SCALE = 1.12

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS


def _default_asset_path() -> Path:
    return Path(__file__).resolve().parent.parent / "Assets" / "pipeline_complet.png"


class HomePage(BasePage):
    """Page d'accueil : poster du pipeline sous forme d'image (voir frontend/Assets/pipeline_complet.png)."""

    def __init__(self, parent: tk.Misc, app) -> None:
        super().__init__(parent, app)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._photos: list[ImageTk.PhotoImage] = []

        container = ttk.Frame(self, style="Panel.TFrame")
        container.grid(row=0, column=0, sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            container,
            bg=POSTER_BG,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self._build_poster()

    def refresh(self) -> None:
        return None

    def _resolve_image_path(self) -> Path | None:
        p = _default_asset_path()
        if p.is_file():
            return p
        alt = self.state.project_root / "frontend" / "Assets" / "pipeline_complet.png"
        if alt.is_file():
            return alt
        return None

    def _build_poster(self) -> None:
        self.canvas.delete("all")
        self._photos.clear()

        path = self._resolve_image_path()
        if path is None:
            self.canvas.create_text(
                24,
                32,
                text=(
                    "Image du schéma introuvable.\n"
                    "Générez-la avec :  python3 tools/build_pipeline_poster.py"
                ),
                anchor="nw",
                fill=TEXT,
                font=("Arial", 12),
            )
            self.canvas.configure(scrollregion=(0, 0, 640, 200), height=200)
            return

        with Image.open(path) as im:
            poster = im.convert("RGB")
        if POSTER_SCALE != 1.0:
            w0, h0 = poster.size
            poster = poster.resize((int(w0 * POSTER_SCALE), int(h0 * POSTER_SCALE)), RESAMPLE)
        w, h = poster.size
        photo = ImageTk.PhotoImage(poster)
        self._photos.append(photo)
        self.canvas.create_image(0, 0, image=photo, anchor="nw")

        self.update_idletasks()
        pad_x, pad_y = 24, 24
        bg_id = self.canvas.create_rectangle(0, 0, w + pad_x, h + pad_y, fill=POSTER_BG, outline="")
        self.canvas.tag_lower(bg_id)
        self.canvas.configure(scrollregion=(0, 0, w + pad_x, h + pad_y), height=h + pad_y)
