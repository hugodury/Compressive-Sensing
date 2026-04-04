"""Conteneur Canvas + barre verticale pour le contenu des onglets (hauteur > fenêtre)."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from frontend.theme import SURFACE


def build_vertical_scroll_host(
    parent: tk.Misc,
    *,
    scrollbar_width: int = 18,
) -> tuple[tk.Canvas, ttk.Frame, tk.Scrollbar]:
    """
    Remplit ``parent`` (déjà en grid 0,0 avec poids) avec canvas + scrollbar.
    Retourne (canvas, inner_frame, scrollbar) ; placer le contenu de la page dans ``inner_frame``.
    """
    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(0, weight=1)

    canvas = tk.Canvas(parent, highlightthickness=0, bg=SURFACE, bd=0)
    vsb = tk.Scrollbar(
        parent,
        orient="vertical",
        command=canvas.yview,
        width=scrollbar_width,
        troughcolor="#e2e8f0",
        bg="#cbd5e1",
        activebackground="#94a3b8",
        highlightthickness=0,
        borderwidth=0,
        relief="flat",
    )
    inner = ttk.Frame(canvas, style="Panel.TFrame")
    win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

    def _scroll_cfg(_: tk.Event | None = None) -> None:
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _canvas_width(event: tk.Event) -> None:
        canvas.itemconfigure(win_id, width=max(1, int(event.width)))

    inner.bind("<Configure>", _scroll_cfg)
    canvas.bind("<Configure>", _canvas_width)
    canvas.configure(yscrollcommand=vsb.set)
    canvas.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    return canvas, inner, vsb
