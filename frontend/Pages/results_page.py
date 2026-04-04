from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from frontend.utils import array_to_photo, build_metrics_figure, figure_to_photo, metrics_rows
from .base_page import BasePage


class ResultsPage(BasePage):
    def __init__(self, parent: tk.Misc, app) -> None:
        super().__init__(parent, app)
        self._photos: list = []

        top = ttk.Frame(self, style="App.TFrame")
        top.pack(fill="x")
        ttk.Label(top, text="Résultats de reconstruction", style="Title.TLabel").pack(anchor="w")
        self.summary_label = ttk.Label(top, text="Aucun résultat pour le moment.", style="Muted.TLabel")
        self.summary_label.pack(anchor="w", pady=(6, 12))

        body = ttk.Frame(self, style="App.TFrame")
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=2)
        body.columnconfigure(1, weight=3)

        left = ttk.Frame(body, style="Panel.TFrame")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right = ttk.Frame(body, style="Panel.TFrame")
        right.grid(row=0, column=1, sticky="nsew")

        card_table = ttk.Frame(left, style="Card.TFrame", padding=16)
        card_table.pack(fill="both", expand=True)
        ttk.Label(card_table, text="Tableau des métriques", style="CardTitle.TLabel").pack(anchor="w")
        columns = ("methode", "psnr", "mse", "err", "temps")
        self.tree = ttk.Treeview(card_table, columns=columns, show="headings", height=8)
        self.tree.heading("methode", text="Méthode")
        self.tree.heading("psnr", text="PSNR")
        self.tree.heading("mse", text="MSE")
        self.tree.heading("err", text="Erreur rel.")
        self.tree.heading("temps", text="Temps (s)")
        for col, width in zip(columns, (100, 90, 90, 100, 90)):
            self.tree.column(col, width=width, anchor="center")
        self.tree.pack(fill="both", expand=True, pady=(10, 0))

        card_chart = ttk.Frame(left, style="Card.TFrame", padding=16)
        card_chart.pack(fill="both", expand=True, pady=(10, 0))
        ttk.Label(card_chart, text="Synthèse graphique", style="CardTitle.TLabel").pack(anchor="w")
        self.chart_label = ttk.Label(card_chart, style="App.TLabel")
        self.chart_label.pack(fill="both", expand=True, pady=(10, 0))

        self.images_container = ttk.Frame(right, style="Card.TFrame", padding=16)
        self.images_container.pack(fill="both", expand=True)
        ttk.Label(self.images_container, text="Prévisualisation des images", style="CardTitle.TLabel").pack(anchor="w")
        self.grid_frame = ttk.Frame(self.images_container, style="Card.TFrame")
        self.grid_frame.pack(fill="both", expand=True, pady=(10, 0))

    def refresh(self) -> None:
        result = self.state.last_result
        self._photos = []

        for item in self.tree.get_children():
            self.tree.delete(item)
        for widget in self.grid_frame.winfo_children():
            widget.destroy()

        if not result:
            self.summary_label.configure(text="Aucun résultat pour le moment.")
            self.chart_label.configure(image="", text="Lance une reconstruction depuis l'onglet dédié.")
            return

        metrics = result.get("metrics", {})
        rows = metrics_rows(metrics)
        for row in rows:
            self.tree.insert("", "end", values=(row[0], f"{row[1]:.2f}", f"{row[2]:.4f}", f"{row[3]:.4f}", f"{row[4]:.2f}"))

        best = max(rows, key=lambda x: x[1]) if rows else None
        if best:
            self.summary_label.configure(text=f"Meilleure méthode actuelle : {best[0]} - PSNR {best[1]:.2f} dB")

        fig = build_metrics_figure(metrics)
        chart_photo = figure_to_photo(fig)
        self._photos.append(chart_photo)
        self.chart_label.configure(image=chart_photo)

        images = [("Originale", result.get("original"))]
        for method, arr in result.get("images_by_method", {}).items():
            images.append((method.upper(), arr))

        for index, (title, arr) in enumerate(images):
            frame = ttk.Frame(self.grid_frame, style="Panel.TFrame", padding=8)
            frame.grid(row=index // 2, column=index % 2, sticky="nsew", padx=6, pady=6)
            self.grid_frame.columnconfigure(index % 2, weight=1)
            ttk.Label(frame, text=title, style="Section.TLabel").pack(anchor="w")
            photo = array_to_photo(arr, (360, 260))
            self._photos.append(photo)
            ttk.Label(frame, image=photo).pack(anchor="center", pady=(8, 0))
