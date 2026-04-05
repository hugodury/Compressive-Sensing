from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from frontend.utils import (
    array_to_photo,
    build_metrics_figure,
    clear_ttk_label_image,
    co2eq_par_methode_prorata_temps,
    figure_to_photo,
    format_empreinte_pour_ui,
    format_stockage_bcs_pour_ui,
    metrics_rows,
)
from .base_page import BasePage


class ResultsPage(BasePage):
    def __init__(self, parent: tk.Misc, app) -> None:
        super().__init__(parent, app)
        self._photos: list = []

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self, style="App.TFrame")
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(top, text="Résultats de reconstruction", style="Title.TLabel").pack(anchor="w")
        self.summary_label = ttk.Label(top, text="Aucun résultat pour le moment.", style="Muted.TLabel")
        self.summary_label.pack(anchor="w", pady=(6, 0))
        self.stockage_label = ttk.Label(top, text="", style="Muted.TLabel", wraplength=960, justify="left")
        self.stockage_label.pack(anchor="w", pady=(4, 0))

        body = ttk.Frame(self, style="App.TFrame")
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=2)
        body.columnconfigure(1, weight=3)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body, style="Panel.TFrame")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.rowconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        right = ttk.Frame(body, style="Panel.TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        card_table = ttk.Frame(left, style="Card.TFrame", padding=14)
        card_table.grid(row=0, column=0, sticky="nsew")
        card_table.columnconfigure(0, weight=1)
        card_table.rowconfigure(1, weight=1)
        ttk.Label(card_table, text="Métriques et empreinte carbone (estimation)", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        columns = ("methode", "psnr", "mse", "err", "temps", "co2")
        self.tree = ttk.Treeview(card_table, columns=columns, show="headings", height=6)
        self.tree.heading("methode", text="Méthode")
        self.tree.heading("psnr", text="PSNR")
        self.tree.heading("mse", text="MSE")
        self.tree.heading("err", text="Erreur rel.")
        self.tree.heading("temps", text="Temps (s)")
        self.tree.heading("co2", text="CO₂eq (g)")
        for col, width in zip(columns, (88, 72, 72, 88, 72, 88)):
            self.tree.column(col, width=width, anchor="center")
        self.tree.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        self.empreinte_table_note = ttk.Label(
            card_table,
            text="",
            style="CardMuted.TLabel",
            justify="left",
            wraplength=920,
        )
        self.empreinte_table_note.grid(row=2, column=0, sticky="ew", pady=(8, 0))

        card_chart = ttk.Frame(left, style="Card.TFrame", padding=14)
        card_chart.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        card_chart.columnconfigure(0, weight=1)
        card_chart.rowconfigure(2, weight=1)
        ttk.Label(card_chart, text="Synthèse graphique", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            card_chart,
            text="PSNR en dB : plus c’est élevé, mieux c’est (souvent >30 dB utilisable, >40 dB très bon). MSE : plus bas = mieux.",
            style="CardMuted.TLabel",
            wraplength=520,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(4, 6))
        self.chart_label = ttk.Label(card_chart, style="CardBody.TLabel", anchor="center")
        self.chart_label.grid(row=2, column=0, sticky="nsew", pady=(4, 0))

        self.images_container = ttk.Frame(right, style="Card.TFrame", padding=14)
        self.images_container.grid(row=0, column=0, sticky="nsew")
        self.images_container.columnconfigure(0, weight=1)
        self.images_container.rowconfigure(1, weight=1)
        ttk.Label(self.images_container, text="Prévisualisation des images", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.grid_frame = ttk.Frame(self.images_container, style="Card.TFrame")
        self.grid_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        self.grid_frame.columnconfigure(0, weight=1)
        self.grid_frame.columnconfigure(1, weight=1)

    def refresh(self) -> None:
        result = self.state.last_result
        self._photos = []

        for item in self.tree.get_children():
            self.tree.delete(item)
        for widget in self.grid_frame.winfo_children():
            widget.destroy()

        if not result:
            self.summary_label.configure(text="Aucun résultat pour le moment.")
            self.stockage_label.configure(text="")
            clear_ttk_label_image(self.chart_label, "Lance une reconstruction depuis l'onglet dédié.")
            self.empreinte_table_note.configure(
                text="Activez « Empreinte carbone » dans Reconstruction pour remplir la colonne CO₂eq (répartition au prorata du temps par méthode)."
            )
            return

        emp_raw = result.get("empreinte") if isinstance(result.get("empreinte"), dict) else None
        metrics = result.get("metrics", {})
        co2_map = co2eq_par_methode_prorata_temps(metrics, emp_raw)
        meth_keys = list(metrics.keys())
        rows = metrics_rows(metrics)
        for i, row in enumerate(rows):
            k = meth_keys[i]
            self.tree.insert(
                "",
                "end",
                values=(
                    row[0],
                    f"{row[1]:.2f}",
                    f"{row[2]:.4f}",
                    f"{row[3]:.4f}",
                    f"{row[4]:.2f}",
                    co2_map[k],
                ),
            )

        _, d_emp = format_empreinte_pour_ui(emp_raw)
        if emp_raw and emp_raw.get("co2e_g_estime") is not None:
            total = float(emp_raw["co2e_g_estime"])
            self.empreinte_table_note.configure(
                text=(
                    f"Total estimé pour cette exécution : ≈ {total:.4f} g CO₂eq. "
                    "Chaque ligne donne la part attribuée à la méthode au prorata de son temps (même hypothèses W et g/kWh). "
                    f"{d_emp}"
                )
            )
        else:
            self.empreinte_table_note.configure(
                text="Colonne CO₂eq : activez le calcul dans Reconstruction. "
                + d_emp
            )

        sk_txt = format_stockage_bcs_pour_ui(result.get("stockage_bcs") if isinstance(result.get("stockage_bcs"), dict) else None)
        self.stockage_label.configure(text=sk_txt)

        best = max(rows, key=lambda x: x[1]) if rows else None
        if best:
            line = f"Meilleure méthode actuelle : {best[0]} — PSNR {best[1]:.2f} dB"
            if emp_raw and emp_raw.get("co2e_g_estime") is not None:
                line += f" — empreinte totale estimée ≈ {float(emp_raw['co2e_g_estime']):.4f} g CO₂eq"
            self.summary_label.configure(text=line)
        elif rows:
            self.summary_label.configure(text="Résultats chargés.")
        else:
            self.summary_label.configure(text="Aucune métrique dans ce résultat.")

        fig = build_metrics_figure(metrics)
        self.update_idletasks()
        avail_w = max(720, min(1280, (self.winfo_width() or 1000) - 100))
        chart_photo = figure_to_photo(fig, max_width_px=avail_w, dpi=120)
        self._photos.append(chart_photo)
        self.chart_label.configure(image=chart_photo, text="")

        images = [("Originale", result.get("original"))]
        for method, arr in result.get("images_by_method", {}).items():
            images.append((method.upper(), arr))

        tw = max(200, min(460, (self.winfo_width() or 900) // 3))
        th = max(150, int(tw * 0.68))

        for index, (title, arr) in enumerate(images):
            frame = ttk.Frame(self.grid_frame, style="Panel.TFrame", padding=8)
            frame.grid(row=index // 2, column=index % 2, sticky="nsew", padx=6, pady=6)
            self.grid_frame.rowconfigure(index // 2, weight=1)
            ttk.Label(frame, text=title, style="Section.TLabel").pack(anchor="w")
            photo = array_to_photo(arr, (tw, th))
            self._photos.append(photo)
            ttk.Label(frame, image=photo).pack(anchor="center", pady=(8, 0))
