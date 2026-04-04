from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from main import run_pipeline, setupParam
from frontend.utils import (
    DICTIONARY_COMBO_TEXT,
    DICTIONARY_COMBO_TO_KEY,
    PHI_COURS_KEYS,
    PHI_COURS_RADIO_LABELS,
    SOLVER_UI_CHOICES,
    build_sweep_figure,
    clear_ttk_label_image,
    dictionary_key_from_combo_selection,
    figure_to_photo,
    parse_float_list,
    parse_int,
)
from .base_page import BasePage


class ComparisonPage(BasePage):
    def __init__(self, parent: tk.Misc, app) -> None:
        super().__init__(parent, app)
        self.photo = None
        self.vars = {
            "ratios": tk.StringVar(value="15,25,50,75"),
            "block_size": tk.StringVar(value="8"),
            "ratio": tk.StringVar(value="25"),
            "dictionary_type": tk.StringVar(),
            "measurement_mode": tk.StringVar(value="phi4"),
            "output_path": tk.StringVar(value=self.state.output_path),
            "image_path": tk.StringVar(value=self.state.image_path),
            "seed": tk.StringVar(value="0"),
        }
        for disp, key in DICTIONARY_COMBO_TO_KEY.items():
            if key == "dct":
                self.vars["dictionary_type"].set(disp)
                break
        else:
            self.vars["dictionary_type"].set(next(iter(DICTIONARY_COMBO_TO_KEY)))
        self.method_vars: dict[str, tk.BooleanVar] = {}
        self._build()

    def _build(self) -> None:
        ttk.Label(self, text="Comparaisons — sweep PSNR vs ratio", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            self,
            text="Même pipeline que la reconstruction : un point par ratio de mesures, une courbe par méthode. Quatre familles Φ disponibles.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(6, 12))

        top = ttk.Frame(self, style="App.TFrame")
        top.pack(fill="x")
        form = ttk.Frame(top, style="Card.TFrame", padding=18)
        form.pack(fill="x")

        self._entry(form, 0, "Image", "image_path")
        self._entry(form, 1, "Ratios à tester", "ratios")
        self._entry(form, 2, "B", "block_size")
        self._entry(form, 3, "Ratio de base (calcul M)", "ratio")
        self._entry(form, 4, "Seed", "seed")
        self._entry(form, 5, "Sortie", "output_path")
        self._combo(form, 6, "Dictionnaire", "dictionary_type", list(DICTIONARY_COMBO_TEXT))

        phi_lf = ttk.LabelFrame(form, text=" Matrice Φ — quatre familles (Φ₁…Φ₄) ", padding=10)
        phi_lf.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(10, 6))
        phi_grid = ttk.Frame(phi_lf, style="Card.TFrame")
        phi_grid.pack(fill="x")
        for i, pk in enumerate(PHI_COURS_KEYS):
            ttk.Radiobutton(
                phi_grid,
                text=PHI_COURS_RADIO_LABELS[pk],
                variable=self.vars["measurement_mode"],
                value=pk,
            ).grid(row=i // 2, column=i % 2, sticky="nw", padx=(0, 14), pady=3)

        methods = ttk.Frame(form, style="Card.TFrame", padding=(10, 0))
        methods.grid(row=0, column=2, rowspan=11, sticky="ne", padx=(18, 0))
        ttk.Label(methods, text="Méthodes", style="CardTitle.TLabel").pack(anchor="w")
        mgrid = ttk.Frame(methods, style="Card.TFrame")
        mgrid.pack(fill="x", pady=(6, 0))
        for i, (method, caption) in enumerate(SOLVER_UI_CHOICES):
            var = tk.BooleanVar(value=method in {"omp", "cosamp"})
            self.method_vars[method] = var
            r, c = divmod(i, 2)
            ttk.Checkbutton(mgrid, text=caption, variable=var).grid(row=r, column=c, sticky="w", padx=4, pady=3)

        ttk.Button(form, text="Lancer le sweep", style="Primary.TButton", command=self.run_sweep).grid(
            row=10, column=0, columnspan=2, sticky="ew", pady=(16, 0)
        )

        form.columnconfigure(1, weight=1)

        bottom = ttk.Frame(self, style="App.TFrame")
        bottom.pack(fill="both", expand=True, pady=(12, 0))
        left = ttk.Frame(bottom, style="Card.TFrame", padding=16)
        left.pack(side="left", fill="both", expand=True)
        ttk.Label(left, text="Courbe PSNR", style="CardTitle.TLabel").pack(anchor="w")
        self.chart_label = ttk.Label(left, style="CardBody.TLabel")
        self.chart_label.pack(fill="both", expand=True, pady=(10, 0))

        right = ttk.Frame(bottom, style="Card.TFrame", padding=16)
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))
        ttk.Label(right, text="Tableau résumé", style="CardTitle.TLabel").pack(anchor="w")
        cols = ("ratio", "methode", "psnr")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=14)
        for col, text in zip(cols, ["Ratio", "Méthode", "PSNR"]):
            self.tree.heading(col, text=text)
        self.tree.pack(fill="both", expand=True, pady=(10, 0))

    def _entry(self, parent: ttk.Frame, row: int, label: str, key: str) -> None:
        ttk.Label(parent, text=label, style="CardBody.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
        ttk.Entry(parent, textvariable=self.vars[key], width=28).grid(row=row, column=1, sticky="ew", pady=4)
        parent.columnconfigure(1, weight=1)

    def _combo(self, parent: ttk.Frame, row: int, label: str, key: str, values: list[str]) -> None:
        ttk.Label(parent, text=label, style="CardBody.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
        ttk.Combobox(parent, textvariable=self.vars[key], values=values, state="readonly", width=36).grid(
            row=row, column=1, sticky="ew", pady=4
        )

    def run_sweep(self) -> None:
        try:
            methods = [m for m, v in self.method_vars.items() if v.get()]
            if not methods:
                raise ValueError("Sélectionne au moins une méthode.")

            params = setupParam(
                image_path=self.vars["image_path"].get().strip(),
                block_size=int(self.vars["block_size"].get()),
                ratio=float(self.vars["ratio"].get()),
                methodes=methods,
                dictionary_type=dictionary_key_from_combo_selection(self.vars["dictionary_type"].get()),
                measurement_mode=self.vars["measurement_mode"].get().strip(),
                output_path=self.vars["output_path"].get().strip(),
                seed=parse_int(self.vars["seed"].get(), 0),
            )
            sweep = run_pipeline(
                params,
                etapes=("sweep_graph",),
                sweep_ratios=parse_float_list(self.vars["ratios"].get()),
            )["sweep_graphique"]

            self.state.last_sweep = sweep
            self.state.add_log("Sweep PSNR terminé")
            self.refresh()
            messagebox.showinfo("Succès", "Sweep terminé.")
        except Exception as exc:
            messagebox.showerror("Erreur", str(exc))

    def refresh(self) -> None:
        sweep = self.state.last_sweep
        for item in self.tree.get_children():
            self.tree.delete(item)
        if not sweep:
            clear_ttk_label_image(self.chart_label, "Aucun sweep pour le moment.")
            return

        fig = build_sweep_figure(sweep["ratios"], sweep["psnr_by_method"])
        self.photo = figure_to_photo(fig)
        self.chart_label.configure(image=self.photo)

        ratios = list(sweep["ratios"])
        for method, values in sweep["psnr_by_method"].items():
            for ratio, psnr in zip(ratios, values):
                self.tree.insert("", "end", values=(ratio, method.upper(), f"{psnr:.2f}"))
