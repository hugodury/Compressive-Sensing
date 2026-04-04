from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from main import run_pipeline, setupParam
from frontend.utils import build_sweep_figure, figure_to_photo, parse_float_list, parse_int
from .base_page import BasePage


class ComparisonPage(BasePage):
    def __init__(self, parent: tk.Misc, app) -> None:
        super().__init__(parent, app)
        self.photo = None
        self.vars = {
            "ratios": tk.StringVar(value="15,25,50,75"),
            "block_size": tk.StringVar(value="8"),
            "ratio": tk.StringVar(value="25"),
            "dictionary_type": tk.StringVar(value="dct"),
            "measurement_mode": tk.StringVar(value="gaussian"),
            "output_path": tk.StringVar(value=self.state.output_path),
            "image_path": tk.StringVar(value=self.state.image_path),
            "seed": tk.StringVar(value="0"),
        }
        self.method_vars: dict[str, tk.BooleanVar] = {}
        self._build()

    def _build(self) -> None:
        ttk.Label(self, text="Comparaisons et sweep de ratios", style="Title.TLabel").pack(anchor="w")
        ttk.Label(self, text="Comparer les méthodes sur plusieurs ratios de mesures et générer la courbe PSNR.", style="Muted.TLabel").pack(anchor="w", pady=(6, 12))

        top = ttk.Frame(self, style="App.TFrame")
        top.pack(fill="x")
        form = ttk.Frame(top, style="Card.TFrame", padding=16)
        form.pack(fill="x")

        self._entry(form, 0, "Image", "image_path")
        self._entry(form, 1, "Ratios à tester", "ratios")
        self._entry(form, 2, "B", "block_size")
        self._entry(form, 3, "Ratio de base", "ratio")
        self._entry(form, 4, "Seed", "seed")
        self._entry(form, 5, "Sortie", "output_path")
        self._combo(form, 6, "Dictionnaire", "dictionary_type", ["dct", "mixte", "ksvd", "ksvd_dct", "ksvd_mixte"])
        self._combo(form, 7, "Matrice de mesure", "measurement_mode", ["gaussian", "uniform", "bernoulli_1", "bernoulli_01"])

        methods = ttk.Frame(form, style="Card.TFrame")
        methods.grid(row=0, column=2, rowspan=8, sticky="n", padx=(16, 0))
        ttk.Label(methods, text="Méthodes", style="CardTitle.TLabel").pack(anchor="w")
        for method in ["mp", "omp", "stomp", "cosamp", "irls"]:
            var = tk.BooleanVar(value=method in {"omp", "cosamp"})
            self.method_vars[method] = var
            ttk.Checkbutton(methods, text=method.upper(), variable=var).pack(anchor="w", pady=3)

        ttk.Button(form, text="Lancer le sweep", style="Primary.TButton", command=self.run_sweep).grid(row=8, column=0, columnspan=2, sticky="ew", pady=(16, 0))

        bottom = ttk.Frame(self, style="App.TFrame")
        bottom.pack(fill="both", expand=True, pady=(10, 0))
        left = ttk.Frame(bottom, style="Card.TFrame", padding=16)
        left.pack(side="left", fill="both", expand=True)
        ttk.Label(left, text="Courbe PSNR", style="CardTitle.TLabel").pack(anchor="w")
        self.chart_label = ttk.Label(left)
        self.chart_label.pack(fill="both", expand=True, pady=(10, 0))

        right = ttk.Frame(bottom, style="Card.TFrame", padding=16)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))
        ttk.Label(right, text="Tableau résumé", style="CardTitle.TLabel").pack(anchor="w")
        cols = ("ratio", "methode", "psnr")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=14)
        for col, text in zip(cols, ["Ratio", "Méthode", "PSNR"]):
            self.tree.heading(col, text=text)
        self.tree.pack(fill="both", expand=True, pady=(10, 0))

    def _entry(self, parent: ttk.Frame, row: int, label: str, key: str) -> None:
        ttk.Label(parent, text=label, style="App.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
        ttk.Entry(parent, textvariable=self.vars[key], width=28).grid(row=row, column=1, sticky="ew", pady=4)
        parent.columnconfigure(1, weight=1)

    def _combo(self, parent: ttk.Frame, row: int, label: str, key: str, values: list[str]) -> None:
        ttk.Label(parent, text=label, style="App.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
        ttk.Combobox(parent, textvariable=self.vars[key], values=values, state="readonly").grid(row=row, column=1, sticky="ew", pady=4)

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
                dictionary_type=self.vars["dictionary_type"].get(),
                measurement_mode=self.vars["measurement_mode"].get(),
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
            self.chart_label.configure(image="", text="Aucun sweep pour le moment.")
            return

        fig = build_sweep_figure(sweep["ratios"], sweep["psnr_by_method"])
        self.photo = figure_to_photo(fig)
        self.chart_label.configure(image=self.photo)

        ratios = list(sweep["ratios"])
        for method, values in sweep["psnr_by_method"].items():
            for ratio, psnr in zip(ratios, values):
                self.tree.insert("", "end", values=(ratio, method.upper(), f"{psnr:.2f}"))
