from __future__ import annotations

import csv
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from main import run_pipeline, setupParam
from frontend.utils import open_path
from .base_page import BasePage


class Section6Page(BasePage):
    def __init__(self, parent: tk.Misc, app) -> None:
        super().__init__(parent, app)
        self.vars = {
            "image_path": tk.StringVar(value=self.state.image_path),
            "block_size": tk.StringVar(value="8"),
            "ratio": tk.StringVar(value="25"),
            "output_path": tk.StringVar(value=self.state.output_path),
            "seed": tk.StringVar(value="0"),
            "max_iter": tk.StringVar(value="80"),
            "with_errors": tk.BooleanVar(value=True),
        }
        self._build()

    def _build(self) -> None:
        ttk.Label(self, text="Tableaux de la section 6", style="Title.TLabel").pack(anchor="w")
        ttk.Label(self, text="Génère les CSV attendus dans le sujet : M pour P, cohérence mutuelle, erreurs relatives.", style="Muted.TLabel").pack(anchor="w", pady=(6, 12))

        form = ttk.Frame(self, style="Card.TFrame", padding=16)
        form.pack(fill="x")
        self._entry(form, 0, "Image", "image_path")
        self._entry(form, 1, "Taille B", "block_size")
        self._entry(form, 2, "Ratio de base", "ratio")
        self._entry(form, 3, "Sortie", "output_path")
        self._entry(form, 4, "Seed", "seed")
        self._entry(form, 5, "max_iter tableau", "max_iter")
        ttk.Checkbutton(form, text="Inclure les erreurs relatives", variable=self.vars["with_errors"]).grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(form, text="Générer les tableaux", style="Primary.TButton", command=self.generate_tables).grid(row=7, column=0, columnspan=2, sticky="ew", pady=(14, 0))

        bottom = ttk.Frame(self, style="App.TFrame")
        bottom.pack(fill="both", expand=True, pady=(10, 0))
        left = ttk.Frame(bottom, style="Card.TFrame", padding=16)
        left.pack(side="left", fill="y")
        ttk.Label(left, text="Fichiers générés", style="CardTitle.TLabel").pack(anchor="w")
        self.files_list = tk.Listbox(left, height=14)
        self.files_list.pack(fill="both", expand=True, pady=(10, 8))
        ttk.Button(left, text="Prévisualiser", command=self.preview_selected).pack(fill="x")
        ttk.Button(left, text="Ouvrir le dossier", command=self.open_folder).pack(fill="x", pady=(6, 0))

        right = ttk.Frame(bottom, style="Card.TFrame", padding=16)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))
        ttk.Label(right, text="Aperçu CSV", style="CardTitle.TLabel").pack(anchor="w")
        self.preview = tk.Text(right, wrap="none", height=22)
        self.preview.pack(fill="both", expand=True, pady=(10, 0))

    def _entry(self, parent: ttk.Frame, row: int, label: str, key: str) -> None:
        ttk.Label(parent, text=label, style="App.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
        ttk.Entry(parent, textvariable=self.vars[key], width=32).grid(row=row, column=1, sticky="ew", pady=4)
        parent.columnconfigure(1, weight=1)

    def generate_tables(self) -> None:
        try:
            params = setupParam(
                image_path=self.vars["image_path"].get().strip(),
                block_size=int(self.vars["block_size"].get()),
                ratio=float(self.vars["ratio"].get()),
                methodes=["mp", "omp", "stomp", "cosamp", "irls"],
                dictionary_type="dct",
                output_path=self.vars["output_path"].get().strip(),
                seed=int(self.vars["seed"].get()),
            )
            out = run_pipeline(
                params,
                etapes=("tableaux_s6",),
                tableaux_avec_erreurs=bool(self.vars["with_errors"].get()),
                tableaux_max_iter=int(self.vars["max_iter"].get()),
            )
            self.state.last_section6_dir = out["dossier_tableaux_section6"]
            self.state.add_log(f"Section 6 générée dans {self.state.last_section6_dir}")
            self.refresh()
            messagebox.showinfo("Succès", "Tableaux générés.")
        except Exception as exc:
            messagebox.showerror("Erreur", str(exc))

    def refresh(self) -> None:
        self.files_list.delete(0, tk.END)
        folder = self.state.last_section6_dir
        if not folder:
            return
        for path in sorted(Path(folder).glob("*.csv")):
            self.files_list.insert(tk.END, str(path))

    def preview_selected(self) -> None:
        sel = self.files_list.curselection()
        if not sel:
            return
        path = self.files_list.get(sel[0])
        self.preview.delete("1.0", tk.END)
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                self.preview.insert(tk.END, " ; ".join(row) + "\n")
                if i > 30:
                    break

    def open_folder(self) -> None:
        open_path(self.state.last_section6_dir)
