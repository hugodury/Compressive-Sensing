from __future__ import annotations

import csv
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from main import run_pipeline, setupParam
from frontend.utils import open_path
from .base_page import BasePage


def _read_csv_rows(path: Path) -> list[list[str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


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
        self._csv_paths_cache: list[Path] = []
        self._build()

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

        ttk.Label(self, text="Cohérence mutuelle, mesures et erreurs", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            self,
            text="Export CSV : M pour chaque %, cohérence μ(Φ,D), erreurs relatives. Après génération, les tableaux s’affichent ci‑dessous (vue structurée + texte intégral).",
            style="Muted.TLabel",
            wraplength=980,
            justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(6, 12))

        form = ttk.Frame(self, style="Card.TFrame", padding=16)
        form.grid(row=2, column=0, sticky="ew", pady=(0, 0))
        self._entry(form, 0, "Image", "image_path")
        self._entry(form, 1, "Taille B (patchs carrés)", "block_size")
        self._entry(form, 2, "Ratio de base", "ratio")
        self._entry(form, 3, "Sortie", "output_path")
        self._entry(form, 4, "Seed", "seed")
        self._entry(form, 5, "max_iter tableau", "max_iter")
        ttk.Label(
            form,
            text=(
                "Quatre familles Φ₁…Φ₄, dictionnaire DCT tronqué, méthodes MP, OMP, StOMP, CoSaMP, IRLS. "
                "N = B² pour ces tableaux."
            ),
            style="Hint.TLabel",
            wraplength=720,
            justify="left",
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Checkbutton(form, text="Inclure les erreurs relatives", variable=self.vars["with_errors"]).grid(
            row=7, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        ttk.Button(form, text="Générer les tableaux", style="Primary.TButton", command=self.generate_tables).grid(
            row=8, column=0, columnspan=2, sticky="ew", pady=(14, 0)
        )
        form.columnconfigure(1, weight=1)

        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.grid(row=3, column=0, sticky="nsew", pady=(12, 0))

        left = ttk.Frame(paned, style="Card.TFrame", padding=12)
        paned.add(left, weight=0)
        ttk.Label(left, text="Fichiers CSV", style="CardTitle.TLabel").pack(anchor="w")
        self.files_list = tk.Listbox(left, height=16, width=36, exportselection=False)
        self.files_list.pack(fill="both", expand=True, pady=(8, 8))
        self.files_list.bind("<<ListboxSelect>>", self._on_file_select)
        ttk.Button(left, text="Ouvrir le dossier", command=self.open_folder).pack(fill="x")

        right = ttk.Frame(paned, style="Card.TFrame", padding=12)
        paned.add(right, weight=1)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.view_nb = ttk.Notebook(right)
        self.view_nb.grid(row=0, column=0, sticky="nsew")

        tab_table = ttk.Frame(self.view_nb, style="Card.TFrame", padding=4)
        self.view_nb.add(tab_table, text="Vue tableau")
        tab_table.rowconfigure(0, weight=1)
        tab_table.columnconfigure(0, weight=1)
        self.tree_container = ttk.Frame(tab_table, style="Card.TFrame")
        self.tree_container.grid(row=0, column=0, sticky="nsew")
        ttk.Label(
            self.tree_container,
            text="Sélectionnez un fichier dans la liste.",
            style="CardMuted.TLabel",
        ).pack(anchor="center", pady=40)

        tab_text = ttk.Frame(self.view_nb, style="Card.TFrame", padding=4)
        self.view_nb.add(tab_text, text="Fichier complet")
        tab_text.rowconfigure(0, weight=1)
        tab_text.columnconfigure(0, weight=1)
        txt_frame = ttk.Frame(tab_text, style="Card.TFrame")
        txt_frame.grid(row=0, column=0, sticky="nsew")
        txt_frame.rowconfigure(0, weight=1)
        txt_frame.columnconfigure(0, weight=1)
        self.preview = tk.Text(txt_frame, wrap="none", height=20, font=("Ubuntu Mono", 9))
        self.preview.grid(row=0, column=0, sticky="nsew")
        vsb = ttk.Scrollbar(txt_frame, orient="vertical", command=self.preview.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        self.preview.configure(yscrollcommand=vsb.set)
        hsb = ttk.Scrollbar(tab_text, orient="horizontal", command=self.preview.xview)
        hsb.grid(row=1, column=0, sticky="ew")
        self.preview.configure(xscrollcommand=hsb.set)

    def _entry(self, parent: ttk.Frame, row: int, label: str, key: str) -> None:
        ttk.Label(parent, text=label, style="App.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
        ttk.Entry(parent, textvariable=self.vars[key], width=32).grid(row=row, column=1, sticky="ew", pady=4)
        parent.columnconfigure(1, weight=1)

    def _on_file_select(self, _event: tk.Event | None = None) -> None:
        sel = self.files_list.curselection()
        if not sel:
            return
        idx = int(sel[0])
        if idx < 0 or idx >= len(self._csv_paths_cache):
            return
        self._load_csv_path(self._csv_paths_cache[idx])

    def _clear_tree_container(self) -> None:
        for w in self.tree_container.winfo_children():
            w.destroy()

    def _load_csv_path(self, path: Path) -> None:
        try:
            rows = _read_csv_rows(path)
        except OSError as e:
            messagebox.showerror("Lecture", str(e))
            return

        # Onglet texte : fichier entier
        self.preview.delete("1.0", tk.END)
        for r in rows:
            self.preview.insert(tk.END, " ; ".join(r) + "\n")

        # Onglet tableau
        self._clear_tree_container()
        if not rows:
            ttk.Label(self.tree_container, text="Fichier vide.", style="CardMuted.TLabel").pack(pady=20)
            self.view_nb.select(1)
            return

        headers = rows[0]
        data = rows[1:]
        ncols = len(headers)
        if ncols == 0:
            ttk.Label(self.tree_container, text="Pas d’en-têtes.", style="CardMuted.TLabel").pack(pady=20)
            self.view_nb.select(1)
            return

        col_ids = tuple(f"c{i}" for i in range(ncols))
        tree = ttk.Treeview(
            self.tree_container,
            columns=col_ids,
            show="headings",
            height=min(22, max(8, len(data) + 1)),
        )
        for i, h in enumerate(headers):
            cid = col_ids[i]
            tree.heading(cid, text=h[:28] + ("…" if len(h) > 28 else ""))
            w = 100 if h.startswith("P_") else 140
            tree.column(cid, width=w, anchor="center", stretch=True)

        for r in data:
            pad = r + [""] * ncols
            vals = tuple(str(pad[j]) for j in range(ncols))
            tree.insert("", "end", values=vals)

        yscroll = ttk.Scrollbar(self.tree_container, orient="vertical", command=tree.yview)
        xscroll = ttk.Scrollbar(self.tree_container, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        self.tree_container.rowconfigure(0, weight=1)
        self.tree_container.columnconfigure(0, weight=1)

        self.view_nb.select(0)

    def generate_tables(self) -> None:
        try:
            params = setupParam(
                image_path=self.vars["image_path"].get().strip(),
                block_size=int(self.vars["block_size"].get()),
                ratio=float(self.vars["ratio"].get()),
                methodes=["omp"],
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
            self.state.add_log(f"Tableaux cohérence / erreurs générés dans {self.state.last_section6_dir}")
            self.refresh(after_generate=True)
            messagebox.showinfo("Succès", "Tableaux générés — affichage mis à jour.")
        except Exception as exc:
            messagebox.showerror("Erreur", str(exc))

    def refresh(self, *, after_generate: bool = False) -> None:
        self.files_list.delete(0, tk.END)
        self._csv_paths_cache = []
        folder = self.state.last_section6_dir
        if not folder:
            return
        root = Path(folder)
        csv_paths = sorted(root.rglob("*.csv"))
        self._csv_paths_cache = csv_paths
        for path in csv_paths:
            try:
                rel = path.relative_to(root)
            except ValueError:
                rel = path.name
            self.files_list.insert(tk.END, str(rel))
        if not csv_paths and root.is_dir():
            self.state.add_log(f"Aucun CSV trouvé sous {root}.")

        if after_generate and csv_paths:
            preferred = 0
            for i, p in enumerate(csv_paths):
                n = p.name.lower()
                if "coherence" in n:
                    preferred = i
                    break
                if "m_pour" in n:
                    preferred = i
            self.files_list.selection_clear(0, tk.END)
            self.files_list.selection_set(preferred)
            self.files_list.see(preferred)
            self._load_csv_path(csv_paths[preferred])

    def open_folder(self) -> None:
        open_path(self.state.last_section6_dir)
