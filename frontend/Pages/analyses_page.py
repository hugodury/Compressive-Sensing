"""
Onglet unique : comparaisons (sweep PSNR), tableaux M / cohérence / erreurs,
aperçus graphiques et estimation CO₂ de session.
"""

from __future__ import annotations

import csv
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from main import run_pipeline, setupParam
from frontend.utils import (
    DICTIONARY_COMBO_TEXT,
    DICTIONARY_COMBO_TO_KEY,
    PHI_COURS_KEYS,
    PHI_COURS_RADIO_LABELS,
    SOLVER_UI_CHOICES,
    build_section6_mp_coherence_figure,
    build_dico_comparison_table,
    build_sparsity_figure,
    build_sweep_figure,
    clear_ttk_label_image,
    dictionary_key_from_combo_selection,
    figure_to_photo,
    format_empreinte_pour_ui,
    open_path,
    parse_float_list,
    parse_int,
)
from .base_page import BasePage


def _read_csv_rows(path: Path) -> list[list[str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


class AnalysesPage(BasePage):
    # Indices du notebook de droite (ordre des add())
    _TAB_SWEEP_GRAPH = 0
    _TAB_SECTION6_GRAPH = 1
    _TAB_SYNTH_SWEEP = 2
    _TAB_CSV_TABLE = 3
    _TAB_CSV_RAW = 4
    _TAB_SPARSITY = 5
    _TAB_DICO_COMPARE = 6

    def __init__(self, parent: tk.Misc, app) -> None:
        super().__init__(parent, app)
        self.photo = None
        self.photo_s6 = None
        self.photo_sparsity = None
        self._csv_paths_cache: list[Path] = []
        self.vars_sweep = {
            "ratios": tk.StringVar(value="15,25,50,75"),
            "block_size": tk.StringVar(value="8"),
            "ratio": tk.StringVar(value="25"),
            "dictionary_type": tk.StringVar(),
            "measurement_mode": tk.StringVar(value="phi4"),
            "output_path": tk.StringVar(value=self.state.output_path),
            "image_path": tk.StringVar(value=self.state.image_path),
            "seed": tk.StringVar(value="0"),
            "empreinte_w": tk.StringVar(value="45"),
            "empreinte_g": tk.StringVar(value="85"),
        }
        for disp, key in DICTIONARY_COMBO_TO_KEY.items():
            if key == "dct":
                self.vars_sweep["dictionary_type"].set(disp)
                break
        else:
            self.vars_sweep["dictionary_type"].set(next(iter(DICTIONARY_COMBO_TO_KEY)))
        self.vars_sweep_emp = tk.BooleanVar(value=True)
        self.method_vars: dict[str, tk.BooleanVar] = {}

        self.vars_s6 = {
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
        ttk.Label(
            self,
            text="Analyses — comparaisons, cohérence & erreurs",
            style="Title.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            self,
            text=(
                "Courbes PSNR vs ratio (plusieurs méthodes), export CSV M(P), μ(Φ,D), erreurs relatives "
                "(trois vecteurs de test), et synthèse CO₂ estimée après chaque calcul."
            ),
            style="Muted.TLabel",
            wraplength=980,
            justify="left",
        ).pack(anchor="w", pady=(6, 12))

        lf_sw = ttk.LabelFrame(self, text=" Comparaisons — sweep PSNR vs ratios ", padding=14)
        lf_sw.pack(fill="x", pady=(0, 10))
        grid_sw = ttk.Frame(lf_sw, style="Card.TFrame")
        grid_sw.pack(fill="x")
        self._entry(grid_sw, self.vars_sweep, 0, "Image", "image_path")
        self._entry(grid_sw, self.vars_sweep, 1, "Ratios à tester", "ratios")
        self._entry(grid_sw, self.vars_sweep, 2, "B", "block_size")
        self._entry(grid_sw, self.vars_sweep, 3, "Ratio de base (M)", "ratio")
        self._entry(grid_sw, self.vars_sweep, 4, "Seed", "seed")
        self._entry(grid_sw, self.vars_sweep, 5, "Sortie", "output_path")
        self._combo(grid_sw, self.vars_sweep, 6, "Dictionnaire", "dictionary_type", list(DICTIONARY_COMBO_TEXT))

        phi_lf = ttk.LabelFrame(grid_sw, text=" Φ (une famille) ", padding=8)
        phi_lf.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(10, 6))
        pg = ttk.Frame(phi_lf, style="Card.TFrame")
        pg.pack(fill="x")
        for i, pk in enumerate(PHI_COURS_KEYS):
            ttk.Radiobutton(
                pg,
                text=PHI_COURS_RADIO_LABELS[pk],
                variable=self.vars_sweep["measurement_mode"],
                value=pk,
            ).grid(row=i // 2, column=i % 2, sticky="nw", padx=(0, 12), pady=3)

        emp_fr = ttk.Frame(grid_sw, style="Card.TFrame")
        emp_fr.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Checkbutton(emp_fr, text="Empreinte carbone (session)", variable=self.vars_sweep_emp).pack(anchor="w")
        ttk.Label(emp_fr, text="W", style="CardBody.TLabel").pack(side="left", padx=(12, 4))
        ttk.Entry(emp_fr, textvariable=self.vars_sweep["empreinte_w"], width=6).pack(side="left")
        ttk.Label(emp_fr, text="g CO₂/kWh", style="CardBody.TLabel").pack(side="left", padx=(12, 4))
        ttk.Entry(emp_fr, textvariable=self.vars_sweep["empreinte_g"], width=6).pack(side="left")

        methods = ttk.Frame(grid_sw, style="Card.TFrame", padding=(8, 0))
        methods.grid(row=0, column=2, rowspan=10, sticky="ne", padx=(16, 0))
        ttk.Label(methods, text="Méthodes", style="CardTitle.TLabel").pack(anchor="w")
        mg = ttk.Frame(methods, style="Card.TFrame")
        mg.pack(fill="x", pady=(6, 0))
        for i, (method, caption) in enumerate(SOLVER_UI_CHOICES):
            var = tk.BooleanVar(value=method in {"omp", "cosamp"})
            self.method_vars[method] = var
            r, c = divmod(i, 2)
            ttk.Checkbutton(mg, text=caption, variable=var).grid(row=r, column=c, sticky="w", padx=4, pady=3)

        ttk.Button(grid_sw, text="Lancer le sweep", style="Primary.TButton", command=self.run_sweep).grid(
            row=10, column=0, columnspan=2, sticky="ew", pady=(14, 0)
        )
        grid_sw.columnconfigure(1, weight=1)

        lf6 = ttk.LabelFrame(self, text=" Tableaux — M(P), cohérence mutuelle, erreurs relatives ", padding=14)
        lf6.pack(fill="x", pady=(0, 10))
        f6 = ttk.Frame(lf6, style="Card.TFrame")
        f6.pack(fill="x")
        self._entry(f6, self.vars_s6, 0, "Image", "image_path")
        self._entry(f6, self.vars_s6, 1, "Taille B", "block_size")
        self._entry(f6, self.vars_s6, 2, "Ratio de base", "ratio")
        self._entry(f6, self.vars_s6, 3, "Sortie", "output_path")
        self._entry(f6, self.vars_s6, 4, "Seed", "seed")
        self._entry(f6, self.vars_s6, 5, "max_iter (tableaux)", "max_iter")
        ttk.Label(
            f6,
            text="DCT tronquée, Φ₁…Φ₄, MP / OMP / StOMP / CoSaMP / IRLS. Erreurs : trois vecteurs de test (colonne « vecteur » dans le CSV).",
            style="Hint.TLabel",
            wraplength=720,
            justify="left",
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Checkbutton(f6, text="Inclure erreurs_relatives.csv", variable=self.vars_s6["with_errors"]).grid(
            row=7, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        ttk.Button(f6, text="Générer les tableaux", style="Primary.TButton", command=self.generate_tables).grid(
            row=8, column=0, columnspan=2, sticky="ew", pady=(12, 0)
        )
        f6.columnconfigure(1, weight=1)

        self.emp_line = ttk.Label(self, text="", style="CardMuted.TLabel", wraplength=960, justify="left")
        self.emp_line.pack(anchor="w", pady=(0, 8))
        self._refresh_empreinte_line()

        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill="both", expand=True, pady=(4, 0))

        left = ttk.Frame(paned, style="Card.TFrame", padding=12)
        paned.add(left, weight=0)
        ttk.Label(left, text="Fichiers CSV exportés", style="CardTitle.TLabel").pack(anchor="w")
        self.files_list = tk.Listbox(left, height=12, width=34, exportselection=False)
        self.files_list.pack(fill="both", expand=True, pady=(8, 8))
        self.files_list.bind("<<ListboxSelect>>", self._on_file_select)
        ttk.Button(left, text="Ouvrir le dossier d’export", command=self.open_folder_s6).pack(fill="x")

        right = ttk.Frame(paned, style="Card.TFrame", padding=8)
        paned.add(right, weight=1)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.view_nb = ttk.Notebook(right)
        self.view_nb.grid(row=0, column=0, sticky="nsew")

        tab_graph = ttk.Frame(self.view_nb, style="Card.TFrame", padding=8)
        self.view_nb.add(tab_graph, text="Graphique sweep")
        tab_graph.columnconfigure(0, weight=1)
        tab_graph.rowconfigure(1, weight=1)
        ttk.Label(tab_graph, text="Courbe PSNR (après « Lancer le sweep »)", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.chart_label = ttk.Label(tab_graph, style="CardBody.TLabel")
        self.chart_label.grid(row=1, column=0, sticky="nsew", pady=(8, 0))

        tab_s6 = ttk.Frame(self.view_nb, style="Card.TFrame", padding=8)
        self.view_nb.add(tab_s6, text="Graphiques M et μ")
        tab_s6.columnconfigure(0, weight=1)
        tab_s6.rowconfigure(1, weight=1)
        ttk.Label(
            tab_s6,
            text="M(P) et cohérence mutuelle μ(Φ,D) pour Φ₁…Φ₄ (après « Générer les tableaux »)",
            style="CardTitle.TLabel",
        ).grid(row=0, column=0, sticky="w")
        self.chart_label_s6 = ttk.Label(tab_s6, style="CardBody.TLabel")
        self.chart_label_s6.grid(row=1, column=0, sticky="nsew", pady=(8, 0))

        tab_sum = ttk.Frame(self.view_nb, style="Card.TFrame", padding=8)
        self.view_nb.add(tab_sum, text="Synthèse sweep")
        ttk.Label(tab_sum, text="Tableau PSNR par ratio et méthode", style="CardTitle.TLabel").pack(anchor="w")
        cols = ("ratio", "methode", "psnr")
        self.tree = ttk.Treeview(tab_sum, columns=cols, show="headings", height=12)
        for col, text in zip(cols, ["Ratio", "Méthode", "PSNR"]):
            self.tree.heading(col, text=text)
        self.tree.pack(fill="both", expand=True, pady=(8, 0))

        tab_table = ttk.Frame(self.view_nb, style="Card.TFrame", padding=4)
        self.view_nb.add(tab_table, text="Vue tableau CSV")
        tab_table.rowconfigure(0, weight=1)
        tab_table.columnconfigure(0, weight=1)
        self.tree_container = ttk.Frame(tab_table, style="Card.TFrame")
        self.tree_container.grid(row=0, column=0, sticky="nsew")
        ttk.Label(
            self.tree_container,
            text="Sélectionnez un CSV dans la liste de gauche.",
            style="CardMuted.TLabel",
        ).pack(anchor="center", pady=40)

        tab_text = ttk.Frame(self.view_nb, style="Card.TFrame", padding=4)
        self.view_nb.add(tab_text, text="Fichier CSV brut")
        tab_text.rowconfigure(0, weight=1)
        tab_text.columnconfigure(0, weight=1)
        txt_frame = ttk.Frame(tab_text, style="Card.TFrame")
        txt_frame.grid(row=0, column=0, sticky="nsew")
        txt_frame.rowconfigure(0, weight=1)
        txt_frame.columnconfigure(0, weight=1)
        self.preview = tk.Text(txt_frame, wrap="none", height=16, font=("Ubuntu Mono", 9))
        self.preview.grid(row=0, column=0, sticky="nsew")
        vsb = ttk.Scrollbar(txt_frame, orient="vertical", command=self.preview.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        self.preview.configure(yscrollcommand=vsb.set)
        hsb = ttk.Scrollbar(tab_text, orient="horizontal", command=self.preview.xview)
        hsb.grid(row=1, column=0, sticky="ew")
        self.preview.configure(xscrollcommand=hsb.set)

        # Onglet Parcimonie
        tab_sparsity = ttk.Frame(self.view_nb, style="Card.TFrame", padding=8)
        self.view_nb.add(tab_sparsity, text="Parcimonie")
        tab_sparsity.columnconfigure(0, weight=1)
        tab_sparsity.rowconfigure(1, weight=1)
        ttk.Label(
            tab_sparsity,
            text="‖α‖₀ moyen par méthode, moyenné sur tous les patchs reconstruits.",
            style="CardMuted.TLabel",
            wraplength=600,
            justify="left",
        ).grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.sparsity_label = ttk.Label(tab_sparsity, style="CardBody.TLabel", anchor="center")
        self.sparsity_label.grid(row=1, column=0, sticky="nsew")

        tab_dico = ttk.Frame(self.view_nb, style="Card.TFrame", padding=8)
        self.view_nb.add(tab_dico, text="Comparaison dicos")
        tab_dico.columnconfigure(0, weight=1)
        tab_dico.rowconfigure(1, weight=1)
        self._dico_header = ttk.Label(
            tab_dico,
            text="Faites deux reconstructions successives puis venez ici.",
            style="CardMuted.TLabel",
            wraplength=700,
            justify="left",
        )
        self._dico_header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self._dico_tree_frame = ttk.Frame(tab_dico, style="Card.TFrame")
        self._dico_tree_frame.grid(row=1, column=0, sticky="nsew")
        self._dico_tree_frame.columnconfigure(0, weight=1)
        self._dico_tree_frame.rowconfigure(0, weight=1)
        self._dico_tree = None

        self.refresh()

    def _entry(self, parent: ttk.Frame, vars_d: dict, row: int, label: str, key: str) -> None:
        ttk.Label(parent, text=label, style="CardBody.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
        ttk.Entry(parent, textvariable=vars_d[key], width=28).grid(row=row, column=1, sticky="ew", pady=4)

    def _combo(self, parent: ttk.Frame, vars_d: dict, row: int, label: str, key: str, values: list[str]) -> None:
        ttk.Label(parent, text=label, style="CardBody.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
        ttk.Combobox(parent, textvariable=vars_d[key], values=values, state="readonly", width=62).grid(
            row=row, column=1, sticky="ew", pady=4
        )

    def _refresh_empreinte_line(self) -> None:
        emp = self.state.last_analyses_empreinte
        main, detail = format_empreinte_pour_ui(emp)
        self.emp_line.configure(text=f"CO₂eq (dernière analyse sur cet onglet) : {main} — {detail}")

    def run_sweep(self) -> None:
        try:
            methods = [m for m, v in self.method_vars.items() if v.get()]
            if not methods:
                raise ValueError("Sélectionnez au moins une méthode.")

            params = setupParam(
                image_path=self.vars_sweep["image_path"].get().strip(),
                block_size=int(self.vars_sweep["block_size"].get()),
                ratio=float(self.vars_sweep["ratio"].get()),
                methodes=methods,
                dictionary_type=dictionary_key_from_combo_selection(self.vars_sweep["dictionary_type"].get()),
                measurement_mode=self.vars_sweep["measurement_mode"].get().strip(),
                output_path=self.vars_sweep["output_path"].get().strip(),
                seed=parse_int(self.vars_sweep["seed"].get(), 0),
                empreinte_carbone=bool(self.vars_sweep_emp.get()),
                empreinte_afficher_console=False,
                empreinte_puissance_w=float(self.vars_sweep["empreinte_w"].get().replace(",", ".") or 45),
                empreinte_g_co2_par_kwh=float(self.vars_sweep["empreinte_g"].get().replace(",", ".") or 85),
            )
            out = run_pipeline(
                params,
                etapes=("sweep_graph",),
                sweep_ratios=parse_float_list(self.vars_sweep["ratios"].get()),
            )
            self.state.last_sweep = out["sweep_graphique"]
            self.state.last_analyses_empreinte = out.get("empreinte_session")
            self.state.add_log("Sweep PSNR terminé (onglet Analyses)")
            self._refresh_empreinte_line()
            self.refresh()
            self.view_nb.select(self._TAB_SWEEP_GRAPH)
            messagebox.showinfo("Succès", "Sweep terminé — voir l’onglet « Graphique sweep ».")
        except Exception as exc:
            messagebox.showerror("Erreur", str(exc))

    def generate_tables(self) -> None:
        try:
            params = setupParam(
                image_path=self.vars_s6["image_path"].get().strip(),
                block_size=int(self.vars_s6["block_size"].get()),
                ratio=float(self.vars_s6["ratio"].get()),
                methodes=["omp"],
                dictionary_type="dct",
                output_path=self.vars_s6["output_path"].get().strip(),
                seed=int(self.vars_s6["seed"].get()),
                empreinte_carbone=bool(self.vars_sweep_emp.get()),
                empreinte_afficher_console=False,
                empreinte_puissance_w=float(self.vars_sweep["empreinte_w"].get().replace(",", ".") or 45),
                empreinte_g_co2_par_kwh=float(self.vars_sweep["empreinte_g"].get().replace(",", ".") or 85),
            )
            out = run_pipeline(
                params,
                etapes=("tableaux_s6",),
                tableaux_avec_erreurs=bool(self.vars_s6["with_errors"].get()),
                tableaux_max_iter=int(self.vars_s6["max_iter"].get()),
            )
            self.state.last_section6_dir = out["dossier_tableaux_section6"]
            self.state.last_analyses_empreinte = out.get("empreinte_session")
            self.state.add_log(f"Tableaux (M, μ, erreurs) générés → {self.state.last_section6_dir}")
            self._refresh_empreinte_line()
            self.refresh(after_generate=True)
            self.view_nb.select(self._TAB_SECTION6_GRAPH)
            messagebox.showinfo("Succès", "Tableaux générés — voir « Graphiques M et μ » et la liste CSV à gauche.")
        except Exception as exc:
            messagebox.showerror("Erreur", str(exc))

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
        self.preview.delete("1.0", tk.END)
        for r in rows:
            self.preview.insert(tk.END, " ; ".join(r) + "\n")

        self._clear_tree_container()
        if not rows:
            ttk.Label(self.tree_container, text="Fichier vide.", style="CardMuted.TLabel").pack(pady=20)
            self.view_nb.select(self._TAB_CSV_RAW)
            return
        headers = rows[0]
        data = rows[1:]
        ncols = len(headers)
        if ncols == 0:
            ttk.Label(self.tree_container, text="Pas d’en-têtes.", style="CardMuted.TLabel").pack(pady=20)
            self.view_nb.select(self._TAB_CSV_RAW)
            return

        col_ids = tuple(f"c{i}" for i in range(ncols))
        tree = ttk.Treeview(
            self.tree_container,
            columns=col_ids,
            show="headings",
            height=min(18, max(8, len(data) + 1)),
        )
        for i, h in enumerate(headers):
            cid = col_ids[i]
            tree.heading(cid, text=h[:28] + ("…" if len(h) > 28 else ""))
            hl = h.strip().lower()
            if h.startswith("P_"):
                w = 100
            elif hl == "vecteur":
                w = 64
            else:
                w = 140
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
        self.view_nb.select(self._TAB_CSV_TABLE)

    def _find_section6_csv(self, root: Path, filename: str) -> Path | None:
        direct = root / filename
        if direct.is_file():
            return direct
        for p in root.rglob(filename):
            if p.is_file():
                return p
        return None

    def _update_section6_charts(self, folder: Path | None) -> None:
        if not folder or not folder.is_dir():
            clear_ttk_label_image(
                self.chart_label_s6,
                "Générez les tableaux pour afficher M(P) et μ(Φ,D) pour les quatre Φ.",
            )
            return
        mp_path = self._find_section6_csv(folder, "M_pour_P.csv")
        coh_path = self._find_section6_csv(folder, "coherence_mutuelle.csv")
        mp_rows = _read_csv_rows(mp_path) if mp_path else None
        coh_rows = _read_csv_rows(coh_path) if coh_path else None
        if mp_rows is None and coh_rows is None:
            clear_ttk_label_image(
                self.chart_label_s6,
                "Aucun M_pour_P.csv / coherence_mutuelle.csv dans le dossier d’export.",
            )
            return
        try:
            fig = build_section6_mp_coherence_figure(mp_rows, coh_rows)
            self.photo_s6 = figure_to_photo(fig)
            self.chart_label_s6.configure(image=self.photo_s6)
        except OSError as e:
            clear_ttk_label_image(self.chart_label_s6, f"Lecture CSV : {e}")

    def refresh(self, *, after_generate: bool = False) -> None:
        sweep = self.state.last_sweep
        for item in self.tree.get_children():
            self.tree.delete(item)
        if sweep:
            fig = build_sweep_figure(sweep["ratios"], sweep["psnr_by_method"])
            self.photo = figure_to_photo(fig)
            self.chart_label.configure(image=self.photo)
            ratios = list(sweep["ratios"])
            for method, values in sweep["psnr_by_method"].items():
                for ratio, psnr in zip(ratios, values):
                    self.tree.insert("", "end", values=(ratio, method.upper(), f"{psnr:.2f}"))
        else:
            clear_ttk_label_image(self.chart_label, "Aucun sweep — lancez « Lancer le sweep ».")

        self.files_list.delete(0, tk.END)
        self._csv_paths_cache = []
        folder = self.state.last_section6_dir
        self._update_section6_charts(Path(folder) if folder else None)
        if folder and Path(folder).is_dir():
            root = Path(folder)
            csv_paths = sorted(root.rglob("*.csv"))
            self._csv_paths_cache = csv_paths
            for path in csv_paths:
                try:
                    rel = path.relative_to(root)
                except ValueError:
                    rel = path.name
                self.files_list.insert(tk.END, str(rel))
            if after_generate and csv_paths:
                preferred = 0
                for i, p in enumerate(csv_paths):
                    if "coherence" in p.name.lower():
                        preferred = i
                        break
                else:
                    for i, p in enumerate(csv_paths):
                        n = p.name.lower()
                        if "erreurs" in n and "relatives" in n:
                            preferred = i
                            break
                        if "m_pour" in n:
                            preferred = i
                self.files_list.selection_clear(0, tk.END)
                self.files_list.selection_set(preferred)
                self.files_list.see(preferred)
                self._load_csv_path(csv_paths[preferred])

        # Diagramme de parcimonie
        result = self.state.last_result
        alphas = result.get("alphas_by_method", {}) if result else {}
        if alphas:
            fig_sp = build_sparsity_figure(alphas)
            self.photo_sparsity = figure_to_photo(fig_sp)
            self.sparsity_label.configure(image=self.photo_sparsity, text="")
        else:
            clear_ttk_label_image(
                self.sparsity_label,
                "Lancez une reconstruction pour afficher le diagramme.",
            )

        res_b = self.state.last_result
        res_a = self.state.last_result_prev
        for w in self._dico_tree_frame.winfo_children():
            w.destroy()
        self._dico_tree = None
        if res_a and res_b:
            try:
                cols, rows, label_a, label_b = build_dico_comparison_table(res_a, res_b)
                self._dico_header.configure(text=f"A : {label_a}    vs    B : {label_b}")
                tree = ttk.Treeview(
                    self._dico_tree_frame,
                    columns=list(range(len(cols))),
                    show="headings",
                    height=10,
                )
                col_widths = [72, 72, 72, 80, 72, 72, 72, 72, 72, 72, 72, 72]
                for i, (col, w) in enumerate(zip(cols, col_widths)):
                    tree.heading(i, text=col)
                    tree.column(i, width=w, anchor="center", minwidth=55)
                tree.column(0, width=80, anchor="w")
                for row in rows:
                    tag = ""
                    try:
                        delta = float(row[3])
                        tag = "better" if delta > 0 else ("worse" if delta < 0 else "")
                    except (ValueError, TypeError):
                        pass
                    tree.insert("", "end", values=row, tags=(tag,))
                tree.tag_configure("better", foreground="#16a34a")
                tree.tag_configure("worse", foreground="#dc2626")
                hsb = ttk.Scrollbar(self._dico_tree_frame, orient="horizontal", command=tree.xview)
                tree.configure(xscrollcommand=hsb.set)
                tree.grid(row=0, column=0, sticky="nsew")
                hsb.grid(row=1, column=0, sticky="ew")
                self._dico_tree = tree
            except Exception as e:
                ttk.Label(
                    self._dico_tree_frame,
                    text=f"Erreur : {e}",
                    style="CardMuted.TLabel",
                ).grid(row=0, column=0, pady=20)
        else:
            self._dico_header.configure(
                text="Faites deux reconstructions successives pour comparer les dictionnaires."
            )
            ttk.Label(
                self._dico_tree_frame,
                text="Aucune comparaison — lancez deux reconstructions successives.",
                style="CardMuted.TLabel",
            ).grid(row=0, column=0, pady=20)

    def open_folder_s6(self) -> None:
        open_path(self.state.last_section6_dir or "")