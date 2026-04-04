from __future__ import annotations

from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from main import main as run_main, run_coarse_best_search, setupParam
from backend.utils.save import save_results
from frontend.theme import SURFACE
from frontend.utils import (
    DICTIONARY_COMBO_TEXT,
    DICTIONARY_COMBO_TO_KEY,
    PHI_COURS_DETAIL,
    PHI_COURS_KEYS,
    PHI_COURS_RADIO_LABELS,
    SOLVER_METHOD_IDS,
    SOLVER_UI_CHOICES,
    UI_HELP_DICT_BLOC,
    UI_HELP_METHODS_BLOC,
    UI_HELP_PHI_BLOC,
    UI_HELP_RATIO_M,
    dictionary_key_from_combo_selection,
    latest_subdir,
    parse_float,
    parse_int,
)
from .base_page import BasePage


class ReconstructionPage(BasePage):
    """Paramètres de reconstruction : Φ₁–Φ₄, patchs carrés B×B, dictionnaire, solveurs, empreinte."""

    def __init__(self, parent: tk.Misc, app) -> None:
        super().__init__(parent, app)
        self.vars: dict[str, tk.Variable] = {}
        self.method_vars: dict[str, tk.BooleanVar] = {}
        self._build()
        self._set_defaults()

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        outer = ttk.Frame(self, style="App.TFrame")
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(0, weight=5)
        outer.columnconfigure(1, weight=2)
        outer.rowconfigure(0, weight=1)

        # — Colonne gauche : Canvas + barre de défilement toujours visible (tk.Scrollbar = plus large)
        left_wrap = ttk.Frame(outer, style="Panel.TFrame")
        left_wrap.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left_wrap.rowconfigure(0, weight=1)
        left_wrap.columnconfigure(0, weight=1)

        canvas = tk.Canvas(left_wrap, highlightthickness=0, bg=SURFACE, bd=0)
        vsb = tk.Scrollbar(
            left_wrap,
            orient="vertical",
            command=canvas.yview,
            width=18,
            troughcolor="#e2e8f0",
            bg="#cbd5e1",
            activebackground="#94a3b8",
            highlightthickness=0,
            borderwidth=0,
            relief="flat",
        )
        scroll_inner = ttk.Frame(canvas, style="Panel.TFrame")
        win_id = canvas.create_window((0, 0), window=scroll_inner, anchor="nw")

        def _scroll_cfg(_: tk.Event | None = None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _canvas_width(event: tk.Event) -> None:
            # Largeur du contenu = largeur utile du canvas (redimensionnement fenêtre)
            canvas.itemconfigure(win_id, width=max(1, event.width))

        scroll_inner.bind("<Configure>", _scroll_cfg)
        canvas.bind("<Configure>", _canvas_width)
        canvas.configure(yscrollcommand=vsb.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        left_wrap.columnconfigure(0, weight=1)

        # Molette : détection par position du pointeur (fiable Windows / Linux ; évite les autres onglets)
        topv = canvas.winfo_toplevel()
        scroll_zone_widgets = frozenset({canvas, scroll_inner, vsb})

        def _pointer_in_scroll_column() -> bool:
            try:
                x, y = topv.winfo_pointerxy()
            except tk.TclError:
                return False
            if x <= 0 or y <= 0:
                return False
            w = topv.winfo_containing(x, y)
            while w is not None:
                if w in scroll_zone_widgets:
                    return True
                w = getattr(w, "master", None)
            return False

        def _on_mousewheel(ev: tk.Event) -> None:
            if not _pointer_in_scroll_column():
                return
            if getattr(ev, "num", None) == 4 or (getattr(ev, "delta", 0) or 0) > 0:
                canvas.yview_scroll(-1, "units")
            elif getattr(ev, "num", None) == 5 or (getattr(ev, "delta", 0) or 0) < 0:
                canvas.yview_scroll(1, "units")

        topv.bind_all("<MouseWheel>", _on_mousewheel)
        topv.bind_all("<Button-4>", _on_mousewheel)
        topv.bind_all("<Button-5>", _on_mousewheel)

        # — Colonne droite : méthodes + actions
        right = ttk.Frame(outer, style="Panel.TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)

        # Variables communes
        for key in (
            "image_path",
            "dictionary_train_image_path",
            "output_path",
            "block_size",
            "ratio",
            "M_explicit",
            "n_atoms",
            "n_iter_ksvd",
            "seed",
            "max_patches",
            "max_iter",
            "epsilon",
            "t_stomp",
            "s_cosamp",
            "norm_p",
            "psnr_target_db",
            "lambda_lasso",
            "empreinte_puissance_w",
            "empreinte_g_co2_par_kwh",
        ):
            self.vars[key] = tk.StringVar()

        self.vars["measurement_mode"] = tk.StringVar()
        self.vars["dictionary_type"] = tk.StringVar()
        self.vars["psnr_stop"] = tk.BooleanVar(value=False)
        self.vars["auto_save"] = tk.BooleanVar(value=True)
        self.vars["s_cosamp_auto"] = tk.BooleanVar(value=False)
        self.vars["empreinte_carbone"] = tk.BooleanVar(value=True)

        # --- Section fichiers
        lf_files = ttk.LabelFrame(scroll_inner, text=" Fichiers & sortie ", padding=12)
        lf_files.pack(fill="x", pady=(0, 10))
        self._entry_row(lf_files, 0, "Image à reconstruire", "image_path", browse_file=True)
        self._entry_row(lf_files, 1, "Image entraînement dictionnaire (vide = même image)", "dictionary_train_image_path", browse_file=True)
        self._entry_row(lf_files, 2, "Dossier résultats", "output_path", browse_dir=True)
        ttk.Label(
            lf_files,
            text="Pour un dictionnaire appris (K-SVD, mixte), vous pouvez utiliser une image d’entraînement différente de l’image à reconstruire.",
            style="CardMuted.TLabel",
            wraplength=720,
            justify="left",
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(10, 0))

        # --- Découpage & mesures (patchs carrés B×B uniquement)
        lf_patch = ttk.LabelFrame(scroll_inner, text=" Patchs carrés B×B & mesures y = Φx ", padding=12)
        lf_patch.pack(fill="x", pady=(0, 10))
        self._simple_entry(lf_patch, 0, "Taille de bloc B (côté du patch carré)", self.vars["block_size"])
        self._simple_entry(lf_patch, 1, "Ratio (0.25 ou 25 %)", self.vars["ratio"])
        self._simple_entry(lf_patch, 2, "M mesures (optionnel, prioritaire sur ratio si rempli)", self.vars["M_explicit"])
        ttk.Label(lf_patch, text=UI_HELP_RATIO_M, style="CardMuted.TLabel", wraplength=720, justify="left").grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(6, 4)
        )
        ttk.Label(
            lf_patch,
            text=UI_HELP_PHI_BLOC,
            style="Hint.TLabel",
            wraplength=720,
            justify="left",
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(4, 4))
        phi_lf = ttk.LabelFrame(lf_patch, text=" Choix de Φ (1 parmi 4) ", padding=10)
        phi_lf.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(0, 6))
        phi_inner = ttk.Frame(phi_lf, style="Card.TFrame")
        phi_inner.pack(fill="x")
        for i, pk in enumerate(PHI_COURS_KEYS):
            ttk.Radiobutton(
                phi_inner,
                text=PHI_COURS_RADIO_LABELS[pk],
                variable=self.vars["measurement_mode"],
                value=pk,
            ).grid(row=i // 2, column=i % 2, sticky="nw", padx=(0, 20), pady=4)
        detail = "\n".join(f"• {PHI_COURS_DETAIL[k]}" for k in PHI_COURS_KEYS)
        ttk.Label(lf_patch, text=detail, style="CardMuted.TLabel", justify="left", wraplength=720).grid(
            row=6, column=0, columnspan=3, sticky="w", pady=(4, 0)
        )
        lf_patch.columnconfigure(1, weight=1)

        # --- Dictionnaire & K-SVD
        lf_dict = ttk.LabelFrame(scroll_inner, text=" Dictionnaire D & K-SVD ", padding=12)
        lf_dict.pack(fill="x", pady=(0, 10))
        ttk.Label(lf_dict, text="Type de dictionnaire", style="CardBody.TLabel").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Combobox(
            lf_dict,
            textvariable=self.vars["dictionary_type"],
            values=list(DICTIONARY_COMBO_TEXT),
            state="readonly",
            width=42,
        ).grid(row=0, column=1, sticky="ew", pady=4)
        lf_dict.columnconfigure(1, weight=1)
        self._simple_entry(lf_dict, 1, "Nombre d’atomes K (vide = N)", self.vars["n_atoms"])
        self._simple_entry(lf_dict, 2, "Itérations K-SVD (0 = pas d’apprentissage itéré)", self.vars["n_iter_ksvd"])
        ttk.Label(lf_dict, text=UI_HELP_DICT_BLOC, style="CardMuted.TLabel", wraplength=720, justify="left").grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(10, 0)
        )

        # --- Paramètres solveurs (partagés + spécifiques)
        lf_sol = ttk.LabelFrame(scroll_inner, text=" Paramètres solveurs ", padding=12)
        lf_sol.pack(fill="x", pady=(0, 10))
        self._simple_entry(lf_sol, 0, "max_iter (itératifs)", self.vars["max_iter"])
        self._simple_entry(lf_sol, 1, "epsilon (résidu / tol)", self.vars["epsilon"])
        self._simple_entry(lf_sol, 2, "t StOMP (seuil)", self.vars["t_stomp"])
        ttk.Label(
            lf_sol,
            text="Indication : pour StOMP, t est souvent pris entre 2 et 3 (à ajuster selon le niveau de bruit).",
            style="CardMuted.TLabel",
            wraplength=560,
            justify="left",
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(0, 6))
        self._simple_entry(lf_sol, 4, "s CoSaMP (fixe)", self.vars["s_cosamp"])
        ttk.Checkbutton(
            lf_sol,
            text="CoSaMP : estimer s automatiquement (OMP sur patchs d’entraînement)",
            variable=self.vars["s_cosamp_auto"],
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=6)
        self._simple_entry(lf_sol, 6, "p IRLS (norme ℓp)", self.vars["norm_p"])
        ttk.Label(
            lf_sol,
            text="IRLS : p dans ]0, 1[ (ex. 0,5). Pour ℓ1 pur, préférer BP ou LP.",
            style="CardMuted.TLabel",
            wraplength=560,
            justify="left",
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(0, 6))
        self._simple_entry(lf_sol, 8, "λ LASSO", self.vars["lambda_lasso"])
        ttk.Label(
            lf_sol,
            text="Indication : λ souvent entre 1e-4 et 1e-1 selon l’image et le bruit ; trop élevé = image trop lissée.",
            style="CardMuted.TLabel",
            wraplength=560,
            justify="left",
        ).grid(row=9, column=0, columnspan=2, sticky="w", pady=(0, 6))
        self._simple_entry(lf_sol, 10, "Limiter nb patchs reco (vide = image entière)", self.vars["max_patches"])
        self._simple_entry(lf_sol, 11, "Seed reproductibilité", self.vars["seed"])
        ttk.Checkbutton(lf_sol, text="Arrêt si PSNR patch ≥ cible (expérimental)", variable=self.vars["psnr_stop"]).grid(
            row=12, column=0, columnspan=2, sticky="w", pady=6
        )
        self._simple_entry(lf_sol, 13, "PSNR cible (dB)", self.vars["psnr_target_db"])

        # --- Empreinte carbone (rapport / sensibilisation)
        lf_emp = ttk.LabelFrame(scroll_inner, text=" Empreinte carbone (estimation indicative) ", padding=12)
        lf_emp.pack(fill="x", pady=(0, 10))
        ttk.Label(
            lf_emp,
            text="La colonne CO₂eq du tableau (onglet Résultats) reprend l’estimation totale répartie au prorata du temps par méthode. W et g/kWh sont des hypothèses — voir EMPREINTE.md.",
            style="CardMuted.TLabel",
            wraplength=720,
            justify="left",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        ttk.Checkbutton(lf_emp, text="Activer le calcul", variable=self.vars["empreinte_carbone"]).grid(row=1, column=0, sticky="w")
        self._simple_entry(lf_emp, 2, "Hypothèse puissance (W)", self.vars["empreinte_puissance_w"])
        self._simple_entry(lf_emp, 3, "Intensité (g CO₂eq / kWh)", self.vars["empreinte_g_co2_par_kwh"])

        lf_auto = ttk.LabelFrame(scroll_inner, text=" Assistant — meilleure config indicatif (PSNR) ", padding=12)
        lf_auto.pack(fill="x", pady=(0, 10))
        lf_auto.columnconfigure(0, weight=1)
        ttk.Label(
            lf_auto,
            text=(
                "Les images « étapes » dans Patchs montrent la recomposition progressive des blocs dans l’image recadrée "
                "(aperçu pédagogique du découpage), pas les itérations internes d’un solveur.\n\n"
                "Ici : 12 combinaisons (ratio × Φ) × 8 méthodes = 96 évaluations sur un sous-ensemble de patchs. "
                "Le PSNR est indicatif ; validez ensuite avec une reconstruction complète sur toute l’image."
            ),
            style="CardMuted.TLabel",
            wraplength=720,
            justify="left",
        ).grid(row=0, column=0, sticky="w")
        self.vars["coarse_patch_cap"] = tk.StringVar(value="72")
        self._simple_entry(lf_auto, 1, "Patchs max par essai (balayage)", self.vars["coarse_patch_cap"])
        ttk.Button(
            lf_auto,
            text="Lancer le balayage et afficher la meilleure combinaison",
            style="Primary.TButton",
            command=self.run_coarse_search,
        ).grid(row=2, column=0, sticky="ew", pady=(14, 0))

        # --- Droite : méthodes
        lf_meth = ttk.LabelFrame(right, text=" Méthodes parcimonieuses ", padding=12)
        lf_meth.pack(fill="both", expand=True)
        grid_m = ttk.Frame(lf_meth, style="Card.TFrame")
        grid_m.pack(fill="both", expand=True, pady=(8, 0))
        for i, (mid, caption) in enumerate(SOLVER_UI_CHOICES):
            var = tk.BooleanVar(value=mid in {"omp", "cosamp"})
            self.method_vars[mid] = var
            r, c = divmod(i, 2)
            ttk.Checkbutton(grid_m, text=caption, variable=var).grid(row=r, column=c, sticky="w", padx=6, pady=5)
        ttk.Label(lf_meth, text=UI_HELP_METHODS_BLOC, style="CardMuted.TLabel", wraplength=340, justify="left").pack(
            fill="x", pady=(12, 0)
        )

        lf_act = ttk.LabelFrame(right, text=" Exécution ", padding=12)
        lf_act.pack(fill="x", pady=(10, 0))
        ttk.Button(lf_act, text="Lancer la reconstruction", style="Primary.TButton", command=self.run_reconstruction).pack(
            fill="x", pady=(0, 8)
        )
        ttk.Checkbutton(lf_act, text="Sauvegarder PNG + CSV après succès", variable=self.vars["auto_save"]).pack(anchor="w")
        ttk.Button(lf_act, text="Onglet Résultats", command=lambda: self.app.select_tab("Résultats")).pack(fill="x", pady=(10, 0))

    def _entry_row(self, parent: ttk.Frame, row: int, label: str, key: str, *, browse_file: bool = False, browse_dir: bool = False) -> None:
        if key not in self.vars:
            self.vars[key] = tk.StringVar()
        ttk.Label(parent, text=label, style="CardBody.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
        ttk.Entry(parent, textvariable=self.vars[key], width=48).grid(row=row, column=1, sticky="ew", pady=4)
        if browse_file:
            ttk.Button(parent, text="…", width=3, command=lambda k=key: self._browse_file(k)).grid(row=row, column=2, padx=(6, 0), pady=4)
        elif browse_dir:
            ttk.Button(parent, text="…", width=3, command=lambda k=key: self._browse_dir(k)).grid(row=row, column=2, padx=(6, 0), pady=4)
        parent.columnconfigure(1, weight=1)

    def _simple_entry(self, parent: ttk.Frame, row: int, label: str, var: tk.Variable) -> None:
        ttk.Label(parent, text=label, style="CardBody.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
        ttk.Entry(parent, textvariable=var, width=22).grid(row=row, column=1, sticky="w", pady=4)

    def _browse_file(self, key: str) -> None:
        path = filedialog.askopenfilename(title="Choisir un fichier")
        if path:
            self.vars[key].set(path)

    def _browse_dir(self, key: str) -> None:
        path = filedialog.askdirectory(title="Choisir un dossier")
        if path:
            self.vars[key].set(path)

    def _set_defaults(self) -> None:
        self.vars["image_path"].set(self.state.image_path)
        self.vars["dictionary_train_image_path"].set(self.state.dictionary_train_image_path)
        self.vars["output_path"].set(self.state.output_path)
        self.vars["block_size"].set("8")
        self.vars["ratio"].set("25")
        self.vars["M_explicit"].set("")
        self.vars["n_atoms"].set("")
        self.vars["n_iter_ksvd"].set("0")
        self.vars["seed"].set("0")
        self.vars["max_patches"].set("")
        self.vars["max_iter"].set("20")
        self.vars["epsilon"].set("1e-6")
        self.vars["t_stomp"].set("2.5")
        self.vars["s_cosamp"].set("6")
        self.vars["norm_p"].set("0.5")
        self.vars["psnr_target_db"].set("45")
        self.vars["lambda_lasso"].set("0.01")
        self.vars["measurement_mode"].set("phi4")
        for disp, key in DICTIONARY_COMBO_TO_KEY.items():
            if key == "dct":
                self.vars["dictionary_type"].set(disp)
                break
        else:
            self.vars["dictionary_type"].set(next(iter(DICTIONARY_COMBO_TO_KEY)))
        self.vars["empreinte_puissance_w"].set("45")
        self.vars["empreinte_g_co2_par_kwh"].set("85")

    def _build_method_params(self, methods: list[str]) -> dict[str, dict]:
        mi = parse_int(self.vars["max_iter"].get(), 20) or 20
        eps = parse_float(self.vars["epsilon"].get(), 1e-6) or 1e-6
        lam = parse_float(self.vars["lambda_lasso"].get(), 0.01) or 0.01
        t_st = parse_float(self.vars["t_stomp"].get(), 2.5) or 2.5
        s_c = parse_int(self.vars["s_cosamp"].get(), 6) or 6
        p_ir = parse_float(self.vars["norm_p"].get(), 0.5) or 0.5

        full: dict[str, dict] = {
            "mp": {"max_iter": mi, "epsilon": eps},
            "omp": {"max_iter": mi, "epsilon": eps},
            "stomp": {"max_iter": mi, "epsilon": eps, "t": t_st},
            "cosamp": {"max_iter": mi, "epsilon": eps, "s": s_c},
            "irls": {"max_iter": mi, "epsilon": eps, "norm_p": p_ir},
            "bp": {},
            "lp": {},
            "lasso": {"max_iter": mi, "lambda_reg": lam, "tol": eps},
        }
        return {m: full[m] for m in methods if m in full}

    def run_coarse_search(self) -> None:
        try:
            cap = parse_int(self.vars["coarse_patch_cap"].get(), 72) or 72
            if cap < 4:
                raise ValueError("Indiquez au moins 4 patchs max par essai.")

            image_path = self.vars["image_path"].get().strip()
            if not image_path:
                raise ValueError("Choisissez une image.")
            if not Path(image_path).is_file():
                raise ValueError(f"Image introuvable : {image_path}")
            output_path = self.vars["output_path"].get().strip()
            dictionary_train = self.vars["dictionary_train_image_path"].get().strip() or None
            block_size_placeholder = int(self.vars["block_size"].get())

            patch_params: dict = {
                "order": "C",
                "psnr_stop": bool(self.vars["psnr_stop"].get()),
                "psnr_target_db": parse_float(self.vars["psnr_target_db"].get(), 45.0),
                "lambda_lasso": parse_float(self.vars["lambda_lasso"].get(), 0.01),
                "norm_p": parse_float(self.vars["norm_p"].get(), 0.5),
                "s_cosamp_auto": bool(self.vars["s_cosamp_auto"].get()),
            }

            base = setupParam(
                image_path=image_path,
                block_size=block_size_placeholder,
                ratio=float(self.vars["ratio"].get()),
                methodes=["omp"],
                dictionary_type=dictionary_key_from_combo_selection(self.vars["dictionary_type"].get()),
                measurement_mode=self.vars["measurement_mode"].get(),
                output_path=output_path,
                n_atoms=parse_int(self.vars["n_atoms"].get(), None),
                n_iter_ksvd=parse_int(self.vars["n_iter_ksvd"].get(), 0) or 0,
                dictionary_train_image_path=dictionary_train,
                method_params=self._build_method_params(list(SOLVER_METHOD_IDS)),
                patch_params=patch_params,
                seed=parse_int(self.vars["seed"].get(), 0),
                empreinte_carbone=False,
                empreinte_afficher_console=False,
                empreinte_puissance_w=float(parse_float(self.vars["empreinte_puissance_w"].get(), 45.0) or 45.0),
                empreinte_g_co2_par_kwh=float(parse_float(self.vars["empreinte_g_co2_par_kwh"].get(), 85.0) or 85.0),
            )

            self.state.add_log("Assistant : balayage en cours (plusieurs minutes possibles)…")
            self.update_idletasks()
            report = run_coarse_best_search(base, max_patches_cap=cap)
            b = report.get("best")
            n_ev = int(report.get("nb_evaluations") or 0)
            if not b:
                messagebox.showinfo("Balayage", "Aucun résultat exploitable.")
                return

            r = b["ratio"]
            ratio_str = str(int(r)) if abs(r - round(r)) < 1e-9 else str(r)
            trials = report.get("trials") or []
            top3 = sorted(trials, key=lambda x: float(x.get("psnr", 0)), reverse=True)[:3]
            top_lines = "\n".join(
                f"  {i+1}. {t['method'].upper()}  Φ={t['measurement_mode']}  {t['ratio']}%  PSNR≈{t['psnr']:.2f} dB"
                for i, t in enumerate(top3)
            )
            self.state.add_log(
                f"Balayage terminé ({n_ev} évals) : meilleur PSNR indicatif {b['psnr']:.2f} dB — "
                f"{b['method'].upper()} Φ={b['measurement_mode']} ratio={ratio_str} %"
            )
            apply = messagebox.askyesno(
                "Meilleure configuration (indicative)",
                (
                    f"Meilleur essai (≤ {cap} patchs par tirage) :\n\n"
                    f"Méthode : {b['method'].upper()}\n"
                    f"Φ : {b['measurement_mode']}\n"
                    f"Ratio : {ratio_str} %\n"
                    f"PSNR indicatif : {b['psnr']:.2f} dB\n\n"
                    f"Top 3 :\n{top_lines}\n\n"
                    "Appliquer Φ, ratio et une seule méthode cochée dans le formulaire ?"
                ),
            )
            if apply:
                self.vars["ratio"].set(ratio_str)
                self.vars["measurement_mode"].set(b["measurement_mode"])
                for mid, var in self.method_vars.items():
                    var.set(mid == b["method"])
                messagebox.showinfo(
                    "Réglages appliqués",
                    "Vérifiez les cases, puis lancez une reconstruction complète (éventuellement plusieurs méthodes).",
                )
        except Exception as exc:
            messagebox.showerror("Erreur", str(exc))

    def run_reconstruction(self) -> None:
        try:
            methods = [name for name, var in self.method_vars.items() if var.get()]
            if not methods:
                raise ValueError("Sélectionne au moins une méthode.")

            image_path = self.vars["image_path"].get().strip()
            output_path = self.vars["output_path"].get().strip()
            dictionary_train = self.vars["dictionary_train_image_path"].get().strip() or None

            patch_params: dict = {
                "order": "C",
                "psnr_stop": bool(self.vars["psnr_stop"].get()),
                "psnr_target_db": parse_float(self.vars["psnr_target_db"].get(), 45.0),
                "lambda_lasso": parse_float(self.vars["lambda_lasso"].get(), 0.01),
                "norm_p": parse_float(self.vars["norm_p"].get(), 0.5),
                "s_cosamp_auto": bool(self.vars["s_cosamp_auto"].get()),
            }
            mp = parse_int(self.vars["max_patches"].get(), None)
            if mp is not None:
                patch_params["max_patches"] = mp

            m_explicit = self.vars["M_explicit"].get().strip()
            if m_explicit:
                patch_params["M"] = int(m_explicit)

            block_size_placeholder = int(self.vars["block_size"].get())
            patch_params.pop("B", None)
            patch_params.pop("nrows", None)
            patch_params.pop("ncols", None)

            if mp is None:
                patch_params.pop("max_patches", None)
            if not m_explicit:
                patch_params.pop("M", None)

            result = run_main(
                image_path=image_path,
                block_size=block_size_placeholder,
                ratio=float(self.vars["ratio"].get()),
                methodes=methods,
                dictionary_type=dictionary_key_from_combo_selection(self.vars["dictionary_type"].get()),
                measurement_mode=self.vars["measurement_mode"].get(),
                output_path=output_path,
                n_atoms=parse_int(self.vars["n_atoms"].get(), None),
                n_iter_ksvd=parse_int(self.vars["n_iter_ksvd"].get(), 0) or 0,
                dictionary_train_image_path=dictionary_train,
                method_params=self._build_method_params(methods),
                patch_params=patch_params,
                seed=parse_int(self.vars["seed"].get(), 0),
                empreinte_carbone=bool(self.vars["empreinte_carbone"].get()),
                empreinte_afficher_console=False,
                empreinte_puissance_w=float(parse_float(self.vars["empreinte_puissance_w"].get(), 45.0) or 45.0),
                empreinte_g_co2_par_kwh=float(parse_float(self.vars["empreinte_g_co2_par_kwh"].get(), 85.0) or 85.0),
            )

            self.state.image_path = image_path
            self.state.output_path = output_path
            self.state.dictionary_train_image_path = dictionary_train or ""
            self.state.last_result = result
            self.state.add_log(f"Reconstruction : {', '.join(methods)} (Φ={self.vars['measurement_mode'].get()})")

            if bool(self.vars["auto_save"].get()):
                save_results(result, output_path)
                saved_dir = latest_subdir(output_path)
                if saved_dir:
                    self.state.add_log(f"Sauvegardé : {saved_dir}")

            self.app.refresh_all_pages()
            self.app.select_tab("Résultats")
            messagebox.showinfo("Succès", "Reconstruction terminée.")
        except Exception as exc:
            messagebox.showerror("Erreur", str(exc))
