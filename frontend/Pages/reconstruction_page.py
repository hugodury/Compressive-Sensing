from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

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
    UI_HELP_DICT_COMBO_LINES,
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
        # Attributs UI initialisés ici pour éviter AttributeError
        # si _set_busy est appelé avant que _build() les crée
        self._btn_run: ttk.Button | None = None
        self._btn_coarse_run: ttk.Button | None = None
        self._progressbar: ttk.Progressbar | None = None
        self._status_label: ttk.Label | None = None
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
        self.app.register_scroll_canvas(canvas)

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
            "max_time_s",
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
            width=68,
        ).grid(row=0, column=1, sticky="ew", pady=4)
        lf_dict.columnconfigure(1, weight=1)
        self._simple_entry(lf_dict, 1, "Nombre d’atomes K (vide = N)", self.vars["n_atoms"])
        self._simple_entry(lf_dict, 2, "Itérations K-SVD (0 = pas d’apprentissage itéré)", self.vars["n_iter_ksvd"])
        ttk.Label(lf_dict, text=UI_HELP_DICT_BLOC, style="CardMuted.TLabel", wraplength=720, justify="left").grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(10, 0)
        )
        ttk.Label(lf_dict, text=UI_HELP_DICT_COMBO_LINES, style="Hint.TLabel", wraplength=720, justify="left").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )

        # --- Solveurs : paramètres communs (MP, OMP, BP, LP, etc.)
        lf_sol = ttk.LabelFrame(scroll_inner, text=" Solveurs — paramètres communs ", padding=12)
        lf_sol.pack(fill="x", pady=(0, 10))
        self._simple_entry(lf_sol, 0, "max_iter (itératifs)", self.vars["max_iter"])
        self._simple_entry(lf_sol, 1, "epsilon (résidu / tol)", self.vars["epsilon"])
        self._simple_entry(lf_sol, 2, "Temps max d'exécution par méthode (s, vide = illimité)", self.vars["max_time_s"])
        ttk.Label(
            lf_sol,
            text=(
                "max_iter / epsilon : utilisés par MP, OMP, StOMP, CoSaMP, IRLS, LASSO (selon la méthode cochée). "
                "Typ. 20–80 itérations ; epsilon 1e-6 … 1e-4."
            ),
            style="CardMuted.TLabel",
            wraplength=620,
            justify="left",
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self._simple_entry(lf_sol, 4, "Nombre max de patchs à reconstruire (vide = toute l’image)", self.vars["max_patches"])
        ttk.Label(
            lf_sol,
            text=(
                "Par défaut (champ vide), chaque lancement traite tous les patchs : vous reconstruisez l’image entière recadrée. "
                "Renseigner un entier sert seulement aux essais rapides (aperçu partiel, reste de l’image à 0)."
            ),
            style="CardMuted.TLabel",
            wraplength=680,
            justify="left",
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self._simple_entry(lf_sol, 6, "Seed reproductibilité", self.vars["seed"])
        ttk.Checkbutton(lf_sol, text="Arrêt si PSNR patch ≥ cible (expérimental)", variable=self.vars["psnr_stop"]).grid(
            row=7, column=0, columnspan=2, sticky="w", pady=6
        )
        self._simple_entry(lf_sol, 8, "PSNR cible (dB)", self.vars["psnr_target_db"])
        ttk.Label(
            lf_sol,
            text=(
                "PSNR cible : utilisé seulement si l’arrêt expérimental est coché. C’est le PSNR calculé entre l’original "
                "et la reconstruction (image recadrée). Ordre de grandeur : ~30 dB souvent acceptable sur 8 bits, "
                "35–40 dB bon, au-delà très bon selon le contenu. Une cible trop haute peut allonger le calcul sans gain visible."
            ),
            style="CardMuted.TLabel",
            wraplength=620,
            justify="left",
        ).grid(row=9, column=0, columnspan=2, sticky="w", pady=(6, 0))

        lf_stomp = ttk.LabelFrame(scroll_inner, text=" StOMP — paramètres spécifiques ", padding=12)
        lf_stomp.pack(fill="x", pady=(0, 10))
        self._simple_entry(lf_stomp, 0, "t (seuil de sélection des atomes)", self.vars["t_stomp"])
        ttk.Label(
            lf_stomp,
            text="StOMP sélectionne plusieurs atomes par itération si leur corrélation dépasse t × ‖r‖. Souvent t ∈ [2, 3].",
            style="CardMuted.TLabel",
            wraplength=680,
            justify="left",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        lf_cosamp = ttk.LabelFrame(scroll_inner, text=" CoSaMP — paramètres spécifiques ", padding=12)
        lf_cosamp.pack(fill="x", pady=(0, 10))
        self._simple_entry(lf_cosamp, 0, "s (taille du support cible, mode fixe)", self.vars["s_cosamp"])
        ttk.Checkbutton(
            lf_cosamp,
            text="Estimer s automatiquement (OMP sur patchs d’entraînement, même D)",
            variable=self.vars["s_cosamp_auto"],
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=6)
        ttk.Label(
            lf_cosamp,
            text="CoSaMP alterne sélection / rejet d’atomes ; s borne la parcimonie à chaque étape. Si « estimer s » est coché, la valeur ci-dessus est ignorée.",
            style="CardMuted.TLabel",
            wraplength=680,
            justify="left",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 0))

        lf_irls = ttk.LabelFrame(scroll_inner, text=" IRLS — paramètres spécifiques ", padding=12)
        lf_irls.pack(fill="x", pady=(0, 10))
        self._simple_entry(lf_irls, 0, "p (pseudo-norme ℓp, 0 < p < 1)", self.vars["norm_p"])
        ttk.Label(
            lf_irls,
            text="IRLS repondère les moindres carrés pour approcher la parcimonie ℓp. Ex. p = 0,5. Pour ℓ1 strict, utilisez plutôt BP ou LP.",
            style="CardMuted.TLabel",
            wraplength=680,
            justify="left",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        lf_lasso = ttk.LabelFrame(scroll_inner, text=" LASSO — paramètres spécifiques ", padding=12)
        lf_lasso.pack(fill="x", pady=(0, 10))
        self._simple_entry(lf_lasso, 0, "λ (régularisation ℓ1)", self.vars["lambda_lasso"])
        ttk.Label(
            lf_lasso,
            text="λ contrôle le compromis fidélité / parcimonie ; typ. 1e-4 … 1e-1 selon l’image et le bruit.",
            style="CardMuted.TLabel",
            wraplength=680,
            justify="left",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        # --- Empreinte carbone (rapport / sensibilisation)
        lf_emp = ttk.LabelFrame(scroll_inner, text=" Empreinte carbone (estimation indicative) ", padding=12)
        lf_emp.pack(fill="x", pady=(0, 10))
        ttk.Label(
            lf_emp,
            text="La colonne CO₂eq (onglet Résultats) repartit l’estimation au prorata du temps par méthode. L’onglet « Analyses & graphiques » affiche aussi le CO₂ de la dernière session (sweep ou export des tableaux). Hypothèses W et g/kWh — voir EMPREINTE.md.",
            style="CardMuted.TLabel",
            wraplength=720,
            justify="left",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        ttk.Checkbutton(lf_emp, text="Activer le calcul", variable=self.vars["empreinte_carbone"]).grid(row=1, column=0, sticky="w")
        self._simple_entry(lf_emp, 2, "Hypothèse puissance (W)", self.vars["empreinte_puissance_w"])
        self._simple_entry(lf_emp, 3, "Intensité (g CO₂eq / kWh)", self.vars["empreinte_g_co2_par_kwh"])
        ttk.Label(
            lf_emp,
            text=(
                "CO₂ : l’onglet Résultats montre une fourchette (temps mural vs temps CPU, mêmes W et g/kWh). "
                "Stockage : l’onglet Résultats compare seulement fichier avant ↔ données après compression (mesures y + Φ), "
                "sans inclure les PNG de reconstruction. Détail : stockage_compression.txt (annexe = taille dossier export)."
            ),
            style="CardMuted.TLabel",
            wraplength=720,
            justify="left",
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(10, 0))

        self._assistant_pack_anchor = lf_emp
        self._assistant_outer = ttk.Frame(scroll_inner, style="Panel.TFrame")
        self._assistant_outer.pack(fill="x", pady=(0, 10), after=self._assistant_pack_anchor)

        abar = ttk.Frame(self._assistant_outer)
        abar.pack(fill="x", pady=(0, 8))
        ttk.Label(abar, text="Assistant — repères et balayage partiel", style="CardTitle.TLabel").pack(
            side="left", anchor="w"
        )
        ttk.Button(abar, text="Masquer cette section", command=self._hide_assistant_outer).pack(side="right")

        lf_auto = ttk.LabelFrame(self._assistant_outer, text=" Balayage sur votre image & repères optionnels ", padding=12)
        lf_auto.pack(fill="x")
        lf_auto.columnconfigure(0, weight=1)
        ttk.Label(
            lf_auto,
            text=(
                "Si vous ne savez pas quoi régler : le plus utile est de lancer le balayage ci-dessous avec votre image "
                "et le dictionnaire / B déjà choisis plus haut. Il compare plusieurs ratios (20 %, 35 %, 50 %) × les quatre Φ "
                "× les huit méthodes (sur un nombre limité de patchs). Ce n’est pas une optimisation globale "
                "(pas de recherche du meilleur dictionnaire ni des meilleurs max_iter, t, s, etc.), mais les trois boutons "
                "qui apparaissent ensuite sont basés sur votre image — c’est la recommandation la plus fiable dans cet assistant.\n\n"
                "Les trois boutons tout en bas sont seulement des exemples de formulaire (raccourcis), pas garantis optimaux "
                "pour toutes les images : utilisez-les si vous voulez un point de départ avant d’affiner ou de balayer."
            ),
            style="CardMuted.TLabel",
            wraplength=720,
            justify="left",
        ).grid(row=0, column=0, sticky="w")

        self.vars["coarse_patch_cap"] = tk.StringVar(value="72")
        scan_fr = ttk.Frame(lf_auto)
        scan_fr.grid(row=1, column=0, sticky="ew", pady=(14, 0))
        scan_fr.columnconfigure(1, weight=1)
        self._simple_entry(scan_fr, 0, "Patchs max par essai (balayage)", self.vars["coarse_patch_cap"])
        ttk.Button(
            scan_fr,
            text="Étape recommandée : lancer le balayage → puis appliquer 1ᵉʳ / 2ᵉ / 3ᵉ essai ci-dessous",
            style="Primary.TButton",
            command=self.run_coarse_search,
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(12, 0))

        coarse_wrap = ttk.Frame(lf_auto)
        coarse_wrap.grid(row=2, column=0, sticky="ew", pady=(16, 0))
        ttk.Label(
            coarse_wrap,
            text="Après le balayage — les trois meilleurs essais sur votre image (PSNR indicatif) :",
            style="CardTitle.TLabel",
        ).pack(anchor="w")
        self._coarse_top3_btn_row = ttk.Frame(coarse_wrap)
        self._coarse_top3_btn_row.pack(fill="x", pady=(8, 0))
        self._coarse_top3_placeholder = ttk.Label(
            self._coarse_top3_btn_row,
            text="Les boutons apparaîtront ici après le balayage.",
            style="CardMuted.TLabel",
        )
        self._coarse_top3_placeholder.pack(anchor="w")

        preset_fr = ttk.Frame(lf_auto)
        preset_fr.grid(row=3, column=0, sticky="ew", pady=(18, 0))
        ttk.Label(
            preset_fr,
            text="Optionnel — exemples de réglages (sans calcul ; à titre indicatif seulement) :",
            style="CardBody.TLabel",
        ).pack(anchor="w")
        btn_row = ttk.Frame(preset_fr)
        btn_row.pack(fill="x", pady=(8, 0))
        ttk.Button(
            btn_row,
            text="Exemple A — DCT & Φ₄ (classique)",
            command=lambda: self._apply_assistant_preset(1),
        ).pack(side="left", padx=(0, 8))
        ttk.Button(
            btn_row,
            text="Exemple B — K-SVD depuis DCT",
            command=lambda: self._apply_assistant_preset(2),
        ).pack(side="left", padx=(0, 8))
        ttk.Button(
            btn_row,
            text="Exemple C — peu de patchs, rapide",
            command=lambda: self._apply_assistant_preset(3),
        ).pack(side="left", padx=(0, 8))

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
        self._btn_show_assistant = ttk.Button(
            lf_act,
            text="Afficher l’assistant repères",
            command=self._show_assistant_outer,
        )
        self._btn_run = ttk.Button(lf_act, text="Lancer la reconstruction", style="Primary.TButton", command=self.run_reconstruction)
        self._btn_run.pack(fill="x", pady=(0, 6))
        self._btn_coarse_run: ttk.Button | None = None
        self._progressbar = ttk.Progressbar(lf_act, mode="indeterminate", length=200)
        self._progressbar.pack(fill="x", pady=(0, 2))
        self._status_label = ttk.Label(lf_act, text="", style="Muted.TLabel", anchor="center")
        self._status_label.pack(fill="x", pady=(0, 6))
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

    def _set_busy(self, msg: str = "Calcul en cours…") -> None:
        """Désactive les boutons et lance la barre de progression."""
        if self._btn_run is not None:
            self._btn_run.configure(state="disabled")
        if self._btn_coarse_run is not None:
            try:
                self._btn_coarse_run.configure(state="disabled")
            except tk.TclError:
                pass
        if self._status_label is not None:
            self._status_label.configure(text=msg)
        if self._progressbar is not None:
            self._progressbar.start(12)

    def _set_idle(self, msg: str = "") -> None:
        """Réactive les boutons et stoppe la barre de progression."""
        if self._progressbar is not None:
            self._progressbar.stop()
        if self._btn_run is not None:
            self._btn_run.configure(state="normal")
        if self._btn_coarse_run is not None:
            try:
                self._btn_coarse_run.configure(state="normal")
            except tk.TclError:
                pass
        if self._status_label is not None:
            self._status_label.configure(text=msg)

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
        self.vars["max_time_s"].set("")
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

    @staticmethod
    def _dict_display_for_key(key: str) -> str:
        for disp, k in DICTIONARY_COMBO_TO_KEY.items():
            if k == key:
                return disp
        return DICTIONARY_COMBO_TEXT[0]

    def _hide_assistant_outer(self) -> None:
        self._assistant_outer.pack_forget()
        self._btn_show_assistant.pack(fill="x", pady=(0, 8))

    def _show_assistant_outer(self) -> None:
        self._btn_show_assistant.pack_forget()
        self._assistant_outer.pack(fill="x", pady=(0, 10), after=self._assistant_pack_anchor)

    def _apply_assistant_preset(self, choice: int) -> None:
        """Remplit le formulaire avec trois profils pédagogiques (sans lancer de calcul)."""
        self.vars["M_explicit"].set("")
        self.vars["block_size"].set("8")
        self.vars["measurement_mode"].set("phi4")
        self.vars["s_cosamp_auto"].set(False)
        if choice == 1:
            self.vars["dictionary_type"].set(self._dict_display_for_key("dct"))
            self.vars["ratio"].set("25")
            self.vars["max_iter"].set("40")
            self.vars["max_patches"].set("")
            self.vars["n_iter_ksvd"].set("0")
            self.vars["epsilon"].set("1e-6")
            self.vars["s_cosamp"].set("6")
            for mid, var in self.method_vars.items():
                var.set(mid in ("omp", "cosamp"))
        elif choice == 2:
            self.vars["dictionary_type"].set(self._dict_display_for_key("ksvd_dct"))
            self.vars["ratio"].set("40")
            self.vars["max_iter"].set("55")
            self.vars["max_patches"].set("")
            self.vars["n_iter_ksvd"].set("10")
            self.vars["epsilon"].set("1e-6")
            self.vars["s_cosamp"].set("8")
            for mid, var in self.method_vars.items():
                var.set(mid in ("omp", "cosamp", "irls"))
        elif choice == 3:
            self.vars["dictionary_type"].set(self._dict_display_for_key("dct"))
            self.vars["ratio"].set("20")
            self.vars["max_iter"].set("22")
            self.vars["max_patches"].set("64")
            self.vars["n_iter_ksvd"].set("0")
            self.vars["epsilon"].set("1e-5")
            for mid, var in self.method_vars.items():
                var.set(mid == "omp")
        else:
            return
        letter = "ABC"[choice - 1]
        self.state.add_log(
            f"Assistant : exemple {letter} appliqué au formulaire (indicatif — pas le meilleur pour toutes les images)."
        )
        messagebox.showinfo(
            "Exemple appliqué",
            f"Exemple {letter} : réglages types seulement. Pour une suggestion basée sur votre image, "
            "utilisez plutôt le balayage puis « Appliquer le 1ᵉʳ / 2ᵉ / 3ᵉ essai ».",
        )

    def _apply_coarse_trial(self, trial: dict[str, Any]) -> None:
        r = float(trial["ratio"])
        ratio_str = str(int(r)) if abs(r - round(r)) < 1e-9 else str(r)
        self.vars["ratio"].set(ratio_str)
        self.vars["M_explicit"].set("")
        self.vars["measurement_mode"].set(str(trial["measurement_mode"]))
        meth = str(trial["method"]).lower()
        for mid, var in self.method_vars.items():
            var.set(mid == meth)
        ps = float(trial.get("psnr", 0.0))
        self.state.add_log(
            f"Assistant : appliqué {meth.upper()} · Φ={trial['measurement_mode']} · {ratio_str} % "
            f"(PSNR indicatif {ps:.2f} dB)."
        )
        messagebox.showinfo(
            "Paramètres appliqués",
            "Une seule méthode est cochée. Lancez une reconstruction complète ; vous pourrez recocher d’autres méthodes ensuite.",
        )

    def _populate_coarse_top3(self, trials_sorted: list[dict[str, Any]]) -> None:
        for w in self._coarse_top3_btn_row.winfo_children():
            w.destroy()
        top3 = trials_sorted[:3]
        if not top3:
            ttk.Label(self._coarse_top3_btn_row, text="Aucun essai classé.", style="CardMuted.TLabel").pack(anchor="w")
            return
        for i, trial in enumerate(top3):
            tr = dict(trial)
            r = float(tr["ratio"])
            ratio_str = str(int(r)) if abs(r - round(r)) < 1e-9 else str(r)
            ps = float(tr.get("psnr", 0.0))
            caption = (
                f"Appliquer le {i + 1}ᵉ essai : {str(tr['method']).upper()} · {tr['measurement_mode']} · "
                f"{ratio_str} % · {ps:.1f} dB"
            )
            ttk.Button(
                self._coarse_top3_btn_row,
                text=caption,
                command=lambda t=tr: self._apply_coarse_trial(t),
            ).pack(fill="x", pady=(0, 6))

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
            max_time_s = parse_float(self.vars["max_time_s"].get(), None)
            if max_time_s is not None:
                if max_time_s <= 0:
                    raise ValueError("Le temps max d'exécution doit être > 0 s.")
                patch_params["max_time_s"] = float(max_time_s)

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
        except Exception as exc:
            messagebox.showerror("Erreur", str(exc))
            return

        self._set_busy("Balayage en cours — quelques minutes possibles…")

        def _worker() -> None:
            try:
                report = run_coarse_best_search(base, max_patches_cap=cap)
            except Exception as exc:
                self.after(0, lambda e=exc: self._on_coarse_error(e))
                return
            self.after(0, lambda rep=report, c=cap: self._on_coarse_done(rep, c))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_coarse_error(self, exc: Exception) -> None:
        self._set_idle()
        messagebox.showerror("Erreur balayage", str(exc))

    def _on_coarse_done(self, report: dict, cap: int) -> None:
        b = report.get("best")
        n_ev = int(report.get("nb_evaluations") or 0)
        if not b:
            self._set_idle()
            messagebox.showinfo("Balayage", "Aucun résultat exploitable.")
            return
        b = report.get("best")
        n_ev = int(report.get("nb_evaluations") or 0)
        if not b:
            messagebox.showinfo("Balayage", "Aucun résultat exploitable.")
            return

        r = b["ratio"]
        ratio_str = str(int(r)) if abs(r - round(r)) < 1e-9 else str(r)
        trials = report.get("trials") or []
        trials_sorted = sorted(trials, key=lambda x: float(x.get("psnr", 0.0)), reverse=True)
        top3 = trials_sorted[:3]
        top_lines = "\n".join(
            f"  {i + 1}. {t['method'].upper()}  Φ={t['measurement_mode']}  {t['ratio']}%  PSNR≈{t['psnr']:.2f} dB"
            for i, t in enumerate(top3)
        )
        self.state.add_log(
            f"Balayage terminé ({n_ev} évals) : meilleur PSNR indicatif {b['psnr']:.2f} dB — "
            f"{b['method'].upper()} Φ={b['measurement_mode']} ratio={ratio_str} %"
        )
        self._populate_coarse_top3(trials_sorted)
        messagebox.showinfo(
            "Balayage terminé",
            (
                f"{n_ev} évaluations (≤ {cap} patchs par tirage).\n"
                f"Meilleur PSNR indicatif : {b['psnr']:.2f} dB — {b['method'].upper()}, "
                f"Φ={b['measurement_mode']}, {ratio_str} %.\n\n"
                f"Top 3 :\n{top_lines}\n\n"
                "Utilisez les boutons « Appliquer le 1ᵉʳ / 2ᵉ / 3ᵉ essai » dans l’assistant "
                "(réaffichez-le avec « Afficher l’assistant repères » si vous l’avez masqué)."
            ),
        )
        self._set_idle("Balayage terminé ✔")

    def run_reconstruction(self) -> None:
        methods = [name for name, var in self.method_vars.items() if var.get()]
        if not methods:
            messagebox.showerror("Erreur", "Sélectionne au moins une méthode.")
            return

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
        max_time_s = parse_float(self.vars["max_time_s"].get(), None)
        if max_time_s is not None:
            if max_time_s <= 0:
                messagebox.showerror("Erreur", "Le temps max d'exécution doit être > 0 s.")
                return
            patch_params["max_time_s"] = float(max_time_s)
        mp_val = parse_int(self.vars["max_patches"].get(), None)
        if mp_val is not None:
            patch_params["max_patches"] = mp_val
        m_explicit = self.vars["M_explicit"].get().strip()
        if m_explicit:
            patch_params["M"] = int(m_explicit)
        block_size_placeholder = int(self.vars["block_size"].get())
        patch_params.pop("B", None)
        patch_params.pop("nrows", None)
        patch_params.pop("ncols", None)
        if mp_val is None:
            patch_params.pop("max_patches", None)
        if not m_explicit:
            patch_params.pop("M", None)

        kwargs = dict(
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

        label_meths = ", ".join(m.upper() for m in methods)
        self._set_busy(f"Reconstruction en cours ({label_meths})…")

        def _worker() -> None:
            try:
                result = run_main(**kwargs)
            except Exception as exc:
                self.after(0, lambda e=exc: self._on_reconstruction_error(e))
                return
            self.after(0, lambda res=result: self._on_reconstruction_done(
                res, image_path, output_path, dictionary_train, methods
            ))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_reconstruction_error(self, exc: Exception) -> None:
        self._set_idle()
        messagebox.showerror("Erreur reconstruction", str(exc))

    def _on_reconstruction_done(
        self,
        result: dict,
        image_path: str,
        output_path: str,
        dictionary_train: str | None,
        methods: list[str],
    ) -> None:
        self.state.image_path = image_path
        self.state.output_path = output_path
        self.state.dictionary_train_image_path = dictionary_train or ""
        # Garder l'avant-dernière reconstruction pour la comparaison de dictionnaires
        if self.state.last_result is not None:
            self.state.last_result_prev = self.state.last_result
        self.state.last_result = result
        phi = self.vars["measurement_mode"].get()
        self.state.add_log(f"Reconstruction : {", ".join(methods)} (Φ={phi})")

        if bool(self.vars["auto_save"].get()):
            save_results(result, output_path)
            saved_dir = latest_subdir(output_path)
            if saved_dir:
                self.state.add_log(f"Sauvegardé : {saved_dir}")

        time_limited_methods = [
            str(m).upper()
            for m, met in (result.get("metrics") or {}).items()
            if bool((met or {}).get("time_limit_reached"))
        ]
        if time_limited_methods:
            self.state.add_log(
                "Arrêt sur limite de temps : " + ", ".join(time_limited_methods)
            )

        self._set_idle("Reconstruction terminée ✔")
        self.app.refresh_all_pages()
        self.app.select_tab("Résultats")
        if time_limited_methods:
            messagebox.showwarning(
                "Terminé (limite de temps atteinte)",
                "La reconstruction est terminée, mais la limite de temps a été atteinte pour : "
                + ", ".join(time_limited_methods)
                + ".\nLe résultat peut être partiel.",
            )
        else:
            messagebox.showinfo("Succès", "Reconstruction terminée.")