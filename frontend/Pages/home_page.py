from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from frontend.theme import BG
from frontend.utils import build_pipeline_diagram_figure, figure_to_photo
from .base_page import BasePage


class HomePage(BasePage):
    def __init__(self, parent: tk.Misc, app) -> None:
        super().__init__(parent, app)

        self.columnconfigure(0, weight=1)

        ttk.Label(self, text="Compressive sensing — pilotage du projet", style="Title.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 4)
        )
        ttk.Label(
            self,
            text="Interface locale pour reconstruire des images sous-échantillonnées, comparer des algorithmes et exporter des analyses.",
            style="WelcomeSubtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(0, 14))

        hero = ttk.Frame(self, style="WelcomeHero.TFrame", padding=(22, 20))
        hero.grid(row=2, column=0, sticky="ew", pady=(0, 16))
        hero.columnconfigure(0, weight=1)

        intro_lines = (
            "Vue d’ensemble : l’image est découpée en patchs carrés x (vecteur de taille N = B²) ; on observe y = Φx avec peu "
            "de mesures (M ≪ N), puis on cherche des coefficients parcimonieux α sur un dictionnaire D pour approcher x ≈ Dα ; "
            "enfin on recolle les patchs pour former l’image reconstruite. Le dictionnaire D est une matrice N×K dont les colonnes "
            "sont des « atomes » (bases DCT, mixte ou apprises type K-SVD) : α indique quels atomes combiner.",
            "Méthodes disponibles : MP, OMP, StOMP, CoSaMP, IRLS, BP, LP, LASSO. Quatre familles Φ (Φ₁ à Φ₄), balayage de ratios, "
            "exports cohérence / erreurs, visualisation des patchs. Sorties : PSNR, MSE, erreur relative, temps, CO₂eq estimatif.",
        )
        for i, line in enumerate(intro_lines):
            ttk.Label(hero, text=line, style="WelcomeLead.TLabel", justify="left").grid(
                row=i, column=0, sticky="ew", pady=(0, 12 if i == 0 else 0)
            )

        diag_frame = ttk.LabelFrame(self, text=" Schéma du pipeline (ce que fait le code) ", padding=(14, 12))
        diag_frame.grid(row=3, column=0, sticky="ew", pady=(0, 14))
        diag_frame.columnconfigure(0, weight=1)
        fig = build_pipeline_diagram_figure()
        self._pipeline_photo = figure_to_photo(fig, dpi=120, max_width_px=980)
        diag_lbl = tk.Label(diag_frame, image=self._pipeline_photo, bg=BG, bd=0, highlightthickness=0)
        diag_lbl.image = self._pipeline_photo
        diag_lbl.grid(row=0, column=0, sticky="ew")

        pipeline = ttk.LabelFrame(self, text=" Déroulement conseillé (pipeline) ", padding=(18, 14))
        pipeline.grid(row=4, column=0, sticky="ew", pady=(0, 16))
        pipeline.columnconfigure(0, weight=1)

        steps = (
            (
                "1. Reconstruction",
                "Ouvrez l’onglet « Reconstruction » : choisissez l’image, la taille de patch B, le ratio (ou M), une matrice Φ, "
                "un type de dictionnaire, puis cochez une ou plusieurs méthodes. Vous pouvez activer l’empreinte carbone et, "
                "si besoin, utiliser l’« Assistant » pour un premier réglage rapide (PSNR indicatif sur un sous-ensemble de patchs).",
            ),
            (
                "2. Lancer le calcul",
                "« Lancer la reconstruction » : découpage B×B, mesure y = Φx par patch, résolution parcimonieuse (x̂ ≈ Dα̂), "
                "recomposition de l’image à partir des patchs reconstruits, puis métriques globales (PSNR, MSE, etc.).",
            ),
            (
                "3. Résultats",
                "Allez dans « Résultats » : tableau (PSNR, MSE, erreur, temps, CO₂eq estimé), graphique synthèse et aperçu des images. "
                "Depuis Reconstruction, la sauvegarde automatique peut écrire PNG et CSV sur le disque.",
            ),
            (
                "4. Aller plus loin (au choix)",
                "« Comparaisons » : courbes PSNR en fonction du ratio. « Cohérence & erreurs » : export CSV (μ, M, erreurs). "
                "« Patchs » : images pour vérifier le découpage B×B. Enchaînez les onglets dans l’ordre qui vous convient pour votre rapport ou vos essais.",
            ),
        )
        for si, (title, body) in enumerate(steps):
            ttk.Label(pipeline, text=title, style="CardTitle.TLabel").grid(
                row=si * 2, column=0, sticky="w", pady=(10, 2) if si else (0, 2)
            )
            ttk.Label(pipeline, text=body, style="WelcomeBullet.TLabel", wraplength=940, justify="left").grid(
                row=si * 2 + 1, column=0, sticky="ew", pady=(0, 4)
            )

        cards = ttk.Frame(self, style="App.TFrame")
        cards.grid(row=5, column=0, sticky="nsew")
        cards.columnconfigure((0, 1), weight=1)
        self.rowconfigure(5, weight=1)

        self._make_card(
            cards,
            0,
            0,
            "Reconstruction",
            "Image, taille de patch, ratio ou nombre de mesures M, matrice Φ, dictionnaire, méthodes à comparer.",
        )
        self._make_card(
            cards,
            0,
            1,
            "Résultats",
            "Tableau des métriques avec colonne CO₂eq (estimation), graphique synthèse et aperçu des images.",
        )
        self._make_card(
            cards,
            1,
            0,
            "Comparaisons",
            "Balayage sur plusieurs ratios : courbes PSNR par méthode, tableau récapitulatif.",
        )
        self._make_card(
            cards,
            1,
            1,
            "Analyses & patchs",
            "Export CSV (cohérence mutuelle, erreurs, mesures) et visualisation du découpage en patchs.",
        )

        recap = ttk.Frame(self, style="Card.TFrame", padding=18)
        recap.grid(row=6, column=0, sticky="ew", pady=(18, 0))
        recap.columnconfigure(0, weight=1)
        ttk.Label(recap, text="Fonctionnalités principales", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        bullets = (
            "Matrices Φ₁…Φ₄ ; dictionnaires DCT fixe, mixte DCT+patchs, K-SVD (aléatoire, depuis DCT ou depuis mixte) ; patchs B×B",
            "CoSaMP avec s fixe ou estimé, arrêt optionnel sur PSNR cible",
            "Exports disque (PNG, CSV), estimation CO₂ documentée dans EMPREINTE.md",
        )
        for i, b in enumerate(bullets, start=1):
            ttk.Label(recap, text=f"• {b}", style="WelcomeBullet.TLabel", justify="left").grid(
                row=i, column=0, sticky="w", pady=(8, 0)
            )

    def _make_card(self, parent: ttk.Frame, row: int, col: int, title: str, text: str) -> None:
        frame = ttk.Frame(parent, style="Card.TFrame", padding=18)
        frame.grid(row=row, column=col, sticky="nsew", padx=6, pady=6)
        parent.rowconfigure(row, weight=1)
        parent.columnconfigure(col, weight=1)
        ttk.Label(frame, text=title, style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(frame, text=text, style="WelcomeCardBody.TLabel", wraplength=400, justify="left").pack(
            anchor="w", pady=(10, 0)
        )
