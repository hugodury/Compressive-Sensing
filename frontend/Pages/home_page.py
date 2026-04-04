from __future__ import annotations

import tkinter as tk
from tkinter import ttk

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
            "Le pipeline traite des patchs : mesure y = Φx, choix d’un dictionnaire D, puis solveurs parcimonieux "
            "(MP, OMP, StOMP, CoSaMP, IRLS, BP, LP, LASSO). Les sorties incluent PSNR, MSE, erreur relative, temps "
            "et une estimation d’empreinte carbone indicative.",
            "Quatre familles de matrices Φ sont disponibles (Φ₁ à Φ₄). Vous pouvez balayer plusieurs ratios de mesures, "
            "générer des tableaux de cohérence mutuelle et d’erreurs, et visualiser le découpage en patchs.",
        )
        for i, line in enumerate(intro_lines):
            ttk.Label(hero, text=line, style="WelcomeLead.TLabel", justify="left").grid(
                row=i, column=0, sticky="ew", pady=(0, 12 if i == 0 else 0)
            )

        cards = ttk.Frame(self, style="App.TFrame")
        cards.grid(row=3, column=0, sticky="nsew")
        cards.columnconfigure((0, 1), weight=1)
        self.rowconfigure(3, weight=1)

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
        recap.grid(row=4, column=0, sticky="ew", pady=(18, 0))
        recap.columnconfigure(0, weight=1)
        ttk.Label(recap, text="Fonctionnalités principales", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        bullets = (
            "Matrices Φ₁…Φ₄, dictionnaires DCT / mixte / K-SVD, patchs carrés B×B",
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
