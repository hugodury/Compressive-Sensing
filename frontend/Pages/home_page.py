from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .base_page import BasePage


class HomePage(BasePage):
    def __init__(self, parent: tk.Misc, app) -> None:
        super().__init__(parent, app)

        ttk.Label(self, text="Compressive Sensing - Interface complète", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            self,
            text=(
                "Cette interface permet de piloter tout le projet : reconstruction BCS, comparaisons de méthodes, "
                "génération des tableaux de la section 6, visualisation des patchs et export des résultats."
            ),
            style="App.TLabel",
            wraplength=980,
            justify="left",
        ).pack(anchor="w", pady=(8, 18))

        cards = ttk.Frame(self, style="App.TFrame")
        cards.pack(fill="both", expand=True)
        cards.columnconfigure((0, 1), weight=1)

        self._make_card(
            cards,
            0,
            0,
            "1. Reconstruction",
            "Choisir l'image, la taille B, le ratio, les matrices de mesure, le dictionnaire et les méthodes à comparer.",
        )
        self._make_card(
            cards,
            0,
            1,
            "2. Résultats",
            "Afficher les métriques, les images reconstruites, les meilleurs PSNR, les temps et les exports.",
        )
        self._make_card(
            cards,
            1,
            0,
            "3. Comparaisons",
            "Lancer des sweeps sur plusieurs ratios et tracer les courbes PSNR selon les méthodes.",
        )
        self._make_card(
            cards,
            1,
            1,
            "4. Annexes du projet",
            "Générer les tableaux demandés dans le sujet et visualiser le découpage en patchs."
        )

        recap = ttk.Frame(self, style="Card.TFrame", padding=16)
        recap.pack(fill="x", pady=(18, 0))
        ttk.Label(recap, text="Ce que couvre le frontend", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            recap,
            text=(
                "- Paramétrage complet du pipeline\n"
                "- Sélection de plusieurs méthodes simultanément\n"
                "- Sauvegarde automatique des résultats\n"
                "- Graphiques de comparaison\n"
                "- Tableaux section 6\n"
                "- Visualisation du découpage patch par patch"
            ),
            style="App.TLabel",
            justify="left",
        ).pack(anchor="w", pady=(10, 0))

    def _make_card(self, parent: ttk.Frame, row: int, col: int, title: str, text: str) -> None:
        frame = ttk.Frame(parent, style="Card.TFrame", padding=16)
        frame.grid(row=row, column=col, sticky="nsew", padx=6, pady=6)
        parent.rowconfigure(row, weight=1)
        parent.columnconfigure(col, weight=1)
        ttk.Label(frame, text=title, style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(frame, text=text, style="App.TLabel", wraplength=430, justify="left").pack(anchor="w", pady=(8, 0))
