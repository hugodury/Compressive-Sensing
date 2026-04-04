from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .base_page import BasePage


class AboutPage(BasePage):
    def __init__(self, parent: tk.Misc, app) -> None:
        super().__init__(parent, app)
        ttk.Label(self, text="À propos du frontend", style="Title.TLabel").pack(anchor="w")

        card = ttk.Frame(self, style="Card.TFrame", padding=16)
        card.pack(fill="both", expand=True, pady=(12, 0))
        ttk.Label(card, text="Ce que contient cette interface", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            card,
            text=(
                "- Page d'accueil avec vue d'ensemble\n"
                "- Page de reconstruction complète\n"
                "- Page de résultats avec tableau et prévisualisations\n"
                "- Page de comparaison / sweep des ratios\n"
                "- Page de génération des tableaux section 6\n"
                "- Page de visualisation des patchs\n"
                "- Navigation par onglets\n"
                "- Exports automatiques via le backend existant"
            ),
            style="App.TLabel",
            justify="left",
        ).pack(anchor="w", pady=(10, 14))

        logs = ttk.Frame(card, style="Panel.TFrame", padding=12)
        logs.pack(fill="both", expand=True)
        ttk.Label(logs, text="Journal d'activité", style="Section.TLabel").pack(anchor="w")
        self.text = tk.Text(logs, height=18)
        self.text.pack(fill="both", expand=True, pady=(10, 0))

    def refresh(self) -> None:
        self.text.delete("1.0", tk.END)
        if not self.state.logs:
            self.text.insert(tk.END, "Aucune action pour le moment.")
            return
        for line in self.state.logs:
            self.text.insert(tk.END, f"- {line}\n")
