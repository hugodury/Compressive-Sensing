from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

from frontend.state import AppState
from frontend.theme import apply_theme
from frontend.utils import ensure_project_root
from frontend.Pages import (
    AboutPage,
    ComparisonPage,
    HomePage,
    PatchesPage,
    ReconstructionPage,
    ResultsPage,
    Section6Page,
)


class CompressiveSensingApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.root_path = ensure_project_root()
        default_image = self.root_path / "lena.jpg"
        default_output = self.root_path / "Data" / "Result"
        self.state = AppState(
            project_root=self.root_path,
            image_path=str(default_image),
            output_path=str(default_output),
        )

        self.title("Compressive Sensing - Frontend complet")
        self.geometry("1440x900")
        self.minsize(1200, 780)
        apply_theme(self)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.pages: dict[str, ttk.Frame] = {}
        self._add_page("Accueil", HomePage)
        self._add_page("Reconstruction", ReconstructionPage)
        self._add_page("Résultats", ResultsPage)
        self._add_page("Comparaisons", ComparisonPage)
        self._add_page("Tableaux section 6", Section6Page)
        self._add_page("Patchs", PatchesPage)
        self._add_page("À propos", AboutPage)

        self.refresh_all_pages()

    def _add_page(self, title: str, page_cls) -> None:
        page = page_cls(self.notebook, self)
        self.pages[title] = page
        self.notebook.add(page, text=title)

    def refresh_all_pages(self) -> None:
        for page in self.pages.values():
            if hasattr(page, "refresh"):
                page.refresh()

    def select_tab(self, title: str) -> None:
        page = self.pages.get(title)
        if page is not None:
            self.notebook.select(page)


def create_app() -> CompressiveSensingApp:
    return CompressiveSensingApp()


def launch_app() -> None:
    app = create_app()
    app.mainloop()


if __name__ == "__main__":
    launch_app()
