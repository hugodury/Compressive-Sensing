from __future__ import annotations

# Permet ``python3 frontend/app.py`` : la racine du dépôt doit être sur sys.path.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_rp = str(_PROJECT_ROOT)
if _rp not in sys.path:
    sys.path.insert(0, _rp)

import logging
import traceback

import tkinter as tk
from tkinter import messagebox, ttk

from frontend.state import AppState
from frontend.theme import apply_theme
from frontend.utils import ensure_project_root
from frontend.Pages import (
    ComparisonPage,
    HomePage,
    PatchesPage,
    ReconstructionPage,
    ResultsPage,
    Section6Page,
)


class CompressiveSensingApp(tk.Tk):
    def report_callback_exception(self, exc: type[BaseException], val: BaseException, tb: object) -> None:
        """Évite une fermeture silencieuse : log + boîte de dialogue si possible."""
        logging.error("Exception dans un callback Tkinter", exc_info=(exc, val, tb))
        err = "".join(traceback.format_exception(exc, val, tb))
        try:
            messagebox.showerror("Erreur (interface)", err[:2000], parent=self)
        except tk.TclError:
            print(err, file=sys.stderr)
        super().report_callback_exception(exc, val, tb)

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

        self.title("Compressive sensing — interface projet")
        self.geometry("1280x820")
        self.minsize(920, 600)
        apply_theme(self)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=18, pady=(10, 16))

        self.pages: dict[str, ttk.Frame] = {}
        self._add_page("Accueil", HomePage)
        self._add_page("Reconstruction", ReconstructionPage)
        self._add_page("Résultats", ResultsPage)
        self._add_page("Comparaisons", ComparisonPage)
        self._add_page("Cohérence & erreurs", Section6Page)
        self._add_page("Patchs", PatchesPage)

        self.refresh_all_pages()

    def _add_page(self, title: str, page_cls) -> None:
        page = page_cls(self.notebook, self)
        self.pages[title] = page
        self.notebook.add(page, text=title)

    def refresh_all_pages(self) -> None:
        for title, page in self.pages.items():
            if not hasattr(page, "refresh"):
                continue
            try:
                page.refresh()
            except Exception:
                logging.exception("refresh() a échoué pour l’onglet %s", title)
                try:
                    messagebox.showerror(
                        "Erreur rafraîchissement",
                        f"Onglet « {title} » :\n{traceback.format_exc()[-1500:]}",
                        parent=self,
                    )
                except tk.TclError:
                    traceback.print_exc()

    def select_tab(self, title: str) -> None:
        page = self.pages.get(title)
        if page is not None:
            self.notebook.select(page)


def create_app() -> CompressiveSensingApp:
    return CompressiveSensingApp()


def launch_app() -> None:
    try:
        app = create_app()
    except Exception:
        err = traceback.format_exc()
        logging.basicConfig(level=logging.ERROR)
        logging.error("Échec création de l’app :\n%s", err)
        root = tk.Tk()
        root.withdraw()
        try:
            messagebox.showerror("Compressive Sensing — erreur au démarrage", err[-2500:], parent=root)
        finally:
            root.destroy()
        raise SystemExit(1) from None
    app.mainloop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    launch_app()
