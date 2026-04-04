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

from frontend.scroll_host import build_vertical_scroll_host
from frontend.state import AppState
from frontend.theme import apply_theme
from frontend.utils import ensure_project_root
from frontend.Pages import (
    AnalysesPage,
    HomePage,
    PatchesPage,
    ReconstructionPage,
    ResultsPage,
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
        # Onglet réel du Notebook (shell avec scroll) — ``select()`` n’accepte que ce widget, pas la page interne
        self._notebook_tab_widgets: dict[str, tk.Misc] = {}
        self._scroll_canvases: list[tk.Canvas] = []
        self._add_page("Accueil", HomePage)
        self._add_page("Reconstruction", ReconstructionPage)
        self._add_page("Résultats", ResultsPage)
        self._add_page("Analyses & graphiques", AnalysesPage)
        self._add_page("Patchs", PatchesPage)

        self._bind_global_mousewheel()
        self.refresh_all_pages()

    def register_scroll_canvas(self, canvas: tk.Canvas) -> None:
        if canvas not in self._scroll_canvases:
            self._scroll_canvases.append(canvas)

    def _closest_scroll_canvas(self, w: tk.Misc | None) -> tk.Canvas | None:
        if w is None:
            return None
        best: tk.Canvas | None = None
        best_d = 10_000
        for c in self._scroll_canvases:
            d = 0
            cur: tk.Misc | None = w
            while cur is not None:
                if cur is c:
                    if d < best_d:
                        best = c
                        best_d = d
                    break
                cur = getattr(cur, "master", None)
                d += 1
        return best

    def _bind_global_mousewheel(self) -> None:
        def on_wheel(ev: tk.Event) -> None:
            if getattr(ev, "num", None) == 4 or (getattr(ev, "delta", 0) or 0) > 0:
                delta = -1
            elif getattr(ev, "num", None) == 5 or (getattr(ev, "delta", 0) or 0) < 0:
                delta = 1
            else:
                return
            try:
                x, y = self.winfo_pointerxy()
                w = self.winfo_containing(x, y)
            except tk.TclError:
                return
            c = self._closest_scroll_canvas(w)
            if c is None:
                return
            try:
                if not c.winfo_ismapped():
                    return
            except tk.TclError:
                return
            c.yview_scroll(delta * 3, "units")

        self.bind_all("<MouseWheel>", on_wheel)
        self.bind_all("<Button-4>", on_wheel)
        self.bind_all("<Button-5>", on_wheel)

    def _add_page(self, title: str, page_cls) -> None:
        if page_cls is ReconstructionPage:
            page = page_cls(self.notebook, self)
            self.pages[title] = page
            self._notebook_tab_widgets[title] = page
            self.notebook.add(page, text=title)
            return

        shell = ttk.Frame(self.notebook, style="App.TFrame")
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(0, weight=1)
        canvas, inner, _vsb = build_vertical_scroll_host(shell)
        self.register_scroll_canvas(canvas)
        page = page_cls(inner, self)
        self.pages[title] = page
        self._notebook_tab_widgets[title] = shell
        # fill=both : évite une bande vide (fond clair) sous les pages hautes comme l’accueil poster
        page.pack(anchor="nw", fill="both", expand=True)
        self.notebook.add(shell, text=title)

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
        tab = self._notebook_tab_widgets.get(title)
        if tab is not None:
            self.notebook.select(tab)


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
