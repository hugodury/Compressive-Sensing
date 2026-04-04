from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class BasePage(ttk.Frame):
    def __init__(self, parent: tk.Misc, app) -> None:
        super().__init__(parent, style="Panel.TFrame", padding=16)
        self.app = app
        self.state = app.state

    def refresh(self) -> None:
        return None
