from __future__ import annotations

import tkinter as tk
from tkinter import ttk

BG = "#0f172a"
PANEL = "#111827"
CARD = "#1f2937"
TEXT = "#e5e7eb"
MUTED = "#9ca3af"
ACCENT = "#3b82f6"
ACCENT_2 = "#10b981"
BORDER = "#374151"


def apply_theme(root: tk.Tk) -> None:
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    root.configure(bg=BG)

    style.configure("App.TFrame", background=BG)
    style.configure("Panel.TFrame", background=PANEL)
    style.configure("Card.TFrame", background=CARD)
    style.configure("App.TLabel", background=BG, foreground=TEXT, font=("Arial", 10))
    style.configure("Muted.TLabel", background=BG, foreground=MUTED, font=("Arial", 10))
    style.configure("Title.TLabel", background=BG, foreground=TEXT, font=("Arial", 20, "bold"))
    style.configure("Section.TLabel", background=PANEL, foreground=TEXT, font=("Arial", 12, "bold"))
    style.configure("CardTitle.TLabel", background=CARD, foreground=TEXT, font=("Arial", 11, "bold"))
    style.configure("TButton", font=("Arial", 10), padding=8)
    style.configure("Primary.TButton", font=("Arial", 10, "bold"))
    style.map("Primary.TButton", background=[("!disabled", ACCENT)], foreground=[("!disabled", "white")])
    style.configure("TNotebook", background=BG, borderwidth=0)
    style.configure("TNotebook.Tab", padding=(12, 8), font=("Arial", 10, "bold"))
    style.configure("TEntry", fieldbackground="#ffffff")
    style.configure("TCombobox", fieldbackground="#ffffff")
    style.configure("Treeview", rowheight=24)
    style.configure("Treeview.Heading", font=("Arial", 10, "bold"))
