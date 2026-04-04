from __future__ import annotations

import tkinter as tk
from tkinter import ttk

# Interface claire, sobre (lisibilité longue session, rendu « rapport / labo »)
BG = "#e8ecf2"
SURFACE = "#f7f8fb"
CARD = "#ffffff"
CARD_EDGE = "#d1d9e6"
TEXT = "#1a2332"
TEXT_SECONDARY = "#4a5568"
ACCENT = "#1e5a8e"
ACCENT_LIGHT = "#e8f0f8"
PRIMARY_BTN = "#1e5a8e"
PRIMARY_BTN_TEXT = "#ffffff"
TITLE_SZ = 20


def apply_theme(root: tk.Tk) -> None:
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    root.configure(bg=BG)

    style.configure("App.TFrame", background=BG)
    style.configure("Panel.TFrame", background=SURFACE)
    style.configure("Card.TFrame", background=CARD, relief="flat")
    style.configure("App.TLabel", background=BG, foreground=TEXT, font=("Ubuntu", 10))
    style.configure("Muted.TLabel", background=BG, foreground=TEXT_SECONDARY, font=("Ubuntu", 9))
    style.configure("Hint.TLabel", background=CARD, foreground=TEXT_SECONDARY, font=("Ubuntu", 8), wraplength=560)
    style.configure("Title.TLabel", background=BG, foreground=TEXT, font=("Ubuntu", TITLE_SZ, "bold"))
    style.configure(
        "WelcomeSubtitle.TLabel",
        background=BG,
        foreground=TEXT_SECONDARY,
        font=("Ubuntu", 11),
    )
    style.configure(
        "WelcomeHero.TFrame",
        background=CARD,
        relief="solid",
        borderwidth=1,
        bordercolor=CARD_EDGE,
    )
    style.configure(
        "WelcomeLead.TLabel",
        background=CARD,
        foreground=TEXT,
        font=("Ubuntu", 11),
        wraplength=960,
    )
    style.configure(
        "WelcomeCardBody.TLabel",
        background=CARD,
        foreground=TEXT,
        font=("Ubuntu", 10),
        wraplength=400,
    )
    style.configure(
        "WelcomeBullet.TLabel",
        background=CARD,
        foreground=TEXT_SECONDARY,
        font=("Ubuntu", 10),
        wraplength=900,
    )
    style.configure("Section.TLabel", background=SURFACE, foreground=ACCENT, font=("Ubuntu", 11, "bold"))
    style.configure("CardTitle.TLabel", background=CARD, foreground=ACCENT, font=("Ubuntu", 11, "bold"))
    style.configure("CardBody.TLabel", background=CARD, foreground=TEXT, font=("Ubuntu", 10))
    style.configure("CardMuted.TLabel", background=CARD, foreground=TEXT_SECONDARY, font=("Ubuntu", 9))
    style.configure("Stat.TLabel", background=CARD, foreground=ACCENT, font=("Ubuntu", 11, "bold"))

    style.configure("TButton", font=("Ubuntu", 10), padding=(12, 7))
    style.map("TButton", background=[("active", ACCENT_LIGHT)])

    style.configure("Primary.TButton", font=("Ubuntu", 10, "bold"), padding=(14, 9))
    style.map(
        "Primary.TButton",
        background=[("!disabled", PRIMARY_BTN), ("active", "#164a72"), ("pressed", "#134066")],
        foreground=[("!disabled", PRIMARY_BTN_TEXT), ("disabled", TEXT_SECONDARY)],
    )

    style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(6, 4, 0, 0))
    style.configure(
        "TNotebook.Tab",
        padding=(20, 11),
        font=("Ubuntu", 10, "bold"),
        background="#d1dae8",
        foreground=TEXT_SECONDARY,
        borderwidth=1,
        focuscolor="",
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", CARD), ("active", ACCENT_LIGHT), ("!selected", "#d1dae8")],
        foreground=[("selected", ACCENT), ("!selected", TEXT_SECONDARY)],
        lightcolor=[("selected", CARD_EDGE), ("!selected", "#c5cfde")],
        darkcolor=[("selected", CARD_EDGE), ("!selected", "#a8b4c8")],
    )

    style.configure("TEntry", fieldbackground="#ffffff", foreground=TEXT, padding=5, relief="solid", borderwidth=1)
    style.configure("TCombobox", fieldbackground="#ffffff", foreground=TEXT, padding=3, arrowsize=14)

    for rb in ("TCheckbutton", "TRadiobutton"):
        style.configure(rb, background=CARD, foreground=TEXT, font=("Ubuntu", 10))
        style.map(rb, background=[("active", CARD), ("selected", CARD)])

    style.configure("TLabelframe", background=CARD, foreground=ACCENT, borderwidth=1, relief="solid")
    style.configure("TLabelframe.Label", background=CARD, foreground=ACCENT, font=("Ubuntu", 10, "bold"))

    style.configure("Treeview", background="#ffffff", fieldbackground="#ffffff", foreground=TEXT, rowheight=24)
    style.configure("Treeview.Heading", font=("Ubuntu", 10, "bold"), background=ACCENT_LIGHT)

    style.configure("Vertical.TScrollbar", troughcolor=SURFACE, background=CARD_EDGE, arrowcolor=TEXT)
    style.configure("Horizontal.TScrollbar", troughcolor=SURFACE, background=CARD_EDGE, arrowcolor=TEXT)

    style.configure("TSeparator", background=CARD_EDGE)
