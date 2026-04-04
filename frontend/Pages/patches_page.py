from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from frontend.theme import CARD
from frontend.visualize_patches import visualize_patches
from frontend.utils import open_path, path_to_photo
from .base_page import BasePage

_PREVIEW_SIZE = (640, 480)


def _grille_patch_png(folder: str) -> Path | None:
    """Préfère ``grille_sur_image`` dans le dossier."""
    root = Path(folder)
    if not root.is_dir():
        return None
    for p in sorted(root.glob("*.png")):
        if "grille_sur_image" in p.name:
            return p
    return None


class PatchesPage(BasePage):
    def __init__(self, parent: tk.Misc, app) -> None:
        super().__init__(parent, app)
        self._photo = None
        self.vars = {
            "image_path": tk.StringVar(value=self.state.image_path),
            "B": tk.StringVar(value="8"),
            "out_dir": tk.StringVar(value=str(self.state.project_root / "Data" / "Result" / "patch_vis")),
        }
        self._build()

    def _build(self) -> None:
        ttk.Label(self, text="Visualisation du découpage en patchs", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            self,
            text=(
                "Un seul fichier PNG : l’image recadrée (comme pour la reconstruction) avec le quadrillage des blocs B×B. "
                "Même découpage que ``image_to_patch_vectors`` dans le pipeline."
            ),
            style="Muted.TLabel",
            wraplength=960,
            justify="left",
        ).pack(anchor="w", pady=(6, 12))

        top = ttk.Frame(self, style="Card.TFrame", padding=16)
        top.pack(fill="x")
        fields = [
            ("Image", "image_path"),
            ("Taille B", "B"),
            ("Dossier de sortie", "out_dir"),
        ]
        for row, (label, key) in enumerate(fields):
            ttk.Label(top, text=label, style="App.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
            ttk.Entry(top, textvariable=self.vars[key], width=38).grid(row=row, column=1, sticky="ew", pady=4)
        top.columnconfigure(1, weight=1)
        ttk.Button(top, text="Générer la grille des patchs", style="Primary.TButton", command=self.generate).grid(
            row=len(fields), column=0, columnspan=2, sticky="ew", pady=(14, 0)
        )

        bottom = ttk.Frame(self, style="App.TFrame")
        bottom.pack(fill="x", expand=False, pady=(10, 0))
        self.gallery = ttk.Frame(bottom, style="Card.TFrame", padding=16)
        self.gallery.pack(fill="x", expand=False)
        self.gallery.columnconfigure(0, weight=1)

        head = ttk.Frame(self.gallery, style="Card.TFrame")
        head.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        head.columnconfigure(0, weight=1)
        ttk.Label(head, text="Aperçu", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Button(head, text="Ouvrir le dossier", command=lambda: open_path(self.state.last_patch_dir or "")).grid(
            row=0, column=1, sticky="e", padx=(12, 0)
        )

        self.preview_frame = ttk.Frame(self.gallery, style="Card.TFrame")
        self.preview_frame.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self.preview_frame.columnconfigure(0, weight=1)

    def generate(self) -> None:
        try:
            out_dir = self.vars["out_dir"].get().strip()
            visualize_patches(
                self.vars["image_path"].get().strip(),
                B=int(self.vars["B"].get()),
                out_dir=out_dir,
            )
            self.state.last_patch_dir = out_dir
            self.state.add_log(f"Grille patchs : {out_dir}")
            self.refresh()
            messagebox.showinfo("Succès", "Image avec grille des patchs enregistrée.")
        except Exception as exc:
            messagebox.showerror("Erreur", str(exc))

    def refresh(self) -> None:
        for widget in self.preview_frame.winfo_children():
            widget.destroy()
        self._photo = None
        folder = self.state.last_patch_dir
        if not folder or not Path(folder).is_dir():
            ttk.Label(
                self.preview_frame,
                text="Générez la grille ou placez un PNG « grille_sur_image » dans le dossier.",
                style="CardMuted.TLabel",
            ).grid(row=0, column=0, sticky="w")
            return
        path = _grille_patch_png(folder)
        if path is None:
            ttk.Label(
                self.preview_frame,
                text="Aucun fichier « grille_sur_image » dans ce dossier.",
                style="CardMuted.TLabel",
            ).grid(row=0, column=0, sticky="w")
            return
        ttk.Label(self.preview_frame, text=path.name, style="Section.TLabel").grid(row=0, column=0, sticky="w")
        try:
            self._photo = path_to_photo(str(path), _PREVIEW_SIZE)
        except OSError as e:
            ttk.Label(self.preview_frame, text=f"Lecture impossible : {e}", style="CardMuted.TLabel").grid(
                row=1, column=0, sticky="w", pady=6
            )
            return
        img_lbl = tk.Label(self.preview_frame, image=self._photo, bg=CARD, bd=0, highlightthickness=0)
        img_lbl.image = self._photo
        img_lbl.grid(row=1, column=0, sticky="w", pady=(6, 0))
