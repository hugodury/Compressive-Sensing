from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from frontend.theme import CARD
from frontend.visualize_patches import visualize_patches
from frontend.utils import open_path, path_to_photo
from .base_page import BasePage

# Aperçu : peu d’images, un peu plus grandes (lisibilité)
_PREVIEW_MAX = 4
_PREVIEW_SIZE = (520, 360)


def _preview_patch_pngs(folder: str) -> list[Path]:
    """Jusqu’à 4 fichiers : grille, début / fin des étapes, recomposition."""
    root = Path(folder)
    if not root.is_dir():
        return []
    all_p = list(root.glob("*.png"))
    grids = sorted(p for p in all_p if "_grid_" in p.name)
    steps = sorted((p for p in all_p if "_step_" in p.name), key=lambda p: p.name)
    finals = sorted(p for p in all_p if "_recompose_" in p.name)

    out: list[Path] = []
    if grids:
        out.append(grids[0])
    if len(steps) <= 1:
        out.extend(steps)
    elif steps:
        out.append(steps[0])
        if steps[-1] != steps[0]:
            out.append(steps[-1])
    if finals:
        out.append(finals[0])
    # Dédupliquer en gardant l’ordre
    seen: set[Path] = set()
    unique = []
    for p in out:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique[:_PREVIEW_MAX]


class PatchesPage(BasePage):
    def __init__(self, parent: tk.Misc, app) -> None:
        super().__init__(parent, app)
        self._photos: list = []
        self.vars = {
            "image_path": tk.StringVar(value=self.state.image_path),
            "B": tk.StringVar(value="8"),
            "max_steps": tk.StringVar(value="4"),
            "gap": tk.StringVar(value="2"),
            "out_dir": tk.StringVar(value=str(self.state.project_root / "Data" / "Result" / "patch_vis")),
        }
        self._build()

    def _build(self) -> None:
        ttk.Label(self, text="Visualisation du découpage en patchs", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            self,
            text=(
                "Découpage en patchs carrés B×B uniquement. Vue en grille avec espacement, images intermédiaires "
                "et recomposition finale. Les « étapes » empilent les blocs un par un (ordre de balayage) : "
                "aperçu du découpage, pas les itérations des solveurs."
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
            ("Nombre d’images d’étape (génération)", "max_steps"),
            ("Espacement (gap)", "gap"),
            ("Dossier de sortie", "out_dir"),
        ]
        for row, (label, key) in enumerate(fields):
            ttk.Label(top, text=label, style="App.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
            ttk.Entry(top, textvariable=self.vars[key], width=38).grid(row=row, column=1, sticky="ew", pady=4)
        top.columnconfigure(1, weight=1)
        ttk.Label(
            top,
            text=f"L’aperçu ci‑dessous montre au plus {_PREVIEW_MAX} images (grille, début/fin des étapes, recomposition).",
            style="CardMuted.TLabel",
            wraplength=720,
            justify="left",
        ).grid(row=len(fields), column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(top, text="Générer les visuels", style="Primary.TButton", command=self.generate).grid(
            row=len(fields) + 1, column=0, columnspan=2, sticky="ew", pady=(14, 0)
        )

        bottom = ttk.Frame(self, style="App.TFrame")
        bottom.pack(fill="both", expand=True, pady=(10, 0))
        self.gallery = ttk.Frame(bottom, style="Card.TFrame", padding=16)
        self.gallery.pack(fill="both", expand=True)
        self.gallery.columnconfigure(0, weight=1)

        head = ttk.Frame(self.gallery, style="Card.TFrame")
        head.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        head.columnconfigure(0, weight=1)
        ttk.Label(
            head,
            text="Aperçu (grille → étapes → recomposition)",
            style="CardTitle.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(head, text="Ouvrir le dossier", command=lambda: open_path(self.state.last_patch_dir or "")).grid(
            row=0, column=1, sticky="e", padx=(12, 0)
        )

        self.preview_frame = ttk.Frame(self.gallery, style="Card.TFrame")
        self.preview_frame.grid(row=1, column=0, sticky="nsew", pady=(4, 0))
        self.gallery.rowconfigure(1, weight=1)
        self.preview_frame.columnconfigure(0, weight=1)

    def generate(self) -> None:
        try:
            out_dir = self.vars["out_dir"].get().strip()
            visualize_patches(
                self.vars["image_path"].get().strip(),
                B=int(self.vars["B"].get()),
                max_steps=int(self.vars["max_steps"].get()),
                out_dir=out_dir,
                gap=int(self.vars["gap"].get()),
            )
            self.state.last_patch_dir = out_dir
            self.state.add_log(f"Visualisation patchs générée dans {out_dir}")
            self.refresh()
            messagebox.showinfo("Succès", "Images de patchs générées.")
        except Exception as exc:
            messagebox.showerror("Erreur", str(exc))

    def refresh(self) -> None:
        for widget in self.preview_frame.winfo_children():
            widget.destroy()
        self._photos = []
        folder = self.state.last_patch_dir
        if not folder or not Path(folder).is_dir():
            ttk.Label(
                self.preview_frame,
                text="Générez des visuels ou choisissez un dossier contenant des PNG.",
                style="CardMuted.TLabel",
            ).grid(row=0, column=0, sticky="w")
            return
        paths = _preview_patch_pngs(folder)
        if not paths:
            ttk.Label(
                self.preview_frame,
                text="Aucun PNG dans ce dossier.",
                style="CardMuted.TLabel",
            ).grid(row=0, column=0, sticky="w")
            return
        for i, path in enumerate(paths):
            block = ttk.Frame(self.preview_frame, style="Panel.TFrame", padding=10)
            block.grid(row=i, column=0, sticky="ew", padx=4, pady=8)
            ttk.Label(block, text=path.name, style="Section.TLabel").pack(anchor="w")
            try:
                photo = path_to_photo(str(path), _PREVIEW_SIZE)
            except OSError as e:
                ttk.Label(block, text=f"(lecture impossible : {e})", style="CardMuted.TLabel").pack(anchor="w", pady=6)
                continue
            self._photos.append(photo)
            # tk.Label : l’image s’affiche correctement (souvent cassé avec ttk.Label + clam)
            img_lbl = tk.Label(block, image=photo, bg=CARD, bd=0, highlightthickness=0)
            img_lbl.image = photo
            img_lbl.pack(anchor="w", pady=(6, 0))
