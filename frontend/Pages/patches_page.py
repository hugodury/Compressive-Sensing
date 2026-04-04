from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from frontend.visualize_patches import visualize_patches
from frontend.utils import open_path, path_to_photo
from .base_page import BasePage


class PatchesPage(BasePage):
    def __init__(self, parent: tk.Misc, app) -> None:
        super().__init__(parent, app)
        self._photos: list = []
        self.vars = {
            "image_path": tk.StringVar(value=self.state.image_path),
            "B": tk.StringVar(value="8"),
            "max_steps": tk.StringVar(value="6"),
            "gap": tk.StringVar(value="2"),
            "out_dir": tk.StringVar(value=str(self.state.project_root / "Data" / "Result" / "patch_vis")),
        }
        self._build()

    def _build(self) -> None:
        ttk.Label(self, text="Visualisation du découpage en patchs", style="Title.TLabel").pack(anchor="w")
        ttk.Label(self, text="Génère la grille des patchs, des étapes intermédiaires et l'image recomposée.", style="Muted.TLabel").pack(anchor="w", pady=(6, 12))

        top = ttk.Frame(self, style="Card.TFrame", padding=16)
        top.pack(fill="x")
        for row, (label, key) in enumerate([
            ("Image", "image_path"),
            ("B", "B"),
            ("max_steps", "max_steps"),
            ("gap", "gap"),
            ("Dossier de sortie", "out_dir"),
        ]):
            ttk.Label(top, text=label, style="App.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
            ttk.Entry(top, textvariable=self.vars[key], width=38).grid(row=row, column=1, sticky="ew", pady=4)
        top.columnconfigure(1, weight=1)
        ttk.Button(top, text="Générer les visuels", style="Primary.TButton", command=self.generate).grid(row=5, column=0, columnspan=2, sticky="ew", pady=(14, 0))

        bottom = ttk.Frame(self, style="App.TFrame")
        bottom.pack(fill="both", expand=True, pady=(10, 0))
        self.gallery = ttk.Frame(bottom, style="Card.TFrame", padding=16)
        self.gallery.pack(fill="both", expand=True)
        ttk.Label(self.gallery, text="Aperçu", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Button(self.gallery, text="Ouvrir le dossier", command=lambda: open_path(self.state.last_patch_dir)).pack(anchor="e")
        self.preview_frame = ttk.Frame(self.gallery, style="Card.TFrame")
        self.preview_frame.pack(fill="both", expand=True, pady=(10, 0))

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
        if not folder:
            return
        paths = sorted(Path(folder).glob("*.png"))[:6]
        for i, path in enumerate(paths):
            frame = ttk.Frame(self.preview_frame, style="Panel.TFrame", padding=8)
            frame.grid(row=i // 2, column=i % 2, sticky="nsew", padx=6, pady=6)
            ttk.Label(frame, text=path.name, style="Section.TLabel").pack(anchor="w")
            photo = path_to_photo(str(path), (360, 220))
            self._photos.append(photo)
            ttk.Label(frame, image=photo).pack(anchor="center", pady=(8, 0))
