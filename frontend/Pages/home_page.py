from __future__ import annotations

import math
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageTk

from .base_page import BasePage


POSTER_BG = "#0b1220"
CARD = "#172033"
CARD_ALT = "#13243a"
CARD_GREEN = "#122b1d"
CARD_RED = "#35171d"
CARD_AMBER = "#3c2a12"

TEXT = "#e5e7eb"
MUTED = "#9fb0c7"
ACCENT = "#38bdf8"
ACCENT_2 = "#22c55e"
ACCENT_3 = "#f59e0b"
BORDER = "#334155"

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS


class HomePage(BasePage):
    def __init__(self, parent: tk.Misc, app) -> None:
        super().__init__(parent, app)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._photos: list[ImageTk.PhotoImage] = []

        container = ttk.Frame(self, style="Panel.TFrame")
        container.grid(row=0, column=0, sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            container,
            bg=POSTER_BG,
            highlightthickness=0,
            bd=0,
        )
        self.vscroll = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vscroll.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vscroll.grid(row=0, column=1, sticky="ns")

        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", self._on_linux_scroll_up)
        self.canvas.bind("<Button-5>", self._on_linux_scroll_down)
        self.canvas.bind("<Enter>", lambda e: self.canvas.focus_set())

        self._build_poster()

    def refresh(self) -> None:
        return None

    def _on_mousewheel(self, event: tk.Event) -> None:
        try:
            delta = int(-event.delta / 120)
            if delta != 0:
                self.canvas.yview_scroll(delta, "units")
        except Exception:
            pass

    def _on_linux_scroll_up(self, event: tk.Event) -> None:
        self.canvas.yview_scroll(-3, "units")

    def _on_linux_scroll_down(self, event: tk.Event) -> None:
        self.canvas.yview_scroll(3, "units")

    def _build_poster(self) -> None:
        self.canvas.delete("all")
        self._photos.clear()

        width = 1320
        height = 1520
        self.canvas.config(scrollregion=(0, 0, width, height))
        self.canvas.create_rectangle(0, 0, width, height, fill=POSTER_BG, outline="")

        self.canvas.create_text(
            24,
            20,
            text="Pipeline complet du projet de Compressive Sensing sur Lena",
            anchor="nw",
            fill=TEXT,
            font=("Arial", 24, "bold"),
        )
        self.canvas.create_text(
            24,
            60,
            text=(
                "Schéma unique, complet et illustré : image Lena, patchs, vectorisation, acquisition compressée, "
                "dictionnaires DCT / K-SVD, résolution sparse, reconstruction patch par patch, image finale, "
                "métriques et comparaisons demandées dans le projet."
            ),
            anchor="nw",
            fill=MUTED,
            font=("Arial", 11),
            width=1240,
        )

        lena = self._load_lena()
        grid_img = self._make_grid_overlay(lena, grid=8, selected=(2, 3))
        patch_img = self._make_patch_zoom(lena, cell=(2, 3), grid=8)
        patch_small = patch_img.convert("L").resize((8, 8), RESAMPLE)

        x = self._patch_to_vector(patch_small)
        dct_dict = self._build_dct_dictionary(8)
        m = 24
        phi = self._build_measurement_matrix(m, x.size, seed=7)
        y = phi @ x
        a = phi @ dct_dict

        alpha_dense = dct_dict.T @ x
        alpha_sparse = self._keep_top_k(alpha_dense, 10)
        x_hat = dct_dict @ alpha_sparse

        patch_hat = np.clip(x_hat.reshape(8, 8), 0, 255)
        patch_hat_img = (
            Image.fromarray(patch_hat.astype(np.uint8), mode="L")
            .convert("RGB")
            .resize((220, 220), RESAMPLE)
        )
        recon_img = self._insert_reconstructed_patch(lena, patch_hat, cell=(2, 3), grid=8)

        vector_img = self._make_vector_strip_from_values(x, width=80, bar_h=5, color_mode="gray")
        measurement_img = self._make_measurement_panel(phi, y)
        dct_dict_img = self._make_dct_mosaic_image(8, atoms=16, tile=28)
        ksvd_like_img = self._make_ksvd_like_mosaic(lena, tile=28)
        a_img = self._make_heatmap_image(np.abs(a), size=(170, 170), palette="cyan")
        alpha_img = self._make_vector_strip_from_values(alpha_sparse, width=80, bar_h=7, color_mode="magenta")
        xhat_img = self._make_vector_strip_from_values(x_hat, width=80, bar_h=5, color_mode="green")

        compare_patch_img = self._make_side_by_side(
            patch_img.resize((170, 170), RESAMPLE),
            patch_hat_img.resize((170, 170), RESAMPLE),
            "Patch original",
            "Patch reconstruit",
        )
        compare_image_img = self._make_side_by_side(
            ImageOps.contain(lena.copy(), (170, 170), RESAMPLE),
            ImageOps.contain(recon_img.copy(), (170, 170), RESAMPLE),
            "Image originale",
            "Image reconstruite",
        )
        summary_img = self._make_summary_outputs_image()

        x_col = [30, 465, 900]
        y_row = [125, 470, 815, 1160]
        card_w = 390
        card_h = 300

        self._draw_step_card(
            x_col[0],
            y_row[0],
            card_w,
            card_h,
            "1. Image d'entrée",
            "Image Lena utilisée comme signal d'entrée.",
            fill=CARD,
        )
        self._place_image(x_col[0] + 90, y_row[0] + 95, self._thumb(lena, (210, 170)))
        self._formula(x_col[0] + 22, y_row[0] + card_h - 36, "I ∈ R^(H×W)")

        self._draw_step_card(
            x_col[1],
            y_row[0],
            card_w,
            card_h,
            "2. Découpage BCS en patchs",
            "L'image est découpée en blocs B×B sans chevauchement.",
            fill=CARD,
        )
        self._place_image(x_col[1] + 65, y_row[0] + 88, self._thumb(grid_img, (260, 180)))
        self._formula(x_col[1] + 22, y_row[0] + card_h - 36, "extractPatch(I, B) → {P₁, P₂, ..., Pn}")

        self._draw_step_card(
            x_col[2],
            y_row[0],
            card_w,
            card_h,
            "3. Patch sélectionné",
            "Zoom réel sur le patch choisi dans Lena.",
            fill=CARD,
        )
        self._place_image(x_col[2] + 86, y_row[0] + 88, self._thumb(patch_img, (220, 180)))
        self._formula(x_col[2] + 22, y_row[0] + card_h - 36, "Pᵢ ∈ R^(B×B)")

        self._draw_step_card(
            x_col[2],
            y_row[1],
            card_w,
            card_h,
            "4. Vectorisation",
            "Le patch B×B est aplati en vecteur x de taille N = B².",
            fill=CARD_ALT,
        )
        self._place_image(x_col[2] + 28, y_row[1] + 102, self._thumb(patch_img, (120, 120)))
        self._place_image(x_col[2] + 208, y_row[1] + 88, self._thumb(vector_img, (95, 150)))
        self.canvas.create_text(
            x_col[2] + 158,
            y_row[1] + 156,
            text="→",
            fill=ACCENT,
            font=("Arial", 28, "bold"),
        )
        self._formula(x_col[2] + 22, y_row[1] + card_h - 36, "x = vec(Pᵢ) ∈ R^N")

        self._draw_step_card(
            x_col[1],
            y_row[1],
            card_w,
            card_h,
            "5. Acquisition compressée",
            "Application d'une vraie matrice Φ sur x pour produire y.",
            fill=CARD_GREEN,
        )
        self._place_image(x_col[1] + 44, y_row[1] + 88, self._thumb(measurement_img, (300, 170)))
        self._formula(x_col[1] + 22, y_row[1] + card_h - 36, "y = Φx,   Φ ∈ R^(M×N),   y ∈ R^M")

        self._draw_step_card(
            x_col[0],
            y_row[1],
            card_w,
            card_h,
            "6. Dictionnaire du signal",
            "Comparaison visuelle entre DCT et K-SVD-like issu de vrais patchs.",
            fill=CARD_GREEN,
        )
        self._place_image(x_col[0] + 28, y_row[1] + 95, self._thumb(dct_dict_img, (140, 140)))
        self._place_image(x_col[0] + 218, y_row[1] + 95, self._thumb(ksvd_like_img, (140, 140)))
        self.canvas.create_text(
            x_col[0] + 98,
            y_row[1] + 86,
            text="DCT",
            fill=TEXT,
            font=("Arial", 12, "bold"),
        )
        self.canvas.create_text(
            x_col[0] + 272,
            y_row[1] + 86,
            text="K-SVD-like",
            fill=TEXT,
            font=("Arial", 12, "bold"),
        )
        self._formula(x_col[0] + 22, y_row[1] + card_h - 36, "x ≈ Dα,   D ∈ R^(N×K)")

        self._draw_step_card(
            x_col[0],
            y_row[2],
            card_w,
            card_h,
            "7. Problème sparse",
            "On construit A = ΦD puis on cherche α tel que y ≈ Aα.",
            fill=CARD_ALT,
        )
        self._place_image(x_col[0] + 24, y_row[2] + 90, self._thumb(a_img, (170, 170)))
        self.canvas.create_text(
            x_col[0] + 220,
            y_row[2] + 96,
            text="Méthodes :\n• MP\n• OMP\n• StOMP\n• CoSaMP\n• IRLS",
            anchor="nw",
            fill=TEXT,
            font=("Arial", 12),
        )
        self._formula(x_col[0] + 22, y_row[2] + card_h - 56, "A = ΦD")
        self._formula(x_col[0] + 22, y_row[2] + card_h - 36, "y ≈ Aα")

        self._draw_step_card(
            x_col[1],
            y_row[2],
            card_w,
            card_h,
            "8. Coefficients α",
            "Sortie sparse réelle obtenue en gardant les plus grands coefficients.",
            fill=CARD_RED,
        )
        self._place_image(x_col[1] + 150, y_row[2] + 90, self._thumb(alpha_img, (90, 165)))
        self._formula(x_col[1] + 22, y_row[2] + card_h - 36, "α sparse")

        self._draw_step_card(
            x_col[2],
            y_row[2],
            card_w,
            card_h,
            "9. Reconstruction du vecteur",
            "Le signal reconstruit x̂ est obtenu à partir de α et du dictionnaire.",
            fill=CARD_AMBER,
        )
        self._place_image(x_col[2] + 150, y_row[2] + 90, self._thumb(xhat_img, (90, 165)))
        self._formula(x_col[2] + 22, y_row[2] + card_h - 36, "x̂ = Dα")

        self._draw_step_card(
            x_col[2],
            y_row[3],
            card_w,
            card_h,
            "10. Retour au patch 2D",
            "Le vecteur reconstruit est remis sous forme matricielle B×B.",
            fill=CARD_AMBER,
        )
        self._place_image(x_col[2] + 28, y_row[3] + 90, self._thumb(compare_patch_img, (320, 165)))
        self._formula(x_col[2] + 22, y_row[3] + card_h - 36, "P̂ᵢ = reshape(x̂, B, B)")

        self._draw_step_card(
            x_col[1],
            y_row[3],
            card_w,
            card_h,
            "11. Réinsertion dans l'image",
            "Tous les patchs reconstruits reforment l'image finale.",
            fill=CARD_AMBER,
        )
        self._place_image(x_col[1] + 28, y_row[3] + 90, self._thumb(compare_image_img, (320, 165)))
        self._formula(x_col[1] + 22, y_row[3] + card_h - 36, "Î = reconstructImage({P̂₁, ..., P̂n})")

        self._draw_step_card(
            x_col[0],
            y_row[3],
            card_w,
            card_h,
            "12. Sorties et comparaisons",
            "Métriques, graphes et variables expérimentales attendues dans le projet.",
            fill=CARD_ALT,
        )
        self._place_image(x_col[0] + 20, y_row[3] + 88, self._thumb(summary_img, (350, 118)))
        self.canvas.create_text(
            x_col[0] + 22,
            y_row[3] + 216,
            text=(
                "À comparer : B, ratio P, matrices Φ1..Φ4, dictionnaires DCT / K-SVD,\n"
                "méthodes MP / OMP / StOMP / CoSaMP / IRLS, puis MSE / PSNR /\n"
                "erreur relative / temps."
            ),
            anchor="nw",
            fill=TEXT,
            font=("Arial", 10),
            width=350,
        )

        self._connect_cards_snake(x_col, y_row, card_w, card_h)

    def _connect_cards_snake(self, x_col: list[int], y_row: list[int], cw: int, ch: int) -> None:
        self._arrow(x_col[0] + cw, y_row[0] + ch // 2, x_col[1], y_row[0] + ch // 2, "patchs")
        self._arrow(x_col[1] + cw, y_row[0] + ch // 2, x_col[2], y_row[0] + ch // 2, "choix")
        self._arrow(x_col[2] + cw // 2, y_row[0] + ch, x_col[2] + cw // 2, y_row[1], "vec")
        self._arrow(x_col[2], y_row[1] + ch // 2, x_col[1] + cw, y_row[1] + ch // 2, "mesure")
        self._arrow(x_col[1], y_row[1] + ch // 2, x_col[0] + cw, y_row[1] + ch // 2, "D")
        self._arrow(x_col[0] + cw // 2, y_row[1] + ch, x_col[0] + cw // 2, y_row[2], "A = ΦD")
        self._arrow(x_col[0] + cw, y_row[2] + ch // 2, x_col[1], y_row[2] + ch // 2, "solve")
        self._arrow(x_col[1] + cw, y_row[2] + ch // 2, x_col[2], y_row[2] + ch // 2, "x̂")
        self._arrow(x_col[2] + cw // 2, y_row[2] + ch, x_col[2] + cw // 2, y_row[3], "reshape")
        self._arrow(x_col[2], y_row[3] + ch // 2, x_col[1] + cw, y_row[3] + ch // 2, "merge")
        self._arrow(x_col[1], y_row[3] + ch // 2, x_col[0] + cw, y_row[3] + ch // 2, "analyse")

    def _draw_step_card(self, x: int, y: int, w: int, h: int, title: str, desc: str, fill: str) -> None:
        self.canvas.create_rectangle(x, y, x + w, y + h, fill=fill, outline=BORDER, width=2)
        self.canvas.create_text(
            x + 16,
            y + 16,
            text=title,
            anchor="nw",
            fill=TEXT,
            font=("Arial", 15, "bold"),
            width=w - 32,
        )
        self.canvas.create_text(
            x + 16,
            y + 48,
            text=desc,
            anchor="nw",
            fill=MUTED,
            font=("Arial", 10),
            width=w - 32,
        )

    def _formula(self, x: int, y: int, text: str) -> None:
        self.canvas.create_text(
            x,
            y,
            text=text,
            anchor="nw",
            fill=ACCENT_3,
            font=("Arial", 11, "bold"),
        )

    def _arrow(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        label: str = "",
        color: str = ACCENT,
    ) -> None:
        self.canvas.create_line(
            x1,
            y1,
            x2,
            y2,
            fill=color,
            width=3,
            arrow=tk.LAST,
            smooth=True,
        )
        if label:
            self.canvas.create_text(
                (x1 + x2) / 2,
                (y1 + y2) / 2 - 12,
                text=label,
                fill=MUTED,
                font=("Arial", 10, "italic"),
            )

    def _place_image(self, x: int, y: int, img: Image.Image) -> None:
        photo = ImageTk.PhotoImage(img)
        self._photos.append(photo)
        self.canvas.create_image(x, y, image=photo, anchor="nw")

    def _load_lena(self) -> Image.Image:
        candidates = [
            self.state.project_root / "lena.jpg",
            self.state.project_root / "Data" / "Images" / "lena.jpg",
            self.state.project_root / "Data" / "Images" / "Lena.jpg",
        ]
        for path in candidates:
            if path.exists():
                return ImageOps.grayscale(Image.open(path)).convert("RGB")
        return Image.new("RGB", (512, 512), "#666666")

    def _thumb(self, img: Image.Image, size: tuple[int, int]) -> Image.Image:
        return ImageOps.contain(img.copy(), size, RESAMPLE)

    def _patch_to_vector(self, patch_small: Image.Image) -> np.ndarray:
        arr = np.asarray(patch_small.convert("L"), dtype=np.float64)
        return arr.reshape(-1)

    def _build_measurement_matrix(self, m: int, n: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        phi = rng.choice([-1.0, 1.0], size=(m, n)) / math.sqrt(m)
        return phi

    def _keep_top_k(self, alpha: np.ndarray, k: int) -> np.ndarray:
        alpha = alpha.copy()
        if k >= alpha.size:
            return alpha
        idx = np.argsort(np.abs(alpha))[:-k]
        alpha[idx] = 0.0
        return alpha

    def _build_dct_dictionary(self, n: int) -> np.ndarray:
        c = np.zeros((n, n), dtype=np.float64)
        for k in range(n):
            for i in range(n):
                if k == 0:
                    c[i, k] = 1.0 / math.sqrt(n)
                else:
                    c[i, k] = math.sqrt(2.0 / n) * math.cos(((2 * i + 1) * k * math.pi) / (2 * n))
        return np.kron(c, c)

    def _make_grid_overlay(
        self,
        img: Image.Image,
        grid: int = 8,
        selected: tuple[int, int] = (2, 3),
    ) -> Image.Image:
        vis = img.copy().resize((280, 280), RESAMPLE)
        draw = ImageDraw.Draw(vis)
        step = vis.width // grid

        for i in range(1, grid):
            x = i * step
            draw.line((x, 0, x, vis.height), fill="#60a5fa", width=1)
            draw.line((0, x, vis.width, x), fill="#60a5fa", width=1)

        x1 = selected[0] * step
        y1 = selected[1] * step
        draw.rectangle((x1, y1, x1 + step, y1 + step), outline="#f59e0b", width=4)
        return vis

    def _make_patch_zoom(
        self,
        img: Image.Image,
        cell: tuple[int, int] = (2, 3),
        grid: int = 8,
    ) -> Image.Image:
        gray = img.convert("L")
        size = min(gray.width, gray.height)
        step = size // grid
        x1 = cell[0] * step
        y1 = cell[1] * step

        patch = gray.crop((x1, y1, x1 + step, y1 + step)).resize((220, 220), RESAMPLE).convert("RGB")

        draw = ImageDraw.Draw(patch)
        sub = 8
        sub_step = patch.width // sub
        for i in range(1, sub):
            x = i * sub_step
            draw.line((x, 0, x, patch.height), fill="#64748b", width=1)
            draw.line((0, x, patch.width, x), fill="#64748b", width=1)
        draw.rectangle((0, 0, patch.width - 1, patch.height - 1), outline="#e5e7eb", width=2)
        return patch

    def _make_vector_strip_from_values(
        self,
        values: np.ndarray,
        width: int = 80,
        bar_h: int = 5,
        color_mode: str = "gray",
    ) -> Image.Image:
        vals = np.asarray(values, dtype=np.float64).reshape(-1)
        if vals.size == 0:
            vals = np.zeros(1, dtype=np.float64)

        vmin = float(vals.min())
        vmax = float(vals.max())
        if abs(vmax - vmin) < 1e-12:
            norm = np.zeros_like(vals)
        else:
            norm = (vals - vmin) / (vmax - vmin)

        img = Image.new("RGB", (width, len(vals) * bar_h), "#0b1220")
        draw = ImageDraw.Draw(img)

        for i, v in enumerate(norm):
            y = i * bar_h

            if color_mode == "gray":
                c = int(255 * v)
                fill = (c, c, c)
            elif color_mode == "magenta":
                fill = (
                    min(255, int(70 + 185 * v)),
                    40,
                    min(255, int(90 + 150 * v)),
                )
            elif color_mode == "green":
                fill = (
                    40,
                    min(255, int(70 + 185 * v)),
                    120,
                )
            else:
                c = int(255 * v)
                fill = (c, c, c)

            draw.rectangle((18, y, width - 18, y + bar_h - 1), fill=fill, outline="#1f2937")

        draw.rectangle((18, 0, width - 18, img.height - 1), outline="#e5e7eb", width=2)
        return img

    def _make_heatmap_image(
        self,
        mat: np.ndarray,
        size: tuple[int, int],
        palette: str = "cyan",
    ) -> Image.Image:
        arr = np.asarray(mat, dtype=np.float64)
        if arr.ndim != 2:
            arr = np.atleast_2d(arr)

        amin = float(arr.min())
        amax = float(arr.max())
        if abs(amax - amin) < 1e-12:
            norm = np.zeros_like(arr)
        else:
            norm = (arr - amin) / (amax - amin)

        h, w = norm.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        if palette == "cyan":
            rgb[..., 0] = (25 + 40 * norm).astype(np.uint8)
            rgb[..., 1] = (90 + 140 * norm).astype(np.uint8)
            rgb[..., 2] = (120 + 120 * norm).astype(np.uint8)
        elif palette == "magenta":
            rgb[..., 0] = (70 + 180 * norm).astype(np.uint8)
            rgb[..., 1] = (30 + 40 * norm).astype(np.uint8)
            rgb[..., 2] = (90 + 130 * norm).astype(np.uint8)
        else:
            c = (255 * norm).astype(np.uint8)
            rgb[..., 0] = c
            rgb[..., 1] = c
            rgb[..., 2] = c

        img = Image.fromarray(rgb, mode="RGB").resize(size, RESAMPLE)
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, img.width - 1, img.height - 1), outline="#e5e7eb", width=2)
        return img

    def _make_measurement_panel(self, phi: np.ndarray, y: np.ndarray) -> Image.Image:
        phi_img = self._make_heatmap_image(np.abs(phi), size=(165, 145), palette="cyan")
        y_img = self._make_vector_strip_from_values(y, width=70, bar_h=5, color_mode="green")

        panel = Image.new("RGB", (280, 170), "#0b1220")
        panel.paste(phi_img, (5, 12))
        panel.paste(ImageOps.contain(y_img, (70, 145), RESAMPLE), (200, 12))

        draw = ImageDraw.Draw(panel)
        draw.text((60, 0), "Phi", fill="#e5e7eb")
        draw.text((226, 0), "y", fill="#e5e7eb")
        return panel

    def _make_dct_mosaic_image(self, n: int = 8, atoms: int = 16, tile: int = 28) -> Image.Image:
        c = np.zeros((n, n), dtype=np.float64)
        for k in range(n):
            for i in range(n):
                if k == 0:
                    c[i, k] = 1.0 / math.sqrt(n)
                else:
                    c[i, k] = math.sqrt(2.0 / n) * math.cos(((2 * i + 1) * k * math.pi) / (2 * n))

        chosen = [
            (0, 0), (0, 1), (1, 0), (1, 1),
            (0, 2), (2, 0), (1, 2), (2, 1),
            (2, 2), (0, 3), (3, 0), (1, 3),
            (3, 1), (2, 3), (3, 2), (3, 3),
        ]

        margin = 6
        cols = 4
        rows = 4
        img = Image.new(
            "RGB",
            (cols * tile + (cols + 1) * margin, rows * tile + (rows + 1) * margin),
            "#102014",
        )
        draw = ImageDraw.Draw(img)

        for idx, (u, v) in enumerate(chosen[:atoms]):
            atom = np.outer(c[:, u], c[:, v])
            atom = (atom - atom.min()) / (atom.max() - atom.min() + 1e-12)
            atom_img = (
                Image.fromarray((atom * 255).astype(np.uint8), mode="L")
                .convert("RGB")
                .resize((tile, tile), Image.NEAREST)
            )

            r = idx // cols
            col = idx % cols
            x = margin + col * (tile + margin)
            y = margin + r * (tile + margin)
            img.paste(atom_img, (x, y))
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline="#334155", width=1)

        return img

    def _make_ksvd_like_mosaic(self, img: Image.Image, tile: int = 28) -> Image.Image:
        gray = img.convert("L")
        size = min(gray.width, gray.height)
        grid = 8
        step = size // grid

        positions = [
            (0, 0), (1, 0), (2, 1), (3, 1),
            (1, 2), (2, 2), (3, 2), (4, 2),
            (2, 3), (3, 3), (4, 3), (5, 3),
            (3, 4), (4, 4), (5, 4), (6, 4),
        ]

        margin = 6
        cols = 4
        rows = 4
        out = Image.new(
            "RGB",
            (cols * tile + (cols + 1) * margin, rows * tile + (rows + 1) * margin),
            "#102014",
        )
        draw = ImageDraw.Draw(out)

        for idx, (cx, cy) in enumerate(positions):
            x1 = cx * step
            y1 = cy * step
            patch = gray.crop((x1, y1, x1 + step, y1 + step)).resize((tile, tile), RESAMPLE).convert("RGB")

            r = idx // cols
            col = idx % cols
            x = margin + col * (tile + margin)
            y = margin + r * (tile + margin)
            out.paste(patch, (x, y))
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline="#334155", width=1)

        return out

    def _insert_reconstructed_patch(
        self,
        img: Image.Image,
        patch_hat: np.ndarray,
        cell: tuple[int, int] = (2, 3),
        grid: int = 8,
    ) -> Image.Image:
        gray = img.convert("L").copy()
        arr = np.asarray(gray, dtype=np.uint8).copy()

        size = min(arr.shape[0], arr.shape[1])
        step = size // grid
        x1 = cell[0] * step
        y1 = cell[1] * step

        patch_big = Image.fromarray(
            np.clip(patch_hat, 0, 255).astype(np.uint8),
            mode="L",
        ).resize((step, step), RESAMPLE)

        patch_big_arr = np.asarray(patch_big, dtype=np.uint8)
        arr[y1:y1 + step, x1:x1 + step] = patch_big_arr

        out = Image.fromarray(arr, mode="L").convert("RGB").resize((280, 280), RESAMPLE)
        draw = ImageDraw.Draw(out)

        step2 = out.width // grid
        x2 = cell[0] * step2
        y2 = cell[1] * step2
        draw.rectangle((x2, y2, x2 + step2, y2 + step2), outline="#22c55e", width=4)
        return out

    def _make_side_by_side(
        self,
        left: Image.Image,
        right: Image.Image,
        label_left: str,
        label_right: str,
    ) -> Image.Image:
        panel = Image.new("RGB", (360, 185), "#0b1220")
        panel.paste(ImageOps.contain(left, (160, 140), RESAMPLE), (10, 28))
        panel.paste(ImageOps.contain(right, (160, 140), RESAMPLE), (190, 28))

        draw = ImageDraw.Draw(panel)
        draw.text((42, 6), label_left, fill="#e5e7eb")
        draw.text((220, 6), label_right, fill="#e5e7eb")
        draw.line((180, 20, 180, 176), fill="#334155", width=2)
        return panel

    def _make_summary_outputs_image(self) -> Image.Image:
        img = Image.new("RGB", (350, 120), "#0b1220")
        draw = ImageDraw.Draw(img)

        draw.rectangle((10, 14, 160, 108), outline="#334155", width=2)
        draw.text((18, 18), "PSNR vs ratio", fill="#e5e7eb")
        draw.line((28, 90, 145, 90), fill="#64748b", width=1)
        draw.line((28, 90, 28, 34), fill="#64748b", width=1)

        pts = [(30, 84), (55, 78), (78, 72), (98, 66), (120, 55), (142, 44)]
        for i in range(len(pts) - 1):
            draw.line((pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]), fill="#22c55e", width=3)

        draw.rectangle((185, 14, 338, 108), outline="#334155", width=2)
        draw.text((195, 18), "Métriques", fill="#e5e7eb")

        metrics = [
            ("MSE", "12.4"),
            ("PSNR", "27.8"),
            ("Err", "0.09"),
            ("Temps", "0.42s"),
        ]
        y = 42
        for key, value in metrics:
            draw.text((198, y), key, fill="#9fb0c7")
            draw.text((286, y), value, fill="#f59e0b")
            y += 16

        return img



























