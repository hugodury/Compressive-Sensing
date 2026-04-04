from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from main import main as run_main
from backend.utils.save import save_results
from frontend.utils import latest_subdir, parse_float, parse_int
from .base_page import BasePage


class ReconstructionPage(BasePage):
    def __init__(self, parent: tk.Misc, app) -> None:
        super().__init__(parent, app)
        self.vars: dict[str, tk.Variable] = {}
        self.method_vars: dict[str, tk.BooleanVar] = {}

        self._build()
        self._set_defaults()

    def _build(self) -> None:
        container = ttk.Frame(self, style="App.TFrame")
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=3)
        container.columnconfigure(1, weight=2)

        left = ttk.Frame(container, style="Panel.TFrame")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right = ttk.Frame(container, style="Panel.TFrame")
        right.grid(row=0, column=1, sticky="nsew")

        form = ttk.Frame(left, style="Card.TFrame", padding=16)
        form.pack(fill="x")
        ttk.Label(form, text="Paramètres de reconstruction", style="CardTitle.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 12))

        self._entry_row(form, 1, "Image", "image_path", browse_file=True)
        self._entry_row(form, 2, "Image d'entraînement dictionnaire", "dictionary_train_image_path", browse_file=True)
        self._entry_row(form, 3, "Dossier de sortie", "output_path", browse_dir=True)

        self.vars["block_size"] = tk.StringVar()
        self.vars["ratio"] = tk.StringVar()
        self.vars["n_atoms"] = tk.StringVar()
        self.vars["n_iter_ksvd"] = tk.StringVar()
        self.vars["seed"] = tk.StringVar()
        self.vars["max_patches"] = tk.StringVar()
        self.vars["max_iter"] = tk.StringVar()
        self.vars["epsilon"] = tk.StringVar()
        self.vars["t_stomp"] = tk.StringVar()
        self.vars["s_cosamp"] = tk.StringVar()
        self.vars["norm_p"] = tk.StringVar()
        self.vars["psnr_target_db"] = tk.StringVar()
        self.vars["auto_save"] = tk.BooleanVar(value=True)
        self.vars["measurement_mode"] = tk.StringVar()
        self.vars["dictionary_type"] = tk.StringVar()
        self.vars["psnr_stop"] = tk.BooleanVar(value=False)

        self._simple_entry(form, 4, "Taille de bloc B", self.vars["block_size"])
        self._simple_entry(form, 5, "Ratio (0.25 ou 25)", self.vars["ratio"])
        self._combo_row(form, 6, "Matrice de mesure", self.vars["measurement_mode"], ["gaussian", "uniform", "bernoulli_1", "bernoulli_01"])
        self._combo_row(form, 7, "Dictionnaire", self.vars["dictionary_type"], ["dct", "mixte", "ksvd", "ksvd_dct", "ksvd_mixte"])
        self._simple_entry(form, 8, "Nombre d'atomes", self.vars["n_atoms"])
        self._simple_entry(form, 9, "Itérations K-SVD", self.vars["n_iter_ksvd"])
        self._simple_entry(form, 10, "Seed", self.vars["seed"])
        self._simple_entry(form, 11, "Limiter le nombre de patchs", self.vars["max_patches"])

        advanced = ttk.Frame(left, style="Card.TFrame", padding=16)
        advanced.pack(fill="x", pady=(10, 0))
        ttk.Label(advanced, text="Paramètres des méthodes", style="CardTitle.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))
        self._simple_entry(advanced, 1, "max_iter", self.vars["max_iter"])
        self._simple_entry(advanced, 2, "epsilon", self.vars["epsilon"])
        self._simple_entry(advanced, 3, "t pour StOMP", self.vars["t_stomp"])
        self._simple_entry(advanced, 4, "s pour CoSaMP", self.vars["s_cosamp"])
        self._simple_entry(advanced, 5, "p pour IRLS", self.vars["norm_p"])
        self._simple_entry(advanced, 6, "PSNR cible", self.vars["psnr_target_db"])
        ttk.Checkbutton(advanced, text="Activer l'arrêt par PSNR", variable=self.vars["psnr_stop"]).grid(row=7, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Checkbutton(advanced, text="Sauvegarder automatiquement", variable=self.vars["auto_save"]).grid(row=8, column=0, columnspan=2, sticky="w", pady=(8, 0))

        methods = ttk.Frame(right, style="Card.TFrame", padding=16)
        methods.pack(fill="x")
        ttk.Label(methods, text="Méthodes à appliquer", style="CardTitle.TLabel").pack(anchor="w")
        for method in ["mp", "omp", "stomp", "cosamp", "irls"]:
            var = tk.BooleanVar(value=method in {"omp", "cosamp"})
            self.method_vars[method] = var
            ttk.Checkbutton(methods, text=method.upper(), variable=var).pack(anchor="w", pady=3)

        actions = ttk.Frame(right, style="Card.TFrame", padding=16)
        actions.pack(fill="x", pady=(10, 0))
        ttk.Label(actions, text="Exécution", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Button(actions, text="Lancer la reconstruction", style="Primary.TButton", command=self.run_reconstruction).pack(fill="x", pady=(10, 6))
        ttk.Button(actions, text="Aller aux résultats", command=lambda: self.app.select_tab("Résultats")).pack(fill="x")

        info = ttk.Frame(right, style="Card.TFrame", padding=16)
        info.pack(fill="both", expand=True, pady=(10, 0))
        ttk.Label(info, text="Conseils", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            info,
            text=(
                "- Ratio : 0.25 ou 25 pour 25%\n"
                "- DCT : rapide pour un premier test\n"
                "- K-SVD : plus lourd mais plus intéressant pour le projet\n"
                "- OMP et CoSaMP sont les meilleurs points de départ\n"
                "- Limiter les patchs si vous voulez tester vite"
            ),
            style="App.TLabel",
            justify="left",
        ).pack(anchor="w", pady=(10, 0))

    def _entry_row(self, parent: ttk.Frame, row: int, label: str, key: str, *, browse_file: bool = False, browse_dir: bool = False) -> None:
        self.vars[key] = tk.StringVar()
        ttk.Label(parent, text=label, style="App.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
        entry = ttk.Entry(parent, textvariable=self.vars[key], width=52)
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        if browse_file:
            ttk.Button(parent, text="Parcourir", command=lambda k=key: self._browse_file(k)).grid(row=row, column=2, padx=(8, 0), pady=4)
        elif browse_dir:
            ttk.Button(parent, text="Dossier", command=lambda k=key: self._browse_dir(k)).grid(row=row, column=2, padx=(8, 0), pady=4)
        parent.columnconfigure(1, weight=1)

    def _simple_entry(self, parent: ttk.Frame, row: int, label: str, var: tk.Variable) -> None:
        ttk.Label(parent, text=label, style="App.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
        ttk.Entry(parent, textvariable=var, width=18).grid(row=row, column=1, sticky="ew", pady=4)
        parent.columnconfigure(1, weight=1)

    def _combo_row(self, parent: ttk.Frame, row: int, label: str, var: tk.StringVar, values: list[str]) -> None:
        ttk.Label(parent, text=label, style="App.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
        ttk.Combobox(parent, textvariable=var, values=values, state="readonly").grid(row=row, column=1, sticky="ew", pady=4)
        parent.columnconfigure(1, weight=1)

    def _browse_file(self, key: str) -> None:
        path = filedialog.askopenfilename(title="Choisir un fichier")
        if path:
            self.vars[key].set(path)

    def _browse_dir(self, key: str) -> None:
        path = filedialog.askdirectory(title="Choisir un dossier")
        if path:
            self.vars[key].set(path)

    def _set_defaults(self) -> None:
        self.vars["image_path"].set(self.state.image_path)
        self.vars["dictionary_train_image_path"].set(self.state.dictionary_train_image_path)
        self.vars["output_path"].set(self.state.output_path)
        self.vars["block_size"].set("8")
        self.vars["ratio"].set("25")
        self.vars["n_atoms"].set("")
        self.vars["n_iter_ksvd"].set("0")
        self.vars["seed"].set("0")
        self.vars["max_patches"].set("")
        self.vars["max_iter"].set("20")
        self.vars["epsilon"].set("1e-6")
        self.vars["t_stomp"].set("2.5")
        self.vars["s_cosamp"].set("6")
        self.vars["norm_p"].set("0.5")
        self.vars["psnr_target_db"].set("45")
        self.vars["measurement_mode"].set("gaussian")
        self.vars["dictionary_type"].set("dct")

    def run_reconstruction(self) -> None:
        try:
            methods = [name for name, var in self.method_vars.items() if var.get()]
            if not methods:
                raise ValueError("Sélectionne au moins une méthode.")

            image_path = self.vars["image_path"].get().strip()
            output_path = self.vars["output_path"].get().strip()
            dictionary_train = self.vars["dictionary_train_image_path"].get().strip() or None

            method_params = {
                "mp": {
                    "max_iter": parse_int(self.vars["max_iter"].get(), 20),
                    "epsilon": parse_float(self.vars["epsilon"].get(), 1e-6),
                },
                "omp": {
                    "max_iter": parse_int(self.vars["max_iter"].get(), 20),
                    "epsilon": parse_float(self.vars["epsilon"].get(), 1e-6),
                },
                "stomp": {
                    "max_iter": parse_int(self.vars["max_iter"].get(), 20),
                    "epsilon": parse_float(self.vars["epsilon"].get(), 1e-6),
                    "t": parse_float(self.vars["t_stomp"].get(), 2.5),
                },
                "cosamp": {
                    "max_iter": parse_int(self.vars["max_iter"].get(), 20),
                    "epsilon": parse_float(self.vars["epsilon"].get(), 1e-6),
                    "s": parse_int(self.vars["s_cosamp"].get(), 6),
                },
                "irls": {
                    "max_iter": parse_int(self.vars["max_iter"].get(), 20),
                    "epsilon": parse_float(self.vars["epsilon"].get(), 1e-6),
                    "norm_p": parse_float(self.vars["norm_p"].get(), 0.5),
                },
            }

            patch_params = {
                "max_patches": parse_int(self.vars["max_patches"].get(), None),
                "psnr_stop": bool(self.vars["psnr_stop"].get()),
                "psnr_target_db": parse_float(self.vars["psnr_target_db"].get(), 45.0),
            }
            patch_params = {k: v for k, v in patch_params.items() if v is not None}

            result = run_main(
                image_path=image_path,
                block_size=int(self.vars["block_size"].get()),
                ratio=float(self.vars["ratio"].get()),
                methodes=methods,
                dictionary_type=self.vars["dictionary_type"].get(),
                measurement_mode=self.vars["measurement_mode"].get(),
                output_path=output_path,
                n_atoms=parse_int(self.vars["n_atoms"].get(), None),
                n_iter_ksvd=parse_int(self.vars["n_iter_ksvd"].get(), 0),
                dictionary_train_image_path=dictionary_train,
                method_params=method_params,
                patch_params=patch_params,
                seed=parse_int(self.vars["seed"].get(), 0),
            )

            self.state.image_path = image_path
            self.state.output_path = output_path
            self.state.dictionary_train_image_path = dictionary_train or ""
            self.state.last_result = result
            self.state.add_log(f"Reconstruction lancée : {', '.join(methods)}")

            if bool(self.vars["auto_save"].get()):
                save_results(result, output_path)
                saved_dir = latest_subdir(output_path)
                if saved_dir:
                    self.state.add_log(f"Résultats sauvegardés dans {saved_dir}")

            self.app.refresh_all_pages()
            self.app.select_tab("Résultats")
            messagebox.showinfo("Succès", "Reconstruction terminée.")
        except Exception as exc:
            messagebox.showerror("Erreur", str(exc))
