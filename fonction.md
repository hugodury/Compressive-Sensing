# 📦 Projet Compressive Sensing - Reconstruction d’Images

## 🎓 Contexte

Ce projet consiste à implémenter un pipeline complet de **Compressive Sensing (CS)** pour la reconstruction d’images à partir de mesures compressées.

Le principe repose sur :

* Représentation parcimonieuse :
  x = Dα
* Acquisition compressée :
  y = Φx
* Reconstruction :
  ΦDα ≈ y puis x̂ = Dα

Le projet inclut :

* Block Compressed Sensing (BCS)
* Apprentissage de dictionnaire (K-SVD)
* Comparaison d’algorithmes de reconstruction
* Analyse des performances 

---

# 🧱 Arborescence

```
Readme.md
main.py

frontend/
│ app.py
└── Pages/

backend/
│ Traitement_Image.py
│ main_backend.py
└── utils/
    │ Methode.py
    │ Mesure.py
    │ Dictionnaire.py
    │ Metrics.py

Data/
├── Dictionnaire/
├── Images/
└── Result/
    └── jj.mm.hh.mm/
        │ Image.png
        │ metrics.csv
        └── Graph/
            │ *.png
```

---

# 🚀 main.py

```python
def setupParam(
    image_path: str,
    block_size: int,
    ratio: float,
    method: str,
    dictionary_type: str
) -> dict:
    """
    Initialise tous les paramètres du pipeline :
    image, taille des blocs BxB, ratio M/N, méthode de reconstruction
    et type de dictionnaire (DCT ou K-SVD).
    """

def main() -> None:
    """
    Lance le pipeline complet de compressive sensing :
    - découpage image en blocs
    - acquisition compressée
    - reconstruction
    - calcul des métriques
    """
```

---

# 🧠 backend/main_backend.py

```python
def run_bcs_pipeline(
    image: np.ndarray,
    B: int,
    Phi: np.ndarray,
    D: np.ndarray,
    method: str
) -> dict:
    """
    Implémente le Block Compressive Sensing :
    - découpe image en blocs Xi
    - vectorisation xi
    - mesures yi = Φxi
    - reconstruction xi ≈ Dα
    """

def reconstruct_patch(
    y: np.ndarray,
    Phi: np.ndarray,
    D: np.ndarray,
    method: str
) -> np.ndarray:
    """
    Résout le problème ΦDα ≈ y avec une méthode parcimonieuse
    puis reconstruit x̂ = Dα.
    """

def compare_algorithms(
    image: np.ndarray,
    methods: list,
    ratios: list
) -> dict:
    """
    Compare MP, OMP, StOMP, CoSaMP, IRLS
    pour différents ratios de mesure.
    """

def save_results(
    image: np.ndarray,
    reconstructed: np.ndarray,
    metrics: dict,
    output_path: str
) -> None:
    """
    Sauvegarde :
    - image reconstruite
    - métriques (CSV)
    - graphes
    """
```

---

# 🖼 backend/Traitement_Image.py

```python
def imgToMatrix(image: np.ndarray) -> np.ndarray:
    """
    Convertit une image 2D en matrice exploitable.
    """

def extractPatch(
    matrix_img: np.ndarray,
    B: int
) -> list[np.ndarray]:
    """
    Découpe l’image en blocs Xi de taille B×B (BCS).
    """

def vectoriser(
    patch: np.ndarray
) -> np.ndarray:
    """
    Transforme un patch Xi en vecteur colonne xi.
    """

def reconstructBlocks(
    patches: list[np.ndarray],
    B: int
) -> np.ndarray:
    """
    Reforme les blocs après reconstruction.
    """

def reconstructImage(
    blocks: np.ndarray,
    image_shape: tuple
) -> np.ndarray:
    """
    Recompose l’image complète à partir des blocs.
    """
```

---

# ⚙️ backend/utils/Methode.py

```python
def mp(
    D: np.ndarray,
    x: np.ndarray,
    max_iter: int
) -> np.ndarray:
    """
    Matching Pursuit :
    sélection itérative d’un atome maximisant la corrélation avec le résiduel.
    """

def omp(
    D: np.ndarray,
    x: np.ndarray,
    max_iter: int
) -> np.ndarray:
    """
    Orthogonal Matching Pursuit :
    amélioration de MP avec projection orthogonale (moindres carrés).
    """

def stomp(
    D: np.ndarray,
    x: np.ndarray,
    threshold: float
) -> np.ndarray:
    """
    StOMP :
    sélection multiple d’atomes via seuillage.
    """

def cosamp(
    D: np.ndarray,
    x: np.ndarray,
    sparsity: int
) -> np.ndarray:
    """
    CoSaMP :
    sélection + rejet d’atomes pour améliorer la stabilité.
    """

def irls(
    D: np.ndarray,
    x: np.ndarray,
    lambda_reg: float,
    max_iter: int
) -> np.ndarray:
    """
    IRLS :
    résolution du problème ℓp via moindres carrés pondérés itératifs.
    """
```

---

# 📐 backend/utils/Mesure.py

```python
def generate_measurement_matrix(
    M: int,
    N: int,
    mode: str
) -> np.ndarray:
    """
    Génère Φ (gaussienne, bernoulli, etc.)
    """

def apply_measurement(
    Phi: np.ndarray,
    x: np.ndarray
) -> np.ndarray:
    """
    Calcule y = Φx (acquisition compressée).
    """

def compute_ratio(
    M: int,
    N: int
) -> float:
    """
    Calcule le ratio r = M / N.
    """

def compute_coherence(
    Phi: np.ndarray,
    D: np.ndarray
) -> float:
    """
    Calcule la cohérence mutuelle entre Φ et D.
    """
```

---

# 📚 backend/utils/Dictionnaire.py

```python
def build_dct_dictionary(
    N: int
) -> np.ndarray:
    """
    Génère un dictionnaire DCT.
    """

def initRandDictionary(
    matrice_patch : np.ndarray,
    K : int
) -> np.ndarray:
    """
    Fonction d'initialisation dictionnaire par choix alea de K vecteurs
    On appliquera ensuite le K-SVD
    """

def initRandDictionaryDCT(
    matrice_patch : np.ndarray, 
    K : int
) -> np.ndarray:
    """
    Fonction d'initialisation dictionnaire par choix alea de K vecteurs + DCT
    On appliquera ensuite le K-SVD
    """


def learn_ksvd_dictionary(
    X: np.ndarray,
    A: np.ndarray,
    D: np.array,
) -> np.ndarray:
    """
    Apprend un dictionnaire via K-SVD.
    """

def save_dictionary(
    D: np.ndarray,
    path: str
) -> None:
    """
    Sauvegarde le dictionnaire.
    """

def load_dictionary(
    path: str
) -> np.ndarray:
    """
    Charge un dictionnaire existant.
    """
```

---

# 📊 backend/utils/Metrics.py

```python
def compute_mse(
    original: np.ndarray,
    reconstructed: np.ndarray
) -> float:
    """
    Erreur quadratique moyenne.
    """

def compute_psnr(
    original: np.ndarray,
    reconstructed: np.ndarray
) -> float:
    """
    PSNR (qualité reconstruction).
    """

def compute_relative_error(
    original: np.ndarray,
    reconstructed: np.ndarray
) -> float:
    """
    Erreur relative demandée dans le projet.
    """

def compute_execution_time(
    start: float,
    end: float
) -> float:
    """
    Temps d’exécution.
    """
```

---

# 🎨 frontend/app.py

```python
def launch_app() -> None:
    """
    Lance l’interface graphique du projet.
    """

def display_image(
    image: np.ndarray
) -> None:
    """
    Affiche une image.
    """

def display_reconstruction(
    original: np.ndarray,
    reconstructed: np.ndarray
) -> None:
    """
    Compare visuellement original vs reconstruction.
    """

def choose_parameters() -> dict:
    """
    Interface pour choisir :
    - ratio
    - méthode
    - dictionnaire
    """
```

---

# 🎯 Objectif final

* Implémenter le pipeline CS complet
* Comparer MP / OMP / StOMP / CoSaMP / IRLS
* Tester plusieurs ratios (15% → 75%) 
* Comparer dictionnaire DCT vs K-SVD
* Évaluer via erreurs et métriques
* Construire une IHM pour visualisation

---

# 🔥 Résultat attendu

* Reconstruction d’image à partir de mesures compressées
* Graphiques de performance
* Analyse critique des résultats


# Frontend
* Charger une image,
* Choisir la taille de bloc B,
* Choisir le pourcentage de mesures,
* Choisir la matrice de mesure Φ1..Φ4,
* Choisir la méthode de reconstruction MP / OMP / StOMP / CoSaMP / IRLS,
* Choisir le dictionnaire DCT ou K-SVD,
* Lancer la reconstruction,
* Afficher l’image originale et l’image reconstruite,
* Afficher les métriques et les temps,
* Afficher des graphes et comparaisons,
* Visualiser les étapes du procédé de compressive sensing.

