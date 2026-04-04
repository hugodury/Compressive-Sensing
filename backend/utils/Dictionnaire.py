"""
Gestion des dictionnaires (DCT, tirages aléatoires, K-SVD, mixtes).
"""
import numpy as np
import os

# ==========================================
# 1. INITIALISATION DU DICTIONNAIRE
# ==========================================

def build_dct_dictionary(N: int) -> np.ndarray:
    k = np.arange(N).reshape(N, 1) # Colonne de 0 à N-1
    n = np.arange(N).reshape(1, N) # Ligne de 0 à N-1

    # On applique la formule
    DCT = np.cos(np.pi * k * (2 * n + 1) / (2 * N))
    DCT = DCT * np.sqrt(2 / N)
    DCT[0, :] = 1 / np.sqrt(N) # La première ligne étant différente des autres

    return DCT

def initRandDictionary(matrice_patch: np.ndarray, K: int) -> np.ndarray:
    """Initialisation par choix aléatoire de K colonnes dans les données."""
    N, Nb = matrice_patch.shape
    indices = np.random.choice(Nb, K, replace=False)
    D = matrice_patch[:, indices].copy()
    
    # Normalisation des atomes
    D = D / np.linalg.norm(D, axis=0, keepdims=True)
    return D

def init_dictionnaire_mixte_dct_patches(matrice_patch: np.ndarray, K: int) -> np.ndarray:
    """
    Initialise D avec K atomes : la moitié vient de la DCT (bases fréquentielles),
    l’autre moitié de colonnes tirées dans les patchs d’entraînement. 
    Combine structure fixe + données réelles.
    """
    N, Nb = matrice_patch.shape
    if K < 1:
        raise ValueError("K doit être >= 1.")
        
    # La DCT orthogonale n’a que N colonnes utiles
    k_dct = min(K // 2, N)
    k_patch = K - k_dct

    DCT = build_dct_dictionary(N)
    partie_dct = DCT[:, :k_dct].copy()

    if k_patch <= 0:
        return partie_dct

    replace = k_patch > Nb
    indices = np.random.choice(Nb, size=k_patch, replace=replace)
    partie_p = matrice_patch[:, indices].copy()
    partie_p = partie_p / np.linalg.norm(partie_p, axis=0, keepdims=True)

    return np.hstack((partie_dct, partie_p))

# ==========================================
# 2. OUTILS POUR LE COMPRESSIVE SENSING
# ==========================================

def estime_ordre_parcimonie_cosamp(
    matrice_patchs: np.ndarray, D: np.ndarray, *,
    max_iter_omp: int = 32, epsilon: float = 1e-3,
    max_echantillons: int = 64, seed: int | None = None
) -> int:
    """Propose un ordre s pour CoSaMP à partir d’un codage OMP sur les patchs."""
    from backend.utils.Methode import omp

    N, L = matrice_patchs.shape
    K = D.shape[1]
    rng = np.random.default_rng(seed)
    nb = min(max_echantillons, L)
    
    if nb < 1:
        return max(1, min(6, K))
        
    indices = np.arange(L, dtype=int) if L <= nb else rng.choice(L, size=nb, replace=False)

    nnz_par_patch: list[int] = []
    for j in indices:
        xj = matrice_patchs[:, j]
        alpha = omp(D, xj, max_iter=max_iter_omp, epsilon=epsilon)
        nnz_par_patch.append(int(np.sum(np.abs(alpha) > epsilon)))

    s = int(np.ceil(float(np.median(nnz_par_patch))))
    return max(1, min(s, K))

# ==========================================
# 3. ALGORITHME K-SVD
# ==========================================

def learn_ksvd_full(
    X: np.ndarray, K: int, n_iter: int, *,
    init: str, omp_max_iter: int = 40, omp_epsilon: float = 1e-3,
    seed: int | None = None, max_train_cols: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Boucle K-SVD complète : codage parcimonieux OMP puis mise à jour atome par atome.
    init : 'dct', 'mixte' ou 'random'.
    """
    from backend.utils.Methode import omp

    init_n = init.lower().strip()
    if init_n not in ("dct", "mixte", "random"):
        raise ValueError("init doit être 'dct', 'mixte' ou 'random'.")

    X = np.asarray(X, dtype=np.float64)
    N, L = X.shape
    
    if seed is not None:
        np.random.seed(int(seed))

    Lw = L if max_train_cols is None else min(int(max_train_cols), L)
    X_work = X[:, :Lw]

    # Initialisation
    if init_n == "dct":
        D = build_dct_dictionary(N)[:, :K].astype(np.float64, copy=True)
    elif init_n == "mixte":
        D = init_dictionnaire_mixte_dct_patches(X_work, K)
    else:
        D = initRandDictionary(X_work, K)

    A = np.zeros((K, Lw), dtype=np.float64)
    
    # Boucle K-SVD
    for _ in range(n_iter):
        # 1. Sparse Coding (OMP)
        for j in range(Lw):
            A[:, j] = omp(D, X_work[:, j], max_iter=omp_max_iter, epsilon=omp_epsilon)
            
        # 2. Dictionary Update
        D = learn_ksvd_dictionary(X_work, A, D)

    return D, A

def learn_ksvd_dictionary(X: np.ndarray, A: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Étape de mise à jour K-SVD (SVD sur l'erreur résiduelle)."""
    K = D.shape[1] 
    epsilon = 1e-10
    
    for i in range(K): 
        d = D[:, i]
        alpha = A[i, :]
        
        w = np.where(np.abs(alpha) > epsilon)[0]
        if len(w) == 0:
            continue
            
        E = X - np.dot(D, A)
        Ei = E + np.outer(d, alpha)
        EiR = Ei[:, w]

        U, S, Vt = np.linalg.svd(EiR, full_matrices=False)

        D[:, i] = U[:, 0]
        A[i, w] = S[0] * Vt[0, :]
        
    return D

# ==========================================
# 4. SAUVEGARDE ET CHARGEMENT
# ==========================================

def save_dictionary(D: np.ndarray, name: str):
    # Sécurisation du dossier de destination
    os.makedirs("Data/Dictionnaire", exist_ok=True)
    path = os.path.join("Data/Dictionnaire", name)
    
    if path.endswith('.csv') or path.endswith('.txt'):
        np.savetxt(path, D, delimiter=',')
        print(f"Dictionnaire sauvegardé en format texte sous : {path}")
    else:
        if not path.endswith('.npy'):
            path += '.npy'
        np.save(path, D)
        print(f"Dictionnaire sauvegardé en format binaire sous : {path}")

def load_dictionary(name: str) -> np.ndarray:
    path = os.path.join("Data/Dictionnaire", name)
    
    if path.endswith('.csv') or path.endswith('.txt'):
        D = np.loadtxt(path, delimiter=',')
        print(f"Dictionnaire chargé depuis : {path}")
    else:
        if not path.endswith('.npy'):
            path += '.npy'
        D = np.load(path)
        print(f"Dictionnaire chargé depuis : {path}")
    return D

# ==========================================
# 5. MENU INTERACTIF UTILISATEUR
# ==========================================

def mainDico(X: np.ndarray) -> np.ndarray:
    """Interface console pour générer ou charger un dictionnaire facilement."""
    print("\n" + "="*50)
    print("--- GESTION DU DICTIONNAIRE ---")
    print("="*50)
    
    choix_action = ""
    while choix_action not in ["1", "2"]:
        choix_action = input("1 - Charger un dictionnaire (.npy/.csv)\n2 - Créer un nouveau dictionnaire\nVotre choix : ")

    if choix_action == "1":
        fichier = input("\nNom du fichier à charger (dans Data/Dictionnaire/) : ")
        try:
            return load_dictionary(fichier)
        except FileNotFoundError:
            print(f"-> Erreur : Le fichier est introuvable. Reprise...\n")
            return mainDico(X)

    print("\n" + "-"*50)
    N = X.shape[0]
    
    try:
        K = int(input(f"Nombre d'atomes souhaité (K >= {N} conseillé) : "))
    except ValueError:
        K = N
        print(f"Entrée invalide. K fixé à {N} par défaut.")

    choix_init_dict = {"1": "dct", "2": "random", "3": "mixte"}
    choix_init = ""
    while choix_init not in choix_init_dict:
        choix_init = input("Méthode (1: DCT pure, 2: Aléatoire, 3: Mixte) : ")
    init_type = choix_init_dict[choix_init]

    choix_ksvd = input("\nOptimiser avec K-SVD ? (o/n) : ").lower()
    
    if choix_ksvd in ['o', 'oui']:
        try:
            maxIter = int(input("Nombre d'itérations K-SVD (ex: 10) : "))
        except ValueError:
            maxIter = 10
            print("Valeur invalide. maxIter fixé à 10.")
            
        print("\nEntraînement en cours...")
        D_final, _ = learn_ksvd_full(X, K, n_iter=maxIter, init=init_type)
    else:
        print("\nGénération sans K-SVD...")
        if init_type == "dct":
            D_final = build_dct_dictionary(N)[:, :K]
        elif init_type == "mixte":
            D_final = init_dictionnaire_mixte_dct_patches(X, K)
        else:
            D_final = initRandDictionary(X, K)

    choix_save = input("\nSauvegarder ce dictionnaire ? (o/n) : ").lower()
    if choix_save in ['o', 'oui']:
        fichier_save = input("Nom du fichier de sauvegarde (ex: dico.npy ou dico.csv) : ")
        save_dictionary(D_final, fichier_save)

    return D_final