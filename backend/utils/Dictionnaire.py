"""
Gestion des dictionnaires (DCT, tirages aléatoires, K-SVD, mélanges).
"""
import numpy as np


def build_dct_dictionary(N: int):
    k = np.arange(N).reshape(N, 1) # Colonne de 0 à N-1
    n = np.arange(N).reshape(1, N) # Ligne de 0 à N-1

    # On applique la formule
    DCT = np.cos(np.pi * k * (2 * n + 1) / (2 * N))
    DCT = DCT * np.sqrt(2 / N)
    DCT[0, :] = 1 / np.sqrt(N) #La première ligne étant différente des autres

    return DCT


def init_dictionnaire_mixte_dct_patches(matrice_patch: np.ndarray, K: int) -> np.ndarray:
    """
    Initialise D avec K atomes : environ la moitié vient de la DCT (bases fréquentielles),
    l’autre moitié de colonnes tirées dans les patchs d’entraînement (comme l’init classique
    avant K-SVD). Ça combine structure fixe + données réelles, souvent plus stable que
    du pur aléatoire quand on enchaîne avec K-SVD.
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


def estime_ordre_parcimonie_cosamp(
    matrice_patchs: np.ndarray,
    D: np.ndarray,
    *,
    max_iter_omp: int = 32,
    epsilon: float = 1e-3,
    max_echantillons: int = 64,
    seed: int | None = None,
) -> int:
    """
    Propose un ordre s pour CoSaMP à partir d’un codage OMP sur les patchs (même idée
    que l’étape « sparse coding » du K-SVD : on regarde combien de coefficients sont
    activement utilisés par patch, puis on prend une valeur centrale (médiane arrondie).
    """
    from backend.utils.Methode import omp

    N, L = matrice_patchs.shape
    K = D.shape[1]
    rng = np.random.default_rng(seed)
    nb = min(max_echantillons, L)
    if nb < 1:
        return max(1, min(6, K))
    if L <= nb:
        indices = np.arange(L, dtype=int)
    else:
        indices = rng.choice(L, size=nb, replace=False)

    nnz_par_patch: list[int] = []
    for j in indices:
        xj = matrice_patchs[:, j]
        alpha = omp(D, xj, max_iter=max_iter_omp, epsilon=epsilon)
        nnz_par_patch.append(int(np.sum(np.abs(alpha) > epsilon)))

    s = int(np.ceil(float(np.median(nnz_par_patch))))
    s = max(1, min(s, K))
    return s


"""Fonction d'initialisation dictionnaire par choix alea de K vecteurs
    On appliquera ensuite le K-SVD """
def initRandDictionary(matrice_patch, K):
    (B2, Nb) = matrice_patch.shape
    indices = np.random.choice(Nb, K, replace=False) # On selectione K indices de colonnes aleatoirement
    D = matrice_patch[:, indices].copy()
    
    # Normalisation des atomes
    D = D / np.linalg.norm(D, axis=0)
    
    return D

"""Fonction d'initialisation dictionnaire par choix alea de K vecteurs + DCT
    On appliquera ensuite le K-SVD """
def initRandDictionaryDCT(matrice_patch, K):
    (N, Nb) = matrice_patch.shape
    
    #On génère la base DCT
    DCT = build_dct_dictionary(N)
    
    #Si le dictionnaire demandé est plus petit ou égal à la base DCT, on tronque la DCT
    if K <= N:
        return DCT[:, :K].copy()
        
    #Sinon, on complète le dictionnaire avec des signaux d'entraînement aléatoires
    K_rand = K - N
    indices = np.random.choice(Nb, K_rand, replace=False)
    D_rand = matrice_patch[:, indices].copy()
    
    # Normalisation des atomes aléatoires
    D_rand = D_rand / np.linalg.norm(D_rand, axis=0)
    
    # 4. On assemble la partie DCT et la partie aléatoire
    D = np.hstack((DCT, D_rand))
    
    return D

"""Ajustement du dictionnaire (K-SVD)
    Met à jour les atomes de D et les coefficients de A de manière itérative"""
def learn_ksvd_dictionary(X, A, D):
    # On récupère K dynamiquement pour que la fonction soit générique
    K = D.shape[1] 
    
    # On travaille sur chaque atome
    for i in range(K): 
        d = D[:, i]
        alpha = A[i, :]
        
        #On définit le support (ie l'indice des elems non nuls) (comparer à 0 est trop restrictive)
        epsilon = 1e-10
        
        #Le support s'évalue sur les coefficients (alpha), pas sur l'atome
        w = np.where(np.abs(alpha) > epsilon)[0]
        
        #Si l'atome n'est pas du tout utilisé, on passe à l'itération suivante
        if len(w) == 0:
            continue
            
        # On calcule l'erreur résiduel global (ie X - DA sans l'atome d)
        E = X - np.dot(D, A)
        Ei = E + np.outer(d, alpha)

        # On calcule l'erreur de reconstruction seulement pour les signaux qui utilisent l'atome
        EiR = Ei[:, w]

        #SVD de EiR
        U, S, Vt = np.linalg.svd(EiR, full_matrices=False)

        # On met à jour di et les coeff non nuls de alpha
        D[:, i] = U[:, 0]
        A[i, w] = S[0] * Vt[0, :]
        
    return D


"""Sauvegarde le dictionnaire."""
#Exemple de noms :  dct_50.csv
def save_dictionary(D: np.ndarray,name: str) :
    path = "Data/Dictionnaire/"+name
    # Si on souhaite save dans un csv ou txt
    if path.endswith('.csv') or path.endswith('.txt'):
        np.savetxt(path, D, delimiter=',')
        print(f"Dictionnaire sauvegardé en format texte sous : {path}")
        
    # Sinon, on utilise le format binaire natif de NumPy (.npy) par défaut
    else:
        np.save(path, D)
        if not path.endswith('.npy'):
            path += '.npy'
    
        print(f"Dictionnaire sauvegardé sous : {path}")
        
    return 0

"""Charge un dictionnaire existant."""
def load_dictionary(name: str) -> np.ndarray:
    path = "Data/Dictionnaire/"+name
    # Si le fichier est au format csv ou txt
    if path.endswith('.csv') or path.endswith('.txt'):
        D = np.loadtxt(path, delimiter=',')
        print(f"Dictionnaire texte chargé depuis : {path}")
        
    else:
        if not path.endswith('.npy'):
            path += '.npy'
            
        D = np.load(path)
        print(f"Dictionnaire binaire chargé depuis : {path}")
        
    return D

