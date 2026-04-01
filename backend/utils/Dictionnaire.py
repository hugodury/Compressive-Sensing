"""
Gestion des dictionnaires.
"""
import numpy as np

"""Fonction crétion d'un dictionnaire par DCT"""
def dictionaryDCT(N):
    k = np.arange(N).reshape(N, 1) # Colonne de 0 à N-1
    n = np.arange(N).reshape(1, N) # Ligne de 0 à N-1

    # On applique la formule
    DCT = np.cos(np.pi * k * (2 * n + 1) / (2 * N))
    DCT = DCT * np.sqrt(2 / N)
    DCT[0, :] = 1 / np.sqrt(N) #La première ligne étant différente des autres

    return DCT

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
    DCT = dictionaryDCT(N)
    
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
def dictionnaireKSVD(X, A, D):
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