import numpy as np
import pandas as pd  # Pour une lecture facile du CSV


# --- 1. FONCTIONS EXISTANTES (OMP, PSNR, K-SVD) ---

def orthogonal_matching_pursuit(x, D, n_iter=None, tol=1e-6):
    N, K = D.shape
    residual = x.copy()
    support = []
    alpha = np.zeros(K)
    max_k = n_iter if n_iter is not None else N
    
    for k in range(max_k):
        contributions = np.abs(D.T @ residual)
        m_k = np.argmax(contributions)
        
        if m_k not in support:
            support.append(m_k)
        else:
            break
            
        D_active = D[:, support]
        alpha_active = np.linalg.lstsq(D_active, x, rcond=None)[0]
        residual = x - D_active @ alpha_active
        
        if np.linalg.norm(residual) < tol:
            break
            
    alpha[support] = alpha_active
    return alpha, residual

def calculate_psnr(X, X_hat):
    mse = np.mean((X - X_hat)**2)
    if mse == 0: return 100
    max_val = np.max(X)
    return 20 * np.log10(max_val / np.sqrt(mse))

def k_svd(X, K, s, n_iterations, tol_omp=1e-6):
    N, L = X.shape
    # Initialisation : premières colonnes de X normalisées
    D = X[:, :K].copy() 
    D /= np.linalg.norm(D, axis=0) 
    
    Lambda = np.zeros((K, L))
    historique_psnr = []
    
    for it in range(n_iterations):
        # ÉTAPE 1 : Codage parcimonieux
        for i in range(L):
            Lambda[:, i], _ = orthogonal_matching_pursuit(X[:, i], D, n_iter=s, tol=tol_omp)
            
        # ÉTAPE 2 : Mise à jour du dictionnaire
        for k in range(K):
            wk = np.where(Lambda[k, :] != 0)[0]
            if len(wk) == 0:
                D[:, k] = X[:, np.random.randint(L)] 
                D[:, k] /= np.linalg.norm(D[:, k])
                continue 
            
            # Calcul de l'erreur sans l'atome k
            Ek = X[:, wk] - np.delete(D, k, axis=1) @ np.delete(Lambda, k, axis=0)[:, wk]
            
            # SVD pour mettre à jour l'atome et les coefficients
            U, S, Vh = np.linalg.svd(Ek, full_matrices=False)
            D[:, k] = U[:, 0]
            Lambda[k, wk] = S[0] * Vh[0, :]

        # Évaluation
        X_hat = D @ Lambda
        current_psnr = calculate_psnr(X, X_hat)
        historique_psnr.append(current_psnr)
        print(f"Itération {it+1}/{n_iterations} | PSNR: {current_psnr:.2f} dB")
            
    return D, Lambda, historique_psnr

# --- 2. RÉSOLUTION DE L'EXERCICE 5 ---

# Paramètres de l'énoncé
N_dim = 98
K_atomes = 100
S_parcimonie = 10  # Noté L=10 dans ton énoncé
EPSILON = 1e-6
NB_ITER = 15 # On définit un nombre d'itérations pour l'apprentissage

# --- REMPLACE LE BLOC DE CHARGEMENT PAR CELUI-CI ---

print("--- Chargement des données 'Data.xlsx' ---")
try:
    # On utilise read_excel. Par défaut, header=0 utilise la 1ère ligne pour les noms.
    # Les données réelles commencent donc à la 2ème ligne du fichier.
    df = pd.read_excel('Data.xlsx') 
    
    # On convertit en matrice NumPy
    X = df.values  
    
    # Vérification des dimensions
    # Si le fichier a 99 lignes (1 header + 98 data) et 150 colonnes :
    # X.shape doit être (98, 150)
    N, L = X.shape
    print(f"Dimensions extraites : N={N} (dimensions), L={L} (signaux)")

    # --- Lancement du K-SVD avec tes paramètres ---
    K_atomes = 100
    S_parcimonie = 10 # Ton "L=10" dans l'énoncé
    NB_ITER = 15
    EPSILON = 1e-6

    D_final, L_final, psnr_vals = k_svd(X, K=K_atomes, s=S_parcimonie, n_iterations=NB_ITER, tol_omp=EPSILON)

except FileNotFoundError:
    print("Erreur : Le fichier 'Data.xlsx' est introuvable.")
except Exception as e:
    print(f"Une erreur est survenue : {e}")
    print("Conseil : Installez openpyxl avec 'pip install openpyxl' si nécessaire.")