import numpy as np
import time

# --- TON ALGORITHME OMP (STRICTEMENT IDENTIQUE) ---
def OMP(x, D, kmax, epsilon):
    N, K = D.shape
    alpha = np.zeros(K)
    residuel = x.copy()
    Dk = []
    P = []
    k = 0
    while k < kmax and np.linalg.norm(residuel) > epsilon:
        mk = np.argmax(np.abs(D.T @ residuel) / np.linalg.norm(D))
        P.append(mk)
        Dk = D[:, P]
        
        alpha_k = np.linalg.inv((Dk.T @ Dk)) @ Dk.T @ x
        residuel = x - Dk @ alpha_k
        k = k + 1
     
    alpha[P] = alpha_k
    norme = np.linalg.norm(residuel)
    return alpha, P, residuel, norme, k


# --- ALGORITHME K-SVD BASÉ SUR TES ÉTAPES ---
def k_svd(X, K, n_iterations, kmax_omp, epsilon_omp):

    N, L = X.shape
    
# --- INITIALISATION : K premières colonnes de X ---
    D = X[:, :K].copy()
    for i in range(K):
        D[:, i] /= np.linalg.norm(D[:, i])
    
    Alpha = np.zeros((K, L))
    
    for it in range(n_iterations):
        # 1. Sparse Coding
        for j in range(L):
            Alpha[:, j] = OMP(X[:, j], D, kmax_omp, epsilon_omp)[0]
            
        # 2. Mise à jour du Dictionnaire (Boucle sur K)
        for i in range(K):
            di = D[:, i].reshape(-1, 1)
            alphai = Alpha[i, :].reshape(1, -1)
            
            # Support wi
            wi = np.where(alphai != 0)[1]
            
            if len(wi) > 0:
                E = X - np.dot(D, Alpha)
                Ei = E + np.dot(di, alphai)
                ER = Ei[:, wi]
                
                # SVD sur ER
                U, Delta, V = np.linalg.svd(ER)
                
                D[:, i] = U[:, 0]
                Alpha[i, wi] = Delta[0] * V[0, :]
                
            else:
                # --- CONDITION ELSE : Atome non utilisé ---
                j = np.random.randint(L)
                D[:, i] = X[:, j]
                D[:, i] /= np.linalg.norm(D[:, i])
                
    return D, Alpha


# --- DONNÉES INITIALES ---
epsilon = 0.000001
kmax = 1000  # On garde ton kmax de 1000

# On définit un X de 24 signaux (pour coller à K=24)
# Chaque colonne est une légère variante de ton vecteur x
x_base = np.array([-10, -10, 10, 20, 15, 10])
X = np.array([x_base + i for i in range(24)]).T.astype(float)

# --- EXÉCUTION ---
debut = time.time()
# On utilise kmax et epsilon fixés plus haut
D_appris, Alpha_appris = k_svd(X, 24, 10, kmax, epsilon)
fin = time.time()

# Test final sur le premier vecteur
test = OMP(X[:, 0], D_appris, kmax, epsilon)

# --- AFFICHAGE (FORMAT EXACT) ---
print("-----------------------------------------------------------------------------\n")
print("Taille de D: \n", D_appris.shape)
print("Alpha: \n", test[0])
print("Atome choisis \n", test[1])
print("Le résiduel est \n", test[2])
print("La norme du résiduel est \n", test[3])
print("Le nombre d iterations est : \n", test[4])

temps_exec = fin - debut
print(f"Temps d'exécution : {temps_exec:.6f} secondes \n")

norme_L0 = np.sum(np.abs(test[0]) > 1e-10)
print(f"Norme L0 : {norme_L0}")
print(f"Nombre de zéros dans Alpha : {len(test[0]) - norme_L0} / {len(test[0])}")