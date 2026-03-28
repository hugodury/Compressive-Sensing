import numpy as np
import time


# ---------------------------------------------------------
# 1. Dictionnaire DCT (Base Orthonormale - CM4/TD1)
# ---------------------------------------------------------
def generate_DF_DCT(N):
    DF = np.zeros((N, N))
    for k in range(N):
        for l in range(N):
            C = np.sqrt(1/N) if k == 0 else np.sqrt(2/N)
            DF[k, l] = C * np.cos((k * (2*l + 1) * np.pi) / (2*N))
    return DF

# ---------------------------------------------------------
# 2. Matrices de mesure (Définitions CM4 p.28)
# ---------------------------------------------------------
def get_phi(idx, M, N):
    if idx == 1: return np.random.uniform(0, 1, (M, N))
    if idx == 2: return np.where(np.random.rand(M, N) < 0.5, -1, 1) / np.sqrt(M)
    if idx == 3: return np.where(np.random.rand(M, N) < 0.5, 0, 1)
    if idx == 4: return np.random.normal(0, np.sqrt(1/M), (M, N))
    if idx == 5:
        mat = np.zeros((M, N))
        mask = np.random.rand(M, N) < 0.1
        mat[mask] = np.random.randn(np.sum(mask))
        return mat

# ---------------------------------------------------------
# 3. Calcul de la Cohérence Mutuelle mu(Phi, D) [CM4 p.26]
# ---------------------------------------------------------
def calculate_coherence(Phi, D, N):
    Phi_norm = Phi / np.linalg.norm(Phi, axis=1, keepdims=True)
    D_norm = D / np.linalg.norm(D, axis=0, keepdims=True)
    G = np.abs(Phi_norm @ D_norm)
    return np.sqrt(N) * np.max(G)

# ---------------------------------------------------------
# 4. Votre Algorithme OMP (Reconstruction - CM2)
# ---------------------------------------------------------
def OMP(y, A, kmax, epsilon):
    # y : vecteur de mesures (M x 1)
    # A : dictionnaire projeté Phi * D (M x N)
    N_atoms = A.shape[1]
    alpha = np.zeros(N_atoms)
    residuel = y.copy()
    P = []
    k = 0
    alpha_k = np.zeros(1)
    
    while k < kmax and np.linalg.norm(residuel) > epsilon:
        # Sélection de l'atome le plus corrélé
        mk = np.argmax(np.abs(A.T @ residuel) / np.linalg.norm(A, axis=0))
        if mk not in P:
            P.append(mk)
        else:
            break # Evite les répétitions infinies si stagnation
            
        Ak = A[:, P]
        # Résolution Moindres Carrés (votre version lstsq)
        alpha_k, _, _, _ = np.linalg.lstsq(Ak, y, rcond=None)
        residuel = y - Ak @ alpha_k
        k += 1
        
    alpha[P] = alpha_k
    return alpha, P, residuel, np.linalg.norm(residuel), k

# ---------------------------------------------------------
# 5. INITIALISATION ET ANALYSE
# ---------------------------------------------------------
N = 500
Ms =  {25, 30, 45, 50, 100, 150, 200, 250}
# Signal d'origine x (On prend un vecteur de 1)
# Note : En DCT, x=[1...1] est 1-épars (seul le premier coefficient alpha est non nul)
x = np.ones((N, 1)) 
D = generate_DF_DCT(N)

all_mu_results = {M: {} for M in Ms}
best_configs = {} # Pour stocker le meilleur Phi/M pour la reconstruction

print(f"--- ANALYSE DE LA COHÉRENCE MUTUELLE (N={N}) ---")

for i in range(1, 6):
    best_mu = float('inf')
    best_M = None
    best_Phi = None
    
    for M in Ms:
        Phi_tmp = get_phi(i, M, N)
        mu_val = calculate_coherence(Phi_tmp, D, N)
        all_mu_results[M][f"Phi{i}"] = mu_val
        
        if mu_val < best_mu:
            best_mu = mu_val
            best_M = M
            best_Phi = Phi_tmp
            
    best_configs[f"y{i}"] = {"M": best_M, "mu": best_mu, "Phi": best_Phi}

# Affichage du tableau de cohérence trié
header = "M | " + " | ".join([f"Phi{i:1}  " for i in range(1, 6)])
print("\n" + header)
print("-" * len(header))
for M in Ms:
    line = f"{M:3} | " + " | ".join([f"{all_mu_results[M][f'Phi{i}']:7.4f}" for i in range(1, 6)])
    print(line)

# ---------------------------------------------------------
# 6. RECONSTRUCTION OMP POUR CHAQUE Y_i OPTIMAL
# ---------------------------------------------------------
print("\n" + "="*80)
print("PHASE DE RECONSTRUCTION OMP (Retrouver alpha)")
print("="*80)

for i in range(1, 6):
    config = best_configs[f"y{i}"]
    Phi = config["Phi"]
    M_val = config["M"]
    
    # Étape d'Acquisition : y = Phi * x
    y_mesure = Phi @ x
    
    # Étape de Reconstruction :
    # Le dictionnaire projeté pour l'OMP est A = Phi * D
    A_equiv = Phi @ D
    
    # Appel OMP (on fixe kmax à 10 pour chercher une solution éparse)
    alpha_rec, support, res_vec, norm_err, it = OMP(y_mesure, A_equiv, kmax=10, epsilon=1e-6)
    
    print(f"\n>>> RÉSULTATS POUR y{i} (Phi{i} | M={M_val} | mu={config['mu']:.4f}) :")
    print(f"- Atomes sélectionnés : {support}")
    print(f"- Norme de l'erreur de reconstruction : {norm_err:.4e}")
    print("- Vecteur y complet :")
    print(y_mesure.flatten())
    print("- Coefficients alpha reconstruits (non nuls) :")
    print(alpha_rec[support])