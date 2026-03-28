import numpy as np

# 1. Dictionnaire DCT (CM 4 / TD 1)
def generate_DCT_matrix(N):
    D = np.zeros((N, N))
    for k in range(N):
        for l in range(N):
            C = np.sqrt(1/N) if k == 0 else np.sqrt(2/N)
            D[k, l] = C * np.cos((k * (2*l + 1) * np.pi) / (2*N))
    return D

# 2. Matrices de Mesure (CM 4 p.28)
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

# 3. Calcul Cohérence (CM 4 p.26)
def mutual_coherence(Phi, D, N):
    Phi_n = Phi / np.linalg.norm(Phi, axis=1, keepdims=True)
    D_n = D / np.linalg.norm(D, axis=0, keepdims=True)
    return np.sqrt(N) * np.max(np.abs(Phi_n @ D_n))

# --- Initialisation ---
N = 500
Ms =  {25, 30, 45, 50, 100, 150, 200, 250}
x = np.ones((N, 1))
D_dct = generate_DCT_matrix(N)
all_results = {M: {} for M in Ms}
best_configs = {}

# --- Calculs ---
for i in range(1, 6):
    best_mu = float('inf')
    best_M = None
    best_Phi = None
    
    for M in Ms:
        Phi_tmp = get_phi(i, M, N)
        mu_val = mutual_coherence(Phi_tmp, D_dct, N)
        all_results[M][f"Phi{i}"] = mu_val
        
        if mu_val < best_mu:
            best_mu = mu_val
            best_M = M
            best_Phi = Phi_tmp
    
    best_configs[f"y{i}"] = {"M": best_M, "mu": best_mu, "matrix": best_Phi}

# --- Affichage du Tableau Trié ---
print("TABLEAU DE LA COHÉRENCE MUTUELLE mu(Phi, D)")
header = "M | " + " | ".join([f"Phi{i:1}  " for i in range(1, 6)])
print("-" * len(header))
print(header)
print("-" * len(header))
for M in Ms:
    line = f"{M:3} | " + " | ".join([f"{all_results[M][f'Phi{i}']:7.4f}" for i in range(1, 6)])
    print(line)

# --- Affichage des Vecteurs y (Complets) ---
print("\n" + "="*80)
print("VECTEURS DE MESURE OPTIMAUX (y = Phi * x)")
print("="*80)

for i in range(1, 6):
    config = best_configs[f"y{i}"]
    y_vec = config["matrix"] @ x
    print(f"\n>>> y{i} (Matrice Phi{i} | M = {config['M']} | mu_min = {config['mu']:.4f}) :")
    # np.set_printoptions pour afficher tout le vecteur sans troncature (...)
    np.set_printoptions(threshold=np.inf)
    print(y_vec.flatten())