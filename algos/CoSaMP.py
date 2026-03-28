import numpy as np
from scipy.linalg import pinv

def CoSaMP(x, D, kmax, epsilon, s):
    N, K = D.shape
    alpha = np.zeros(K)
    residuel = x.copy()
    supp = set()  # Support initial vide
    k = 1
    
    while k <= kmax and np.linalg.norm(residuel) > epsilon:
        # 1. La sélection : 2s atomes de plus grande contribution
        c = np.abs(D.T @ residuel) / np.linalg.norm(D)
        supp1 = set(np.argsort(c)[-2*s:])
        
        # 2. Mise à jour du support : Fusionner avec le support précédent
        merged_supp = list(supp | supp1)
        
        # 3. Estimation : Moindres carrés sur le support mis à jour
        As = D[:, merged_supp]
        z = pinv(As) @ x
        
        # 4. Rejet : Garder les s plus grands coefficients de |z|
        local_best = np.argsort(np.abs(z))[-s:]
        # On remplace l'ancien support par les s meilleurs atomes
        supp = set([merged_supp[i] for i in local_best])
        
        # 5. Estimation : Moindres carrés sur le support final après rejet
        alpha = np.zeros(K)
        final_indices = list(supp)
        Af = D[:, final_indices]
        alpha[final_indices] = pinv(Af) @ x
        
        # 6. Mise à jour du résiduel
        residuel = x - D @ alpha
        k += 1
        
    return alpha, list(supp), residuel, np.linalg.norm(residuel), k-1

# --- DONNÉES DE TEST (SANS L'ERREUR TYPEERROR) ---
epsilon = 0.000001
kmax = 100
s = 6  # Ordre de parcimonie

# LE FIX EST ICI : On met un crochet supplémentaire [ ] pour englober toutes les lignes
D = np.array([[1, 1, 2, 5, 0, 0, 3, -2, 1, 2, 2, 2, 5, 1, 3, 1, -1, 2, 9, 5, 5, 1, 1, 5],
    [0, -1, 4, 2, -1, 1, 0, 0, 5, 0, 2, 2, 7, -12, 2, 5, 5, 2, 7, 4, -9, -2, 1, 2],
    [1, 3, 1, 1, 5, 1, 2, 2, 1, 1, 1, 1, 5, 0, -1, 1, 0, 1, 2, 1, 1, 2, 5, 5],
    [0, 1, 5, 1, 5, 2, 2, -2, 5, 0, -4, 5, 1, 5, 0, 0, -1, -4, -8, 2, 2, -1, 1, 0],
    [0, -1, 2, 3, 2, 2, 3, 1, 1, 0, 0, 0, 0, 4, -1, -2, 0, 7, 4, 3, 4, -1, 1, 0],
    [-1, 8, 6, 3, 2, 2, 2, 4, -2, -3, -4, 1, 1, 1, 1, 0, -2, -3, 4, 1, 1, -1, 1, 0]])

x = np.array([-10, -10, 10, 20, 15, 10])

# --- EXÉCUTION ---
a_res, s_res, r_res, n_res, it_res = CoSaMP(x, D, kmax, epsilon, s)

print(f"Indices du support final : {s_res}")
print(f"Norme du résiduel final : {n_res:.8f}")
print(a_res)