import numpy as np
# --- 2. ORTHOGONAL MATCHING PURSUIT (OMP) ---
# Basé sur TD2 - Exercices 3 et 4
def orthogonal_matching_pursuit(x, D, n_iter=None, tol=1e-6):
    N, K = D.shape
    residual = x.copy()
    support = []
    alpha = np.zeros(K)
    
    # Critère d'arrêt : nombre d'itérations ou tolérance (TD2-2 Page 5)
    max_k = n_iter if n_iter is not None else N
    
    for k in range(max_k):
        # 1. Sélection : m_k = argmax |<dj, R>|
        contributions = np.abs(D.T @ residual)
        m_k = np.argmax(contributions)
        
        if m_k not in support:
            support.append(m_k)
        else:
            break # L'atome est déjà présent
            
        # 2. Mise à jour par Moindres Carrés (TD2 Ex 4)
        # On résout min ||x - D_supp * alpha_supp||2
        D_active = D[:, support]
        # Utilisation de la pseudo-inverse pour la projection orthogonale
        alpha_active = np.linalg.lstsq(D_active, x, rcond=None)[0]
        
        # 3. Nouveau résiduel
        residual = x - D_active @ alpha_active
        
        if np.linalg.norm(residual) < tol:
            break
            
    alpha[support] = alpha_active
    return alpha, residual

# --- AJOUT : Définition de la fonction PSNR ---
def calculate_psnr(X, X_hat):
    mse = np.mean((X - X_hat)**2)
    if mse == 0: return 100
    max_val = np.max(X) # Valeur max du signal
    return 20 * np.log10(max_val / np.sqrt(mse))


def k_svd(X, K, s, n_iterations):
    N, L = X.shape
    
    # --- ANCIENNE MÉTHODE D'INITIALISATION (Aléatoire) ---
    # D = np.random.randn(N, K)
    # D /= np.linalg.norm(D, axis=0)
    
    # --- NOUVELLE MÉTHODE (Basée sur l'énoncé : premières colonnes de X) ---
    D = X[:, :K].copy() 
    D /= np.linalg.norm(D, axis=0) 
    
    Lambda = np.zeros((K, L))
    erreurs_historique = []
    
    for it in range(n_iterations):
        # ÉTAPE 1 : Codage parcimonieux
        for i in range(L):
            Lambda[:, i], _ = orthogonal_matching_pursuit(X[:, i], D, n_iter=s)
            
        # ÉTAPE 2 : Mise à jour du dictionnaire
        for k in range(K):
            wk = np.where(Lambda[k, :] != 0)[0]
            
            # --- MODIFICATION SELON ÉNONCÉ (Point 2) ---
            if len(wk) == 0:
                # ANCIEN : D[:, k] = X[:, np.random.randint(L)]
                D[:, k] = X[:, np.random.randint(L)] 
                D[:, k] /= np.linalg.norm(D[:, k])
                continue 
            
            # --- CALCUL DE L'ERREUR RÉDUITE ---
            # ANCIENNE MÉTHODE (Version optimisée simple) :
            # Ek = (X - D @ Lambda)[:, wk] + np.outer(D[:, k], Lambda[k, wk])
            
            # NOUVELLE MÉTHODE (Plus proche du cours théorique) :
            Ek = X[:, wk] - np.delete(D, k, axis=1) @ np.delete(Lambda, k, axis=0)[:, wk]
            
            U, S, Vh = np.linalg.svd(Ek, full_matrices=False)
            
            D[:, k] = U[:, 0]
            Lambda[k, wk] = S[0] * Vh[0, :]

        # --- ÉTAPE 3 : CRITÈRE D'ARRÊT PSNR (Point 3 de l'énoncé) ---
        # Ce bloc doit être APRES la boucle 'for k' pour évaluer le dictionnaire complet
        X_hat = D @ Lambda
        current_psnr = calculate_psnr(X, X_hat)
        
        # On stocke le PSNR dans l'historique
        erreurs_historique.append(current_psnr)

        # --- ANCIENNE MÉTHODE D'AFFICHAGE (Erreur relative) ---
        # err = np.linalg.norm(X - D @ Lambda, 'fro') / np.linalg.norm(X, 'fro')
        # print(f"Itération {it+1}/{n_iterations} | Erreur: {err:.4f}")

        # NOUVEL AFFICHAGE (PSNR)
        print(f"Itération {it+1}/{n_iterations} | PSNR: {current_psnr:.2f} dB")
        
        # NOUVEAU CRITÈRE D'ARRÊT :
        if current_psnr > 35: # Seuil de qualité 35 dB
            print("Critère PSNR atteint. Arrêt prématuré.")
            break
            
    return D, Lambda, erreurs_historique


# ==========================================
# TEST SIMPLIFIÉ POUR K-SVD (MIS À JOUR PSNR)
# ==========================================

# 1. Création de données de test (100 signaux de dimension 20)
X_test = np.random.randn(20, 100) 

# 2. Paramètres
K_atomes = 30    
parcimonie = 4   
itérations = 10  

# 3. Lancement de l'apprentissage
D_appris, L_appris, erreurs = k_svd(X_test, K=K_atomes, s=parcimonie, n_iterations=itérations)

# 4. Affichage des résultats pour vérification
print("\n" + "="*50)
print("        VÉRIFICATION DE L'APPRENTISSAGE K-SVD")
print("="*50)

if len(erreurs) > 0:
    # --- MODIFICATION PSNR ---
    # ANCIEN : print(f"Erreur au début (Itération 1) : {erreurs[0]:.4f}")
    # ANCIEN : print(f"Erreur à la fin (Itération {len(erreurs)}) : {erreurs[-1]:.4f}")
    print(f"PSNR au début (Itération 1) : {erreurs[0]:.2f} dB")
    print(f"PSNR à la fin (Itération {len(erreurs)}) : {erreurs[-1]:.2f} dB")

    # 5. Interprétation automatique
    # ANCIEN : if erreurs[-1] < erreurs[0]: (L'erreur baisse)
    if erreurs[-1] > erreurs[0]: # LE PSNR MONTE = SUCCÈS
        print("\nANALYSE : SUCCÈS")
        # ANCIEN : print(f"L'erreur a diminué de ...")
        gain_psnr = erreurs[-1] - erreurs[0]
        print(f"Le PSNR a augmenté de {gain_psnr:.2f} dB.")
        print("Le dictionnaire s'est adapté pour mieux représenter les signaux.")
    else:
        print("\nANALYSE : STAGNATION")
        print("Le PSNR n'a pas augmenté. Vérifiez vos paramètres ou vos données.")
else:
    print("Erreur : La liste 'erreurs' est vide.")

print("="*50)

# ============================================================
#            RÉSULTATS DU TEST K-SVD (Réf: TD3)
# ============================================================

signal_index = 0
x_original = X_test[:, signal_index]
coefficients_appris = L_appris[:, signal_index]
support_trouve = np.where(np.abs(coefficients_appris) > 1e-5)[0]

x_reconstruit = D_appris @ coefficients_appris
erreur_reconstruction = np.linalg.norm(x_original - x_reconstruit)

print("\n" + "="*60)
print("        ANALYSE DÉTAILLÉE D'UN SIGNAL (SIGNAL 0)")
print("="*60)
print(f"Support trouvé (L0={parcimonie}) : {support_trouve}")
print(f"Coefficients non nuls    : {np.round(coefficients_appris[support_trouve], 2)}")
print(f"Erreur de reconstruction : {erreur_reconstruction:.2e}")

# --- MODIFICATION AFFICHAGE FINAL ---
# ANCIEN : print(f"Erreur moyenne du dictionnaire : {erreurs[-1]:.4f}")
print(f"PSNR final du dictionnaire : {erreurs[-1]:.2f} dB")

if erreur_reconstruction < 1e-10:
    print("Résultat : Reconstruction exacte.")
else:
    print(f"Résultat : Approximation parcimonieuse (s={parcimonie}).")

print("="*60)

# --- COMPARAISON : DICTIONNAIRE ALÉATOIRE VS APPRIS ---

D_aleatoire = np.random.randn(X_test.shape[0], K_atomes)
D_aleatoire /= np.linalg.norm(D_aleatoire, axis=0)

a_init, r_init = orthogonal_matching_pursuit(X_test[:, 0], D_aleatoire, n_iter=parcimonie)
# Calcul du PSNR pour l'initial
x_hat_init = D_aleatoire @ a_init
psnr_init = calculate_psnr(X_test[:, 0], x_hat_init)

a_final, r_final = orthogonal_matching_pursuit(X_test[:, 0], D_appris, n_iter=parcimonie)
# Calcul du PSNR pour le final
x_hat_final = D_appris @ a_final
psnr_final = calculate_psnr(X_test[:, 0], x_hat_final)

print("\n" + "="*60)
print("   COMPARAISON : EFFICACITÉ DE L'APPRENTISSAGE (PSNR)")
print("="*60)
# ANCIEN : print(f"Erreur OMP (Dictionnaire aléatoire) : {err_init:.4f}")
print(f"PSNR OMP (Dictionnaire aléatoire) : {psnr_init:.2f} dB")
print(f"PSNR OMP (Dictionnaire appris)    : {psnr_final:.2f} dB")

gain_db = psnr_final - psnr_init
print(f"\nGain de qualité grâce au K-SVD : +{gain_db:.2f} dB")
print("="*60)

# =================================================================
#                        CONCLUSION GÉNÉRALE
# =================================================================
"""
OBSERVATIONS :
1. L'OMP permet de trouver les meilleurs coefficients pour un dictionnaire donné.
2. Le K-SVD fait monter le PSNR au fil des itérations, ce qui prouve que 
   les atomes du dictionnaire deviennent plus représentatifs des données X.
3. Un gain positif en dB confirme que l'apprentissage est efficace par rapport 
   à un dictionnaire aléatoire.
"""