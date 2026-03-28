import numpy as np
 
eps = 0.000001
kmax = 1000
t = 2.5
 
D = np.array([
    [1,1,2,5,0,0,3,-2,1,2,2,2,5,1,3,1,-1,2,9,5,5,1,1,5],
    [0,-1,4,2,-1,1,0,0,5,0,2,2,7,-12,2,5,5,2,7,4,-9,-2,1,2],
    [1,3,1,1,5,1,2,2,1,1,1,1,5,0,-1,1,0,1,2,1,1,2,5,5],
    [0,1,5,1,5,2,2,-2,5,0,-4,5,1,5,0,0,-1,-4,-8,2,2,-1,1,0],
    [0,-1,2,3,2,2,3,1,1,0,0,0,0,4,-1,-2,0,7,4,3,4,-1,1,0],
    [-1,8,6,3,2,2,2,4,-2,-3,-4,1,1,1,1,0,-2,-3,4,1,1,-1,1,0]
])
 
X = np.array([-10, -10, 10, 20, 15, 10])
 
 
def StOMP(X, D, kmax, eps, t):
    k = 0
    N, K = D.shape
 
    alpha = np.zeros(K)      # solution finale de taille K
    P = []                   # support courant
    Residuel = X.copy()      # résiduel initial r(0) = X
 
    while k < kmax and np.linalg.norm(Residuel) > eps:
 
        # 1) Calcul du vecteur des contributions
        # C_j = |<d_j , Residuel>| / ||d_j||
 
        C = np.zeros(K)
        for j in range(K):
            dj = D[:, j]
            norme_dj = np.linalg.norm(dj)
 
            if norme_dj != 0:
                C[j] = np.abs(dj.T @ Residuel) / norme_dj
            else:
                C[j] = 0
 
        # 2) Calcul du seuil
        # S^(k) = t * ||Residuel|| / sqrt(K)
        seuil = t * np.linalg.norm(Residuel) / np.sqrt(K)
 
        # 3) Sélection des nouveaux atomes :
        # on prend tous les j tels que C_j > seuil
 
        Lambda = []
        for j in range(K):
            if C[j] > seuil:
                Lambda.append(j)
 
        # Si aucun nouvel atome n'est sélectionné, on arrête
        if len(Lambda) == 0:
            print("Arrêt : aucun nouvel atome sélectionné à l'itération", k + 1)
            break
 
        # 4) Mise à jour du support P = P U Lambda
 
        for j in Lambda:
            if j not in P:
                P.append(j)
        P.sort()
 
        # 5) Construction de la sous-matrice DS
        # DS = D[:, P]
        DS = D[:, P]
 
        # 6) Résolution du problème des moindres carrés
        # alphak = argmin ||X - DS alphak||_2
        alphak, _, _, _ = np.linalg.lstsq(DS, X, rcond=None)
 
        # 7) Mise à jour du résiduel
        Residuel = X - DS @ alphak

 
        # 8) Reconstruction du vecteur alpha complet
        alpha = np.zeros(K)
        alpha[P] = alphak
 
        k = k + 1

 
    return alpha, P, Residuel, k
 
 
test = StOMP(X, D, kmax, eps, t)
 
print("Valeur de alpha\n", test[0], "\n")
print("La position des atomes", test[1])
print("La norme du résiduel", np.linalg.norm(test[2]), "a l'itération", test[3])
print("La taille du dictionnaire D :", D.shape)