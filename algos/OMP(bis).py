import numpy as np
import time  # Import pour le temps

def OMP(x, D, kmax, epsilon):
    N,K = D.shape
    alpha = np.zeros(K)
    residuel = x.copy()
    Dk= []
    P = []
    historique_residu = []
    k = 0
    while k < kmax and np.linalg.norm(residuel) > epsilon:
        mk = np.argmax(np.abs(D.T @ residuel)/np.linalg.norm(D))
        P.append(mk)
        Dk = D[:,P]
        
        
        alpha_k, _, _, _ = np.linalg.lstsq(Dk, x, rcond=None)
        residuel = x - Dk @ alpha_k
        norme_courante = np.linalg.norm(residuel)
        historique_residu.append(norme_courante)
        k = k+1
        print(f"Itération {k}: Norme du résiduel = {norme_courante:.8f}")
     
    alpha[P] = alpha_k
    norme = np.linalg.norm(residuel)
    return alpha, P, residuel,norme, k


# --- Données initiales ---


epsilon=0.000001
kmax=1000
D=np.array([[1,1,2,5,0,0,3,-2,1,2,2,2,5,1,3,1,-1,2,9,5,5,1,1,5],
[0,-1,4,2,-1,1,0,0,5,0,2,2,7,-12,2,5,5,2,7,4,-9,-2,1,2],
[1,3,1,1,5,1,2,2,1,1,1,1,5,0,-1,1,0,1,2,1,1,2,5,5],
[0,1,5,1,5,2,2,-2,5,0,-4,5,1,5,0,0,-1,-4,-8,2,2,-1,1,0],
[0,-1,2,3,2,2,3,1,1,0,0,0,0,4,-1,-2,0,7,4,3,4,-1,1,0],
[-1,8,6,3,2,2,2,4,-2,-3,-4,1,1,1,1,0,-2,-3,4,1,1,-1,1,0]])
x=np.array([-10,-10,10,20,15,10])

# --- Tests et affichages ---

test = OMP(x, D, kmax, epsilon)

print("-----------------------------------------------------------------------------\n")
print("Taille de D: \n", D.shape)
print("Alpha: \n",test[0])
print("Atome choisis \n", test[1])
print("Le résiduel est \n", test[2])
print("La norme du résiduel est \n", test[3])
print("Le nombre d iterations est : \n", test[4])


# --- Mesure du temps ---

debut = time.time()  # On lance le chrono
test = OMP(x, D, 1000, 0.000001)
fin = time.time()    # On arrête le chrono

temps_exec = fin - debut

# --- Affichage ---
print(f"Temps d'exécution : {temps_exec:.6f} secondes \n")


# --- Calculs de parcimonie ---
# La norme L0 est le nombre d'éléments dont la valeur absolue est supérieure à un petit seuil

norme_L0 = np.sum(np.abs(test[0]) > 0) 
nombre_de_zeros = len(test[0]) - norme_L0

print(f"Norme L0 (nombre de valeurs non nulles) : {norme_L0}")
print(f"Nombre de zéros dans Alpha : {nombre_de_zeros} / {len(test[0])}")