# Empreinte carbone — estimation et limites

Le projet affiche une **estimation indicative** de l’impact climatique lié à l’exécution locale du code (sensibilisation du rapport PDF). Ce n’est **pas** une mesure certifiée : elle sert à l’ordre de grandeur et à comparer des runs entre eux.

## Méthode

1. **Temps mur** (`perf_counter`) sur la durée de `main_backend` ou de toute la session `run_pipeline`.
2. **Temps CPU processus** (optionnel, Unix) : delta `utime + stime` via `resource.getrusage` — utile pour voir si le calcul est CPU-bound.
3. **Énergie électrique approximative** :  
   `énergie (Wh) ≈ (puissance_moyenne_W / 1000) × durée_en_heures`  
   avec une hypothèse par défaut de **~45 W** pour un portable sous charge numérique modérée (très variable selon machine).
4. **Émissions CO₂eq** :  
   `g CO₂eq ≈ (énergie_kWh) × intensité_carbone (g/kWh)`  
   avec par défaut **~85 g/kWh**, ordre de grandeur du mix électrique français (à ajuster pour un autre pays ou une autre source).

5. **Fourchette mural / CPU** : le même produit « puissance × durée » est aussi calculé avec le **temps CPU processus** (Unix `getrusage`) lorsqu’il est disponible. Cela donne une **borne basse indicative** si l’énergie ne correspondait qu’aux secondes CPU comptées ; le **temps mural** reste en général un **majorant** plus prudent pour un PC qui fait autre chose en parallèle. La réalité se situe souvent **entre** ces deux ordres de grandeur.

Les constantes sont modifiables dans `setupParam` / le dict `params` :

- `empreinte_puissance_w`
- `empreinte_g_co2_par_kwh`

Pour **désactiver** le calcul : `empreinte_carbone=False`.  
Pour calculer sans afficher sur la console : `empreinte_afficher_console=False`.

## Fichiers produits

- Chaque résultat de `main_backend` peut contenir la clé `empreinte` (détails + message lisible).
- Lors d’un `run_pipeline`, une synthèse session est dans `empreinte_session`.
- `save_results` écrit `empreinte_estimation.txt` dans le dossier horodaté si `empreinte` est présent (y compris `co2e_g_estime_temps_cpu` si mesuré).
- `save_results` écrit aussi `stockage_compression.txt` : comparaison taille raster/fichier vs modèle théorique « mesures + Φ » (voir `backend/utils/stockage_compressif.py`).

## Choix du projet pour limiter l’impact

- **Pas de GPU imposé** : le code tourne en CPU / NumPy ; pas d’entraînement de gros modèles par défaut.
- **Réduction du travail en développement** : `max_patches` dans `patch_params` limite le nombre de patchs traités pour des essais rapides.
- **Tests légers** : la suite smoke évite des pipelines complètes inutiles en CI locale.
- **Vectorisation** : calculs matriciels plutôt que boucles Python lourdes là où c’était pertinent.
- **Étapes optionnelles** : tableaux §6 avec `--no-tableaux-erreurs` ou moins d’itérations réduit le temps CPU.

Pour un rapport sérieux, croiser cette estimation avec des outils type **CodeCarbon** / mesure réelle sur ta machine si besoin.

## Rapport avec la « réalité » énergétique

- Le modèle utilise le **temps mur** (horloge) × une **puissance moyenne supposée** : ce n’est **pas** la puissance instantanée mesurée sur la prise ou par le CPU (TDP, capteurs RAPL, etc.).
- Si le processeur est en partie au repos pendant l’exécution, l’estimation peut **surestimer** l’énergie réelle ; si d’autres processus chargent la machine en parallèle, elle peut **sous-estimer** la part due au projet.
- La formule **Wh = P(W) × Δt(h)** puis **g CO₂eq = kWh × intensité (g/kWh)** est cohérente pour un ordre de grandeur **comparatif entre deux runs** sur la même machine avec les mêmes hypothèses P et intensité.
- Pour une valeur « auditée », il faudrait un wattmètre, des données grid-specific temps réel, ou un outil de mesure logicielle calibré.
