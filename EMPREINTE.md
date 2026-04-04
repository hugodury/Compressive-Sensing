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

Les constantes sont modifiables dans `setupParam` / le dict `params` :

- `empreinte_puissance_w`
- `empreinte_g_co2_par_kwh`

Pour **désactiver** le calcul : `empreinte_carbone=False`.  
Pour calculer sans afficher sur la console : `empreinte_afficher_console=False`.

## Fichiers produits

- Chaque résultat de `main_backend` peut contenir la clé `empreinte` (détails + message lisible).
- Lors d’un `run_pipeline`, une synthèse session est dans `empreinte_session`.
- `save_results` écrit `empreinte_estimation.txt` dans le dossier horodaté si `empreinte` est présent.

## Choix du projet pour limiter l’impact

- **Pas de GPU imposé** : le code tourne en CPU / NumPy ; pas d’entraînement de gros modèles par défaut.
- **Réduction du travail en développement** : `max_patches` dans `patch_params` limite le nombre de patchs traités pour des essais rapides.
- **Tests légers** : la suite smoke évite des pipelines complètes inutiles en CI locale.
- **Vectorisation** : calculs matriciels plutôt que boucles Python lourdes là où c’était pertinent.
- **Étapes optionnelles** : tableaux §6 avec `--no-tableaux-erreurs` ou moins d’itérations réduit le temps CPU.

Pour un rapport sérieux, croiser cette estimation avec des outils type **CodeCarbon** / mesure réelle sur ta machine si besoin.
