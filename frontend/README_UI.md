Lancer l'interface **depuis un terminal**, à la racine du dépôt :

```bash
python3 frontend/app.py
```

Si la fenêtre se ferme tout de suite, relance depuis le terminal (pas un double-clic sans console) : tu verras l’erreur, ou une boîte de dialogue « erreur au démarrage » / « erreur (interface) ». Un fichier de log peut aussi apparaître via `logging` sur stderr.

Cette interface couvre :
- reconstruction BCS (Φ₁–Φ₄, dictionnaires DCT / mixte / K-SVD, solveurs)
- résultats (tableau, graphique, aperçu)
- **Analyses & graphiques** : sweep PSNR vs ratios, export CSV (M, cohérence μ, erreurs sur 3 vecteurs), aperçu des CSV, estimation CO₂ de session
- onglet Patchs : **une image** avec la grille B×B (`grille_sur_image`)

Voir le **README** à la racine du dépôt pour l’ensemble du projet.
