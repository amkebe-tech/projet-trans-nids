"""
Module d'explicabilité (XAI).

Ce module définit la fonction explain() qui, pour une fenêtre de trafic et
un verdict donnés, produit une explication indiquant quelles features ont le
plus influencé la décision.

IMPORTANT - Rôle de ce fichier dans le projet :
------------------------------------------------
À terme (cf. cahier des charges section 1.3 "Explicabilité et robustesse"),
ce module utilisera SHAP, LIME et surtout TRUSTEE (un wrapper personnalisé
qui convertit le score continu du VAE en classes binaires pour en extraire
des règles logiques SI...ALORS).

Pour l'instant, ces bibliothèques ne sont pas branchées : explain() simule
une explication de façon simple (poids aléatoires par feature), mais
respecte le même format de sortie que ce que produiront SHAP/TRUSTEE plus
tard. Cela permet de construire et tester l'affichage de l'explication dans
Streamlit dès maintenant.

Format du contrat :
--------------------
Entrée  : window     -> tableau numpy (WINDOW_SIZE, len(FEATURES)), la même
                         fenêtre que celle passée à model_interface.predict()
          prediction -> dict retourné par predict() ({"score", "verdict"})
Sortie  : dict {"methode": str, "importances": dict{feature: float}}
          - "importances" associe à chaque nom de feature (config.FEATURES)
            un poids d'importance. Plus la valeur est grande (en valeur
            absolue), plus la feature est jugée responsable du verdict.

Convention de signe (alignée sur SHAP, reprise par l'affichage Streamlit) :
    un poids POSITIF pousse le score vers l'anomalie (affiché en rouge),
    un poids NÉGATIF pousse le score vers la normalité (affiché en vert).
"""

import numpy as np

# Import "voisin" (pas "from app.config import ...") : Streamlit exécute le
# script en ajoutant le dossier app/ à sys.path, pas la racine du dépôt.
from config import FEATURES, RANDOM_SEED

_rng = np.random.default_rng(RANDOM_SEED + 1)  # graine différente de model_interface pour varier les tirages


def explain(window: np.ndarray, prediction: dict) -> dict:
    """
    Génère une explication simulée pour une prédiction donnée.

    Args:
        window: tableau numpy (WINDOW_SIZE, len(FEATURES)), la fenêtre
                analysée par predict().
        prediction: dict retourné par model_interface.predict()
                    (contient "score" et "verdict").

    Returns:
        dict avec :
            - "methode": nom de la méthode d'explication utilisée
            - "importances": dict {nom_feature: poids d'importance},
              une entrée par feature définie dans config.FEATURES

    Note pédagogique :
        Les poids retournés ici sont tirés aléatoirement, ils ne reflètent
        AUCUNE analyse réelle des contributions des features. C'est un
        placeholder en attendant l'intégration de SHAP/LIME/TRUSTEE.
    """
    # --- PLACEHOLDER ---
    # Le vrai module utilisera SHAP/LIME pour calculer la contribution
    # réelle de chaque feature à l'erreur de reconstruction du VAE, ou
    # TRUSTEE pour extraire une règle SI...ALORS à partir d'un arbre de
    # décision approximant le modèle.
    poids_bruts = _rng.uniform(low=-1.0, high=1.0, size=len(FEATURES))

    # On normalise pour que la somme des valeurs absolues fasse 1, comme le
    # font typiquement les scores SHAP (contribution relative de chaque
    # feature).
    somme_abs = np.sum(np.abs(poids_bruts))
    poids_normalises = poids_bruts / somme_abs

    importances = {
        feature: float(poids)
        for feature, poids in zip(FEATURES, poids_normalises)
    }

    return {
        "methode": "placeholder (aléatoire) - à remplacer par SHAP/LIME/TRUSTEE",
        "importances": importances,
    }
