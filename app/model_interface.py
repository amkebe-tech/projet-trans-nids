"""
Interface du modèle de détection (contrat modèle).

Ce module définit LA fonction que le reste de l'application (Streamlit)
utilise pour obtenir un verdict sur un morceau de trafic réseau : predict().

IMPORTANT - Rôle de ce fichier dans le projet :
------------------------------------------------
Le vrai modèle (VAE et/ou LSTM entraînés sur Colab, cf. cahier des charges
section 1.3) n'est pas encore prêt. Ce fichier contient donc un PLACEHOLDER :
il simule un score d'anomalie de façon simple (aléatoire), mais respecte
exactement le même format d'entrée/sortie que le futur vrai modèle.

Quand le vrai modèle sera prêt, il suffira de remplacer le contenu de
predict() (charger les poids .pth depuis models/, faire l'inférence PyTorch)
SANS changer sa signature. Le reste de l'application (streamlit_app.py)
n'aura donc aucune modification à faire.

Format du contrat :
--------------------
Entrée  : window -> tableau numpy de forme (WINDOW_SIZE, len(FEATURES))
          C'est une fenêtre glissante de trafic réseau : WINDOW_SIZE pas de
          temps consécutifs, chacun décrit par len(FEATURES) valeurs
          numériques (dans l'ordre défini par config.FEATURES).
Sortie  : dictionnaire {"score": float, "verdict": str}
          - score   : un nombre entre 0 et 1 représentant le degré d'anomalie
                       (0 = trafic parfaitement normal, 1 = anomalie totale)
          - verdict : "normal" ou "anomalie", dérivé du score et du seuil
"""

import numpy as np

# Import "voisin" (pas "from app.config import ...") : Streamlit exécute le
# script en ajoutant le dossier app/ à sys.path, pas la racine du dépôt.
from config import FEATURES, WINDOW_SIZE, SEUIL_ANOMALIE, RANDOM_SEED

# Générateur aléatoire dédié et reproductible (on évite np.random.seed()
# global qui affecterait tout le programme, y compris d'autres modules).
_rng = np.random.default_rng(RANDOM_SEED)


def predict(window: np.ndarray) -> dict:
    """
    Calcule un score d'anomalie pour une fenêtre de trafic réseau.

    Args:
        window: tableau numpy de forme (WINDOW_SIZE, len(FEATURES)).
                Chaque ligne = un pas de temps, chaque colonne = une feature
                (dans l'ordre de config.FEATURES).

    Returns:
        dict avec deux clés :
            - "score": float entre 0 et 1 (score d'anomalie)
            - "verdict": "normal" ou "anomalie"

    Note pédagogique :
        Pour l'instant, le score est simulé (tirage aléatoire). Il ne reflète
        AUCUNE analyse réelle du trafic. C'est uniquement pour permettre de
        développer et tester l'interface Streamlit avant que le vrai modèle
        (VAE/LSTM) ne soit disponible.
    """
    _valider_forme_fenetre(window)

    # --- PLACEHOLDER ---
    # Le vrai modèle calculera ici une erreur de reconstruction (VAE) ou une
    # probabilité issue du LSTM. En attendant, on simule un score aléatoire
    # uniforme entre 0 et 1.
    score = float(_rng.uniform(0.0, 1.0))

    verdict = "anomalie" if score > SEUIL_ANOMALIE else "normal"

    return {"score": score, "verdict": verdict}


def _valider_forme_fenetre(window: np.ndarray) -> None:
    """
    Vérifie que la fenêtre passée en entrée a bien la forme attendue par le
    contrat : (WINDOW_SIZE, len(FEATURES)).

    On lève une erreur explicite plutôt que de laisser le modèle planter
    plus loin avec un message cryptique.
    """
    forme_attendue = (WINDOW_SIZE, len(FEATURES))
    if window.shape != forme_attendue:
        raise ValueError(
            f"Forme de fenêtre invalide : reçu {window.shape}, "
            f"attendu {forme_attendue} "
            f"(WINDOW_SIZE={WINDOW_SIZE} pas de temps x "
            f"{len(FEATURES)} features définies dans config.FEATURES)."
        )
