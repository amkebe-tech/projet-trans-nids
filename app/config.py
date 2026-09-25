"""
Configuration centrale du projet NIDS.

Ce fichier est LA source unique de vérité pour tout ce qui décrit le format
des données attendues par le modèle (noms des features, taille de la fenêtre
temporelle, etc.).

Pourquoi centraliser ça ici ?
-----------------------------
Le projet est encore en phase d'expérimentation : le nombre et le nom des
features utilisées par le VAE/LSTM peuvent encore changer (on est actuellement
à 5 features, mais on envisage d'en ajouter jusqu'à 15-20).

Si on écrivait "5" ou les noms de colonnes directement dans model_interface.py,
explainability.py et streamlit_app.py, il faudrait modifier CHAQUE fichier à
chaque changement de features, avec un gros risque d'oubli et
d'incohérence entre les fichiers.

En centralisant tout ici, il suffit de modifier la liste FEATURES ci-dessous :
tout le reste du code (predict(), explain(), l'interface Streamlit) s'adapte
automatiquement car il utilise FEATURES et len(FEATURES), jamais de valeurs
écrites en dur.
"""

# Liste des features utilisées par le modèle, dans l'ordre attendu par les
# colonnes du CSV en entrée.
# Ce sont pour l'instant les 5 features "de base" utilisées pour le
# prototype. Cette liste sera enrichie (potentiellement 15-20 colonnes)
# une fois les expérimentations sur Colab terminées.
FEATURES = [
    "flow_duration",
    "fwd_pkts",
    "bwd_pkts",
    "fwd_bytes",
    "bwd_bytes",
]

# Nombre de pas de temps (lignes consécutives) qui composent une fenêtre
# analysée par le modèle. Le VAE-LSTM du projet final analyse le trafic sous
# forme de séquences temporelles plutôt que ligne par ligne.
WINDOW_SIZE = 10

# Seuil (entre 0 et 1) au-delà duquel un score d'anomalie déclenche une
# alerte. Purement arbitraire pour le placeholder actuel : le vrai seuil
# sera calibré statistiquement une fois le VAE entraîné (ex: percentile 95
# de l'erreur de reconstruction sur le trafic sain).
SEUIL_ANOMALIE = 0.5

# Graine aléatoire fixée pour que les résultats du placeholder soient
# reproductibles d'une exécution à l'autre (cf. exigence de reproductibilité
# du cahier des charges).
RANDOM_SEED = 42
