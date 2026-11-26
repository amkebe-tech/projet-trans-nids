"""
Configuration centrale du projet NIDS (moteur V7).

Ce fichier reste la source unique de vérité pour ce que l'interface affiche
du modèle. Les listes de variables viennent directement du moteur V7
(app/moteur_v7.py), lui-même aligné sur le notebook NIDS-v7-kaggle.ipynb :
si le moteur change, l'interface suit sans modification.
"""
from pathlib import Path

from moteur_v7 import CTX, DETECTEURS_V7, NOMS_COURTS, features_v3

RACINE = Path(__file__).resolve().parent.parent

# 13 variables de flux (VAE, kNN) et 18 variables de contexte (Isolation Forest)
FEATURES = list(features_v3(*[[0]] * 6).columns)
FEATURES_CONTEXTE = list(CTX)
DETECTEURS = list(DETECTEURS_V7)
NOMS_DETECTEURS = dict(NOMS_COURTS)

# Fichiers
CHEMIN_MODELE = RACINE / "models" / "systeme_v7.joblib"
DOSSIER_ETATS = RACINE / "models" / "etats"          # un état calibré par réseau surveillé
DOSSIER_DEMO = RACINE / "donnees_demo"
DEMO_CALIBRATION = DOSSIER_DEMO / "uwf24_calibration_24h.conn.log"
DEMO_ANALYSE = DOSSIER_DEMO / "uwf24_analyse.conn.log"

# Seuils proposés : taux de fausses alarmes fixé sur la calibration
SEUILS = ["1 %", "5 %"]
SEUIL_PAR_DEFAUT = "1 %"

# Le score V7 est ramené entre 0 et 1 pour la jauge : il vaut exactement 0,5
# au seuil choisi (niveau = score / (score + seuil)).
SEUIL_ANOMALIE = 0.5

# Objectifs du cahier des charges
OBJECTIFS = {"Précision": 0.90, "Rappel": 0.95, "F1": 0.93, "AUC-ROC": 0.97}

# Nombre maximal d'alertes listées dans l'interface (les plus fortes d'abord)
MAX_ALERTES_AFFICHEES = 500
