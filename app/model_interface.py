"""
Interface du modèle de détection (contrat modèle) — moteur V7.

L'interface Streamlit ne parle qu'à ce module. Il cache le fonctionnement
interne du moteur (app/moteur_v7.py) derrière quatre fonctions :

    charger()                    -> le modèle entraîné (models/systeme_v7.joblib)
    calibrer(fichier_sain)       -> état calibré du réseau (24 h de trafic NORMAL)
    analyser(fichier, etat)      -> un tableau avec un score et une alerte par flux
    predict(ligne, etat, seuil)  -> {"score": 0..1, "verdict": "normal" | "anomalie"}

Le format de sortie de predict() est celui du prototype initial : la jauge
et la carte de verdict de l'interface n'ont pas changé. Le score vaut 0,5
exactement au seuil choisi.

Pourquoi une calibration ? Le système n'apprend que du trafic normal (logique
zero-day). Sur un réseau nouveau, il doit d'abord apprendre « le normal de ce
réseau » sur 24 h de trafic sain : sans cela, il produit 89 à 99 % de fausses
alarmes (mesuré en V6 et V7). Voir documentation/V7-documentation-complete.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import moteur_v7 as mv
from config import CHEMIN_MODELE, DOSSIER_ETATS

_MODELE = None


def charger(chemin: Path = CHEMIN_MODELE) -> mv.Modele:
    """Charge (une seule fois) le modèle V7 entraîné sur Kaggle."""
    global _MODELE
    if _MODELE is None:
        if not Path(chemin).exists():
            raise FileNotFoundError(
                f"Modèle introuvable : {chemin}. Téléchargez systeme_v7.joblib depuis les sorties Kaggle "
                "du notebook V7 et placez-le dans le dossier models/."
            )
        _MODELE = mv.charger_modele(chemin)
    return _MODELE


def lire(fichier, nom: str | None = None) -> mv.Flux:
    """Lit un journal Zeek (conn.log, CSV ou parquet) et calcule les variables V7."""
    return mv.preparer(mv.lire_journal(fichier, nom))


def calibrer(flux: mv.Flux, progression=None) -> mv.EtatCalibre:
    return mv.calibrer(charger(), flux, progression)


def analyser(flux: mv.Flux, etat: mv.EtatCalibre, progression=None):
    """Renvoie (tableau des flux avec scores et alertes, erreurs du VAE par variable)."""
    return mv.analyser(charger(), etat, flux, progression)


def predict(ligne, etat: mv.EtatCalibre, seuil: str = "1 %") -> dict:
    """Verdict d'un flux déjà analysé, au format du prototype : score entre 0 et 1, 0,5 au seuil."""
    s = etat.seuils[seuil]
    score = float(mv.niveau(ligne["score V7"], s))
    return {"score": score, "verdict": "anomalie" if ligne["score V7"] >= s else "normal"}


# --- États calibrés sauvegardés (un par réseau) -----------------------------
def _fichier_etat(nom: str) -> Path:
    propre = "".join(c if c.isalnum() or c in "-_" else "_" for c in nom.strip()) or "reseau"
    return DOSSIER_ETATS / f"{propre}.joblib"


def sauver_etat(etat: mv.EtatCalibre, nom: str) -> Path:
    DOSSIER_ETATS.mkdir(parents=True, exist_ok=True)
    p = _fichier_etat(nom)
    mv.sauver_etat(etat, p)
    return p


def etats_disponibles() -> list[str]:
    return sorted(p.stem for p in DOSSIER_ETATS.glob("*.joblib")) if DOSSIER_ETATS.exists() else []


def charger_etat(nom: str) -> mv.EtatCalibre:
    return mv.charger_etat(_fichier_etat(nom))


def metriques(resultats, seuil: str):
    return mv.metriques(resultats, seuil)


def niveau(scores, etat: mv.EtatCalibre, seuil: str):
    return mv.niveau(np.asarray(scores), etat.seuils[seuil])
