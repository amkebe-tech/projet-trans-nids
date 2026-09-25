# NIDS Comportemental — Projet Transversal

Système de détection d'intrusion réseau (NIDS) par analyse comportementale,
basé sur l'apprentissage automatique. Projet transversal — École
Polytechnique de Thiès, Département Informatique et Télécommunications.

Voir [`cahier_des_charges.md`](./cahier_des_charges.md) pour le contexte
complet, les objectifs et les spécifications techniques du projet.

## État d'avancement

⚠️ **Ce dépôt contient actuellement le squelette du prototype de
démonstration, pas le projet final.**

- Les modèles réels (VAE, Pure-LSTM Autoencoder) sont en cours
  d'entraînement et d'évaluation sur Google Colab (notebook séparé, à
  ajouter dans `notebooks/`).
- Le module `app/model_interface.py` est un **placeholder** : il simule un
  score d'anomalie de façon aléatoire, mais respecte le format d'entrée/sortie
  qu'aura le vrai modèle. Il sera remplacé par le vrai modèle sans que
  l'interface Streamlit n'ait besoin d'être modifiée.
- Le module `app/explainability.py` est également un **placeholder** en
  attendant l'intégration de SHAP/LIME/TRUSTEE.

## Structure du dépôt

```
nids-projet/
├── app/
│   ├── streamlit_app.py       # Interface Streamlit principale
│   ├── model_interface.py     # Contrat du modèle : predict() (placeholder → vrai modèle)
│   ├── explainability.py      # Génération d'explication : explain() (placeholder → TRUSTEE)
│   └── config.py              # Configuration centrale (FEATURES, taille de fenêtre, seuil)
├── models/                    # Poids du modèle entraîné (.pth) — vide pour l'instant
├── src/                       # Pipeline de données (à migrer depuis le notebook Colab)
├── notebooks/                 # Notebooks Colab d'expérimentation
├── Dockerfile
├── requirements.txt
└── README.md
```

**Point important** : le nombre et les noms des features utilisées par le
modèle ne sont PAS écrits en dur dans le code. Ils sont centralisés dans
`app/config.py` (liste `FEATURES`). Actuellement 5 features
(`flow_duration`, `fwd_pkts`, `bwd_pkts`, `fwd_bytes`, `bwd_bytes`), mais ce
nombre pourra évoluer (15-20 features envisagées) sans toucher au reste du
code, qui utilise systématiquement `FEATURES` et `len(FEATURES)`.

## Lancer le prototype en local

```bash
python3 -m venv .venv
source .venv/bin/activate          # sous Windows : .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app/streamlit_app.py
```

L'application est ensuite accessible sur http://localhost:8501.

Elle accepte un fichier CSV déjà nettoyé (colonnes = celles définies dans
`app/config.py`), ou permet de générer des données factices pour tester
l'interface sans fichier.

## Lancer le prototype avec Docker

```bash
docker build -t nids-prototype .
docker run -p 8501:8501 nids-prototype
```

L'application est accessible sur http://localhost:8501.

## Prochaines étapes

- Brancher le vrai modèle (VAE / LSTM) dans `app/model_interface.py` une
  fois l'entraînement terminé sur Colab.
- Brancher SHAP/LIME/TRUSTEE dans `app/explainability.py`.
- Migrer le pipeline de nettoyage de données du notebook vers `src/`.
- Ajouter une API FastAPI pour la démonstration d'intégration SDN.
