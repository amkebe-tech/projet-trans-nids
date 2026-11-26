# NIDS comportemental — moteur V7

Système de détection d'intrusion réseau **hybride, non supervisé et explicable**, appliqué aux
journaux **Zeek `conn.log`**. Projet transversal — École Polytechnique de Thiès,
Département Informatique et Télécommunications
(voir [`cahier_des_charges.md`](./cahier_des_charges.md)).

Le moteur n'apprend que du trafic normal : toute attaque est traitée comme jamais vue
(logique *zero-day*). Chaque alerte est expliquée par la contribution exacte de chaque détecteur.

## Architecture du moteur V7

| Détecteur | Signal |
|---|---|
| **VAE** (5 graines, ajusté au réseau) | Le flux est-il atypique ? |
| **Contexte machine** — Isolation Forest sur 1 min / 10 min / 1 h | La machine source se comporte-t-elle anormalement ? |
| **kNN local** | Le flux est-il loin des flux normaux de ce réseau ? |
| **Rareté** | La combinaison protocole / état / port est-elle rare ici ? |

Chaque score est converti en p-valeur par protocole ; le **score V7** est la moyenne des
−log₁₀ p. Une **calibration** sur 24 h de trafic normal du réseau surveillé (sans étiquettes)
fixe les seuils à 1 % ou 5 % de fausses alarmes.

## Performances

| Jeu de test (seuil 1 %) | Résultat |
|---|---|
| **UWF-ZeekData 2024** (extrait de démo) | **F1 0,979 · AUC 0,999** · précision 0,978 · rappel 0,981 · 0,7 % de fausses alarmes |
| UWF 2022 → 2025 (même réseau, 4 années) | 4 objectifs sur 4 chaque année |
| UNSW-NB15 (réseau jamais vu) | F1 0,897 · AUC 0,992 |
| IoT-23 (juge aveugle) | AUC 0,887 · rappel 0,13 (0,55 à 5 %) |

## Installation et lancement

Prérequis : **Python 3.10 à 3.13** (scikit-learn 1.6.1, exigé par le modèle, n'existe pas pour 3.14).

```bash
# 1. Récupérer le dépôt
git clone https://github.com/amkebe-tech/projet-trans-nids.git
cd projet-trans-nids

# 2. Installer les dépendances (PyTorch CPU inclus)
python -m venv .venv
.venv\Scripts\activate            # Linux / macOS : source .venv/bin/activate
pip install -r requirements.txt

# 3. Lancer l'application
streamlit run app/streamlit_app.py
```

Puis ouvrir http://localhost:8501.

**Démo en deux clics** : onglet *Calibration* → **Démo UWF 2024** (≈ 1 min), puis
onglet *Analyse* → **Démo UWF 2024 (attaques)** (≈ 30 s).

### Streamlit Community Cloud

Fichier principal `app/streamlit_app.py`, et **Python 3.12** dans *Advanced settings*.
Les chemins (modèle, démo, thème) sont relatifs au dépôt ; aucune configuration
supplémentaire n'est nécessaire.

### Docker

```bash
docker build -t nids-v7 .
docker run -p 8501:8501 nids-v7
```

## Données attendues

Journal Zeek `conn.log` (TSV natif), ou CSV / parquet avec les mêmes colonnes :
`ts`, `id.orig_h`, `id.resp_h`, `id.resp_p`, `proto`, `duration`, `orig_bytes`, `resp_bytes`,
`orig_pkts`, `resp_pkts`, `conn_state` (recommandées : `orig_ip_bytes`, `resp_ip_bytes`).
Les noms UWF (`src_ip_zeek`, `dest_ip_zeek`, `dest_port_zeek`) sont acceptés. Une colonne
`label` facultative active la comparaison avec la vérité terrain.

## Structure

```
app/
  streamlit_app.py     Interface SOC : calibration, analyse, détail d'un flux, modèle
  model_interface.py   Contrat du modèle : charger(), calibrer(), analyser(), predict()
  moteur_v7.py         Moteur V7 (code du notebook adapté au processeur)
  explainability.py    Décomposition exacte du score par détecteur
  config.py            Variables, chemins, seuils, objectifs
  assets/style.css     Compléments au thème
.streamlit/config.toml Thème sombre de l'application
models/systeme_v7.joblib   Modèle entraîné (16,9 Mo) ; models/etats/ : réseaux calibrés (non suivis)
donnees_demo/          Extraits Zeek UWF-ZeekData24 (calibration 24 h + analyse)
src/                   Conversion UWF parquet → conn.log, génération des démos
```

## Limites

- Réseau inconnu : bon classement mais rappel faible à 1 % ; balayages discrets et C&C les plus difficiles.
- La calibration suppose 24 h de trafic sans attaque ; sans elle, 89 à 99 % de fausses alarmes.
- Débit processeur ≈ 2 000 à 2 500 flux/s (≈ 40 000 flux/s sur GPU dans le notebook).
