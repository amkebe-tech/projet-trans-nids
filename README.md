# NIDS Comportemental — Projet Transversal

Système de détection d'intrusion réseau (NIDS) par analyse comportementale,
basé sur l'apprentissage automatique. Projet transversal — École
Polytechnique de Thiès, Département Informatique et Télécommunications.

Voir [`cahier_des_charges.md`](./cahier_des_charges.md) pour le contexte
complet, les objectifs et les spécifications techniques du projet.

## État d'avancement

L'application intègre le **moteur V7**, le meilleur système obtenu dans le projet
(notebook `NIDS-v7-kaggle.ipynb`, entraîné sur Kaggle).

- **Non supervisé** : il n'apprend que du trafic normal. Toute attaque est donc,
  pour lui, une attaque jamais vue (logique *zero-day*).
- **Quatre détecteurs** :
  - un VAE (5 graines) qui juge chaque flux ;
  - un Isolation Forest sur le **comportement de la machine source**, sur 1 min, 10 min et 1 h ;
  - un kNN qui mesure la distance aux flux normaux du réseau ;
  - un score de rareté de la combinaison protocole, état et port.

  Leurs scores sont convertis en p-valeurs par protocole, puis moyennés.
- **Calibration sur le réseau surveillé** : 24 h de trafic normal, sans aucune
  étiquette d'attaque. Elle fixe les seuils à 1 % ou 5 % de fausses alarmes.
- **Explication réelle de chaque alerte** :
  - la contribution de chaque détecteur ;
  - les variables du flux que le VAE reconstruit mal ;
  - le comportement de la machine source.

**Résultats mesurés**

| Test | Résultat |
|---|---|
| UWF 2022 → 2025 (même réseau, 4 années) | 4 objectifs sur 4 chaque année, au seuil de 1 % |
| UNSW-NB15 (réseau jamais vu) | F1 0,897, AUC 0,992 |
| IoT-23 (juge aveugle) | AUC 0,887, mais rappel de 0,13 à 1 % et de 0,55 à 5 % |

La documentation complète du modèle se trouve dans le dossier `documentation/`
du projet (`V7-documentation-complete.md`).

## Structure du dépôt

```
├── app/
│   ├── streamlit_app.py       # Interface : ① calibration, ② analyse, ③ détail d'un flux, ℹ️ modèle
│   ├── model_interface.py     # Contrat du modèle : charger(), calibrer(), analyser(), predict()
│   ├── moteur_v7.py           # Moteur V7 : code du notebook adapté au processeur
│   ├── explainability.py      # Explication : décomposition exacte du score par détecteur
│   └── config.py              # Configuration centrale (variables, chemins, seuils, objectifs)
├── models/
│   ├── systeme_v7.joblib      # Modèle entraîné (export du notebook V7, 16,9 Mo)
│   └── etats/                 # États calibrés, un par réseau (créés par l'application)
├── donnees_demo/              # Journaux Zeek de démonstration (UWF-ZeekData24)
├── src/
│   ├── uwf24_vers_zeek.py     # Lecture des parquet UWF et écriture au format conn.log
│   └── creer_demo.py          # Fabrique les journaux de démonstration
├── Dockerfile
├── requirements.txt
└── README.md
```

## Données attendues

Un journal **Zeek `conn.log`** : le format TSV natif de Zeek, ou un CSV/parquet
avec les mêmes colonnes.

- **Colonnes obligatoires** : `ts`, `id.orig_h`, `id.resp_h`, `id.resp_p`,
  `proto`, `duration`, `orig_bytes`, `resp_bytes`, `orig_pkts`, `resp_pkts`,
  `conn_state`.
- **Colonnes recommandées** : `orig_ip_bytes`, `resp_ip_bytes`.
- **Les noms UWF sont aussi acceptés** : `src_ip_zeek`, `dest_ip_zeek`, `dest_port_zeek`.
- **Colonne `label` facultative** (`benign` / nom de l'attaque). Si elle est
  présente, l'application affiche aussi la précision, le rappel, le F1 et l'AUC.

## Lancer l'application en local

Il faut **Python 3.10 à 3.13** : scikit-learn 1.6.1, exigé par le modèle,
n'existe pas pour Python 3.14.

```bash
python -m venv .venv
.venv\Scripts\activate             # sous Linux / macOS : source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

streamlit run app/streamlit_app.py
```

L'application est ensuite accessible sur http://localhost:8501.

**Démonstration en deux clics**
1. Onglet ① : *Démonstration : 24 h normales d'UWF 2024*. La calibration prend environ 1 min.
2. Onglet ② : *Démonstration : trafic UWF 2024 avec attaques*. L'analyse prend environ 30 s.

Sur cet extrait, au seuil de 1 %, on obtient P 0,978, R 0,981, F1 0,979,
AUC 0,999 et 0,7 % de fausses alarmes.

Les journaux de démonstration se régénèrent avec
`python src/creer_demo.py <dossier des parquet UWF-ZeekData24>`.

## Lancer l'application avec Docker

```bash
docker build -t nids-v7 .
docker run -p 8501:8501 nids-v7
```

## Limites à connaître

- **Réseau totalement inconnu** : le classement est bon, mais le rappel est
  faible au seuil de 1 %. Les balayages discrets et le C&C sont les plus difficiles.
- **Calibration** : elle suppose 24 h de trafic **sans attaque**.
- **Mode « à froid » absent** : l'application ne propose pas d'analyser sans
  calibration, qui donne 89 à 99 % de fausses alarmes sur un réseau nouveau.
- **Débit** : sur processeur, environ 2 000 à 2 500 flux/s. Le notebook atteint
  environ 40 000 flux/s sur GPU.

## Prochaines étapes

- Ajouter une API FastAPI pour la démonstration d'intégration SDN.
- Analyse en continu d'un `conn.log` Zeek en cours d'écriture.
