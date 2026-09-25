**CAHIER DES CHARGES\
**

**Système de Détection d\'Intrusion Réseau\
par Analyse Comportementale\
Basé sur l\'Apprentissage Automatique\
**

*École Polytechnique de Thiès\
Département Informatique et Télécommunications\
*

Membres : Demba Guissé, Aichatou Mamadou Kébé, Ndeye Fadieye Touré\
Encadreurs : Michel Seck, Abdoul Aziz Ciss, Ndeye Fatou Ngom

**I. Contexte et Objectifs**

**1.1 Contexte et Justification**

Les experts en cybersécurité ont longtemps fondé leurs défenses sur la
reconnaissance de signatures d\'attaques connues. Cette approche, bien
qu\'efficace contre les menaces répertoriées, montre ses limites
structurelles face aux attaques dites zero-day, qui exploitent des
vulnérabilités inédites pour lesquelles aucune signature n\'existe
encore. L\'échec des NIDS traditionnels dans ce contexte nous a conduits
à concevoir une alternative : un NIDS comportemental fondé sur
l\'analyse d\'anomalies par apprentissage automatique.

L\'analyse comportementale repose sur un principe différent : modéliser
le trafic réseau normal, puis détecter toute déviation significative par
rapport à ce profil de référence. Cette approche, combinée aux
architectures modernes de deep learning, permet d\'identifier des
menaces inconnues sans connaissance préalable de leur signature.

Par ailleurs, les environnements réseau réels sont dynamiques : les
usages évoluent, les comportements changent, et les attaquants
s\'adaptent. Ce phénomène de dérive conceptuelle (concept drift) dégrade
progressivement les performances des modèles entraînés sur des données
passées et constitue l\'un des défis centraux de ce projet.

Enfin, un système de détection ne peut être pleinement opérationnel que
si ses décisions sont compréhensibles par les administrateurs réseau.
L\'intégration d\'une couche d\'explicabilité (XAI) est donc une
exigence fonctionnelle à part entière, et non un ajout optionnel.

**1.2 Objectif Général**

Concevoir, implémenter et évaluer un prototype de NIDS comportemental
interprétable, capable de détecter des flux malveillants (y compris des
attaques inconnues) en environnement réseau dynamique, et de résister à
l\'évolution naturelle du trafic dans le temps.

**1.3 Objectifs Spécifiques**

**1) Nettoyage et préparation des données**

-   Analyser la structure des trois datasets de référence (CIC-IDS2017,
    CSE-CIC-IDS2018, UNSW-NB15) pour identifier les variables clés, les
    déséquilibres de classes et les anomalies de collecte.

-   Développer des scripts automatisés de nettoyage : gestion des
    valeurs manquantes, suppression des doublons, traitement des valeurs
    infinies, normalisation et encodage.

-   Convertir les fichiers bruts au format compressé .parquet pour
    optimiser les temps de chargement et réduire l\'empreinte disque.

-   Segmenter les jeux de données pour extraire les flux 100% légitimes,
    indispensables à l\'entraînement non supervisé du VAE.

**2) Modélisation et apprentissage**

-   Implémenter et évaluer deux modèles supervisés classiques (Random
    Forest et XGBoost) servant de baselines et fixant les scores
    minimaux de référence (Précision, Rappel, F1-Score).

-   Concevoir et évaluer un Autoencodeur Variationnel (VAE) capable
    d\'apprendre la structure du trafic sain et de calculer un score
    d\'anomalie basé sur l\'erreur de reconstruction.

-   Implémenter des architectures séquentielles (LSTM et Transformer
    temporel) pour analyser les flux réseau sous forme de séries
    chronologiques et détecter les attaques diffuses ou lentes.

-   Conduire une évaluation cross-dataset : entraîner sur un dataset
    (ex. CIC-IDS2017) et tester sur un autre (ex. UNSW-NB15) pour
    mesurer la capacité de généralisation réelle des modèles.

**3) Explicabilité et robustesse**

-   Implémenter SHAP et LIME pour identifier et cartographier les
    features réseau qui influencent le plus le déclenchement d\'une
    alerte.

-   Étudier et adapter le framework TRUSTEE au VAE via un wrapper
    personnalisé convertissant les scores continus d\'anomalie en
    classes binaires, afin de produire des règles logiques SI...ALORS
    compréhensibles.

-   Comparer les deux approches d\'explicabilité (SHAP/LIME vs TRUSTEE)
    en termes de fidélité, lisibilité et temps de génération.

-   Évaluer la robustesse face à la dérive conceptuelle : découper les
    datasets selon l\'axe temporel, mesurer la dégradation des
    performances, et explorez des stratégies d\'adaptation (fenêtres
    glissantes, réentraînement incrémental, détecteurs de drift
    DDM/ADWIN).

**4) Valorisation et déploiement**

-   Développer un prototype fonctionnel (interface Streamlit ou CLI)
    capable de charger un fichier de flux réseau, afficher le verdict du
    NIDS en temps réel et générer dynamiquement l\'explication associée
    à chaque alerte.

-   Formuler des recommandations techniques pour l\'intégration en
    environnement réel, notamment les contraintes de légèreté pour le
    Edge Computing (quantification, format ONNX) et l\'interconnexion
    avec un contrôleur SDN pour une réponse automatisée aux intrusions.

**II. Etat de l'Art**

Le domaine de la détection d\'intrusions (NIDS) connaît une mutation
rapide, passant de systèmes basés sur des signatures statiques à des
architectures d\'apprentissage profond (Deep Learning) complexes. Il est
marqué par une transition technologique majeure : le passage de modèles
\"boîtes noires\" vers des systèmes explicables et adaptatifs.
Cependant, cette évolution se heurte à des verrous persistants que notre
projet ambitionne de lever. Cette section synthétise l\'état des
connaissances et justifie notre positionnement face aux défis actuels
identifiés dans la littérature 2024-2026.

**2.1. Cartographie des approches :**

La littérature récente classe les travaux en trois piliers, chacun
présentant des angles morts que nous adressons :

-   **- Les approches temporelles (LSTM/Transformer) :** Luay et
    al. (2024) et Andrade et al. (2026) démontrent que l\'aspect
    séquentiel est crucial pour capturer les attaques complexes et
    étalées dans le temps. Leurs limites : D\'une part, ils traitent
    généralement ces modèles comme des classifieurs supervisés purs, ce
    qui les rend aveugles face aux menaces zero-day. D\'autre part, les
    architectures comme le LSTM souffrent d\'un goulot d\'étranglement
    computationnel majeur : leur coût de calcul explose lorsque le
    nombre de variables en entrée est trop élevé (haute
    dimensionnalité).

-   **- Les approches graphiques (GNN/GAT) :** Wang et al. (2025)
    proposent des graphes de similarité comportementale très
    performants. Leur limite : La construction de graphes est
    extrêmement coûteuse en temps de calcul, rendant le déploiement sur
    des nœuds de bordure (Edge) ou en temps réel quasi impossible sans
    dégrader la précision par échantillonnage.

-   **- Les approches par reconstruction (Autoencodeurs/VAE) :** Prabowo
    et al. (2025) prouvent que l\'apprentissage non supervisé par
    reconstruction est le plus robuste face à la \"pollution\" des
    données. Leur limite : Ils manquent d\'un mécanisme de rétroaction
    permettant de traduire une reconstruction erronée en explication
    logique pour l\'analyste.

**1.2. Analyse critique des Datasets : Le choix de la rigueur**

L\'état de l\'art actuel souffre d\'un manque de standardisation. Si
CICIoT2023 est très médiatisé, nous avons choisi de l\'écarter pour
rester fidèles au contexte d\'un réseau d\'entreprise classique. Notre
stratégie de validation repose sur une dualité :

**Référence Historique (CSE-CIC-IDS2018) :** Malgré les critiques
académiques (fuites de données, biais), ce dataset demeure le \"langage
commun\" de la communauté cybersécurité. En l\'utilisant, nous
garantissons que nos performances (Recall, Precision, F1-Score) sont
comparables aux benchmarks internationaux établis par les travaux de
2025-2026.

**Validation de Généralisation (UWF-ZeekData24) :** C\'est ici que
réside notre exigence de modernité. En intégrant UWF-ZeekData24, un
dataset contemporain, nous testons la capacité de notre projet à
généraliser au-delà des biais des datasets de 2018. Cette approche
\"multi-dataset\" répond directement aux recommandations d\'El Mahdaouy
et al. (2026) sur la rigueur de l\'évaluation croisée.

**1.3. Notre positionnement**

Contrairement aux études analysées, notre projet ne cherche pas la
performance isolée, mais l\'intégration opérationnelle. Nous répondons
aux verrous identifiés par trois ruptures :

**Rupture par l\'explicabilité dynamique :** Alors que les modèles GAT
ou CNN se contentent d\'alerter, nous intégrons un workflow
SHAP/LIME/TRUSTEE qui convertit le signal d\'anomalie de l\'espace
latent en variables réseau brutes. Nous transformons une \"alerte
mathématique\" en une \"décision métier\".

**Rupture par la gestion de la dérive (Drift) :** Les modèles actuels
(ex: Waghmode et al., 2025) sont figés. Nous intégrons une surveillance
du concept drift (ADWIN) qui permet de monitorer si la distribution du
trafic change, déclenchant une alerte de \"dégradation de modèle\" bien
avant que la précision ne s\'effondre.

**Rupture par l\'architecture hybride (Autoencodeur + LSTM) :** Nous
combinons la robustesse au bruit de l\'Autoencodeur (pour isoler le
comportement sain) avec la mémoire temporelle du LSTM (pour détecter les
attaques lentes). C\'est précisément pour lever ce verrou que notre
projet n\'entraîne pas le LSTM sur le dataset brut, mais utilise
d\'abord un Autoencodeur pour compresser les données et projeter le
trafic dans un espace latent réduit. Le LSTM n\'analyse ainsi que les
caractéristiques essentielles compressées, optimisant drastiquement le
temps d\'exécution et rendant le système compatible avec les exigences
du temps réel. Cette cascade permet de traiter le trafic en flux continu
tout en conservant une empreinte computationnelle légère pour le
Edge/SDN.

  -------------------------------------------------------------------------------
  Travaux        Méthode        Focus          Faiblesse        Notre plus-value
                                               majeure          
  -------------- -------------- -------------- ---------------- -----------------
  Wang (2025)    GNN/GAT        Topologie      Boîte noire,     Explicabilité +
                                               lourd            Temps réel

  Prabowo (2025) Autoencodeur   Robustesse     Pas de feedback  Rétro-mapping
                                               humain           sémantique

  Andrade (2026) LSTM           Précision IoT  Pas de           Bench
                                               généralisation   multi-dataset +
                                                                Drift

  Luay (2024)    Temporalité    Flux           Pas de zero-day  Hybride VAE-LSTM
                                séquentiels                     explicable
  -------------------------------------------------------------------------------

**III. Besoins Non Fonctionnels**

**2.1 Performance statistique des modèles**

  -----------------------------------------------------------------------
  Métrique                Objectif cible          Justification
                                                  cybersécurité
  ----------------------- ----------------------- -----------------------
  Rappel (Recall)         \> 95%                  Minimiser les faux
                                                  négatifs : une attaque
                                                  manquée est un risque
                                                  inacceptable

  Précision               \> 90%                  Éviter la fatigue des
                                                  alertes en limitant les
                                                  faux positifs pour
                                                  l\'administrateur

  F1-Score                \> 93%                  Assurer la robustesse
                                                  globale face au
                                                  déséquilibre des
                                                  classes dans les
                                                  datasets

  AUC-ROC                 \> 0.97                 Mesurer la capacité de
                                                  discrimination
                                                  indépendamment du seuil
                                                  de décision
  -----------------------------------------------------------------------

**2.2 Performances système et contraintes d\'exploitation**

  -----------------------------------------------------------------------
  Axe                     Exigence technique      Objectif opérationnel
  ----------------------- ----------------------- -----------------------
  Temps réel (inférence)  \< 10 ms par flux       Analyser le trafic au
                                                  vol sans créer de
                                                  goulot d\'étranglement
                                                  réseau

  Fidélité XAI (TRUSTEE)  \> 90%                  L\'arbre de décision
                                                  doit refléter
                                                  fidèlement la logique
                                                  interne du VAE

  Légèreté XAI            \< 50 ms par            Générer la règle
                          explication             SI...ALORS
                                                  quasi-instantanément
                                                  lors d\'une alerte

  Scalabilité             Traitement par          Ingérer les millions de
                          mini-batchs             lignes des datasets
                                                  sans saturer la mémoire
                                                  RAM

  Robustesse au drift     Dégradation \< 5% sur 6 Garantir la fiabilité
                          mois simulés            du système dans un
                                                  environnement réseau
                                                  évolutif
  -----------------------------------------------------------------------

**IV. Spécifications Techniques**

**3.1 Données utilisées**

  -----------------------------------------------------------------------
  Dataset                 Année                   Caractéristiques
                                                  principales
  ----------------------- ----------------------- -----------------------
  UWF-ZeekData24          2024                    Il s\'agit de captures
                                                  de trafic réseau
                                                  réelles

  CSE-CIC-IDS2018         2018                    Scénarios étendus,
                                                  grande volumétrie (\>10
                                                  Go), environnement AWS
  -----------------------------------------------------------------------

L\'évaluation cross-dataset suivra le protocole : entraînement sur
CSE-CIC-IDS2018, validation test final sur UWF-ZeekData24---
garantissant une évaluation de généralisation rigoureuse, sans data
leakage inter-datasets.

**3.2 Stack logicielle**

  -----------------------------------------------------------------------
  Catégorie                           Outils
  ----------------------------------- -----------------------------------
  Langage                             Python 3.10+

  Data Pipeline                       Pandas, NumPy, PyArrow (format
                                      .parquet)

  ML classique                        Scikit-learn, XGBoost, LightGBM

  Deep Learning                       PyTorch (ou TensorFlow/Keras)

  Explicabilité                       SHAP, LIME, TRUSTEE (wrapper
                                      personnalisé VAE)

  Détection de drift                  River, Alibi-Detect (DDM, ADWIN)

  Prototype                           Streamlit ou CLI Python

  Déploiement                         FastAPI, Docker, ONNX (Edge)

  Versionnement                       Git / GitHub

  Environnement                       Google Colab Pro, Kaggle Kernels
  -----------------------------------------------------------------------

**3.3 Contraintes matérielles**

-   **Stockage :** Les datasets bruts (notamment CSE-CIC-IDS2018)
    dépassent plusieurs dizaines de Go en .csv. Conversion obligatoire
    en .parquet (ratio de compression x5). Espace minimal requis : 50
    Go.

-   **Calcul GPU :** L\'entraînement des architectures séquentielles
    (LSTM, Transformers) sur ces volumes nécessite un environnement
    CUDA. Minimum requis : GPU NVIDIA T4 avec 16 Go de VRAM (Google
    Colab Pro ou Kaggle).

-   **RAM :** Traitement par mini-batchs obligatoire pour éviter la
    saturation mémoire lors de l\'ingestion des grands datasets.

**3.4 Reproductibilité**

-   Hébergement sur un dépôt GitHub structuré : séparation stricte entre
    scripts de prétraitement, architectures de modèles et notebooks
    d\'expérimentation.

-   Fixation systématique des graines aléatoires (random seeds) sur tous
    les frameworks (NumPy, PyTorch, XGBoost, Scikit-learn) pour garantir
    la reproductibilité exacte des métriques.

-   Fichier requirements.txt versionné et README.md documentant la
    procédure de reproduction complète.

**V. Livrables Attendus**

  ------------------------------------------------------------------------
  \#                      Livrable                Description
  ----------------------- ----------------------- ------------------------
  1                       Répertoire de code      Dépôt GitHub structuré
                          source                  et documenté : pipeline
                                                  de données, moteur
                                                  d\'apprentissage, module
                                                  XAI, scripts de
                                                  déploiement

  2                       Notebook                Notebook Jupyter
                          d\'expérimentation      reproductible couvrant
                                                  l\'intégralité du
                                                  pipeline : EDA,
                                                  prétraitement,
                                                  entraînement,
                                                  benchmarking,
                                                  visualisations
                                                  SHAP/TRUSTEE

  3                       Modèles sérialisés      Fichiers de poids des
                                                  meilleurs modèles
                                                  (.pt/.pth pour PyTorch,
                                                  .pkl/.json pour
                                                  Scikit-learn/XGBoost),
                                                  rechargeables sans
                                                  réentraînement

  4                       Prototype fonctionnel   Application Streamlit ou
                                                  CLI chargeant un fichier
                                                  de flux réseau,
                                                  affichant le verdict
                                                  NIDS et générant
                                                  l\'explication TRUSTEE
                                                  en cas d\'alerte

  5                       Rapport d\'analyse      Document académique
                          comparative             complet : état de
                                                  l\'art, choix
                                                  méthodologiques, tableau
                                                  de benchmarking, analyse
                                                  du drift,
                                                  recommandations Edge/SDN

  6                       Guide d\'installation   README.md pas-à-pas :
                                                  configuration de
                                                  l\'environnement
                                                  virtuel, installation
                                                  des dépendances,
                                                  commandes de
                                                  reproduction des
                                                  résultats
  ------------------------------------------------------------------------

**VI. Planning Prévisionnel**

  -----------------------------------------------------------------------
  Phase                   Activités               Durée
  ----------------------- ----------------------- -----------------------
  Phase 1                 Étude bibliographique,  2 semaines
                          état de l\'art, prise   
                          en main des datasets    

  Phase 2                 Prétraitement, EDA,     2 semaines
                          pipeline de données,    
                          conversion .parquet,    
                          isolation du trafic     
                          sain                    

  Phase 3                 Implémentation et       2 semaines
                          évaluation des          
                          baselines (Random       
                          Forest, XGBoost)        

  Phase 4                 Implémentation des      3 semaines
                          modèles DL : VAE, DNN,  
                          LSTM, Transformer       
                          temporel                

  Phase 5                 Intégration XAI (SHAP,  2 semaines
                          LIME, TRUSTEE),         
                          évaluation              
                          cross-dataset, analyse  
                          du drift                

  Phase 6                 Développement du        2 semaines
                          prototype,              
                          conteneurisation        
                          Docker, rédaction du    
                          rapport                 

  Phase 7                 Tests finaux,           1 semaine
                          révisions, préparation  
                          de la soutenance        
  -----------------------------------------------------------------------

Durée totale estimée : 14 semaines
