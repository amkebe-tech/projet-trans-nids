"""
Application Streamlit - Prototype de démonstration du NIDS.

Ce fichier ne contient QUE de la logique d'interface (affichage, upload de
fichier, mise en forme des résultats). Il ne contient AUCUNE logique de
modèle ou d'explicabilité : tout ça est délégué à model_interface.py et
explainability.py.

Pourquoi cette séparation ?
----------------------------
Le vrai modèle (VAE/LSTM) et le vrai module d'explicabilité (SHAP/TRUSTEE)
ne sont pas encore prêts. En gardant streamlit_app.py totalement indépendant
de LEUR implémentation interne (il ne connaît que les fonctions predict() et
explain(), pas comment elles calculent leurs résultats), on pourra brancher
les vrais modules plus tard sans toucher à une seule ligne de ce fichier.

À propos du style visuel :
----------------------------
Les couleurs globales (fond, texte, couleur d'accent) viennent de
.streamlit/config.toml (méthode recommandée par Streamlit pour un thème
cohérent). Le fichier app/assets/style.css ne complète ce thème que pour ce
qu'il ne permet pas nativement : police monospace ciblée, carte d'alerte à
bordure colorée, badges, bandeau d'avertissement intégré au thème sombre.

Comment lancer l'app :
    streamlit run app/streamlit_app.py
"""

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Imports "voisins" (pas "from app.config import ...") : Streamlit exécute
# le script en ajoutant le dossier app/ à sys.path, pas la racine du dépôt.
from config import FEATURES, WINDOW_SIZE, SEUIL_ANOMALIE
from model_interface import predict
from explainability import explain

# Couleurs sémantiques strictes, réutilisées partout (carte, jauge,
# graphique) : vert = normal, rouge = anomalie, jamais l'inverse, jamais
# d'autre couleur pour ces deux états. Doit rester synchronisé avec les
# variables CSS du même nom dans app/assets/style.css.
COULEUR_NORMAL = "#22c55e"
COULEUR_ANOMALIE = "#ef4444"
COULEUR_TEXTE = "#e6edf3"
COULEUR_TEXTE_ATTENUE = "#8b98ac"


def _charger_css() -> str:
    """Charge le CSS custom (app/assets/style.css) sous forme de texte."""
    chemin_css = Path(__file__).parent / "assets" / "style.css"
    return chemin_css.read_text(encoding="utf-8")


def _construire_jauge(score: float, seuil: float) -> go.Figure:
    """
    Construit une jauge semi-circulaire (Plotly) pour visualiser le score
    d'anomalie : zone verte sous le seuil, zone rouge au-dessus, avec un
    repère sur le seuil lui-même.
    """
    couleur_aiguille = COULEUR_ANOMALIE if score > seuil else COULEUR_NORMAL

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={
                "valueformat": ".3f",
                "font": {"family": "JetBrains Mono, monospace", "color": COULEUR_TEXTE, "size": 34},
            },
            gauge={
                "axis": {
                    "range": [0, 1],
                    "tickcolor": COULEUR_TEXTE_ATTENUE,
                    "tickfont": {"color": COULEUR_TEXTE_ATTENUE, "size": 10},
                },
                "bar": {"color": couleur_aiguille, "thickness": 0.35},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                # Les deux zones colorées reprennent EXACTEMENT les mêmes
                # couleurs sémantiques que la carte de verdict et le
                # graphique d'explication.
                "steps": [
                    {"range": [0, seuil], "color": "rgba(34, 197, 94, 0.18)"},
                    {"range": [seuil, 1], "color": "rgba(239, 68, 68, 0.18)"},
                ],
                "threshold": {
                    "line": {"color": COULEUR_TEXTE, "width": 2},
                    "thickness": 0.85,
                    "value": seuil,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COULEUR_TEXTE},
        height=220,
        margin=dict(l=30, r=30, t=30, b=10),
    )
    return fig


st.set_page_config(page_title="NIDS Comportemental - Prototype", page_icon="📡", layout="centered")

st.markdown(f"<style>{_charger_css()}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# En-tête
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="nids-header">
        <div class="nids-header__title">NIDS — Console de Détection Comportementale</div>
        <div class="nids-header__subtitle">SOC Preview · Behavioral Anomaly Detection Engine</div>
        <div class="nids-header__divider"></div>
        <div class="nids-badge-row">
            <span class="nids-badge">FEATURES: {len(FEATURES)}</span>
            <span class="nids-badge">WINDOW: {WINDOW_SIZE} steps</span>
            <span class="nids-badge">THRESHOLD: {SEUIL_ANOMALIE:.2f}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="nids-banner">
        <div class="nids-banner__tag">⚠ Mode placeholder</div>
        <div class="nids-banner__text">
            Le modèle de détection (VAE/LSTM) et le module d'explicabilité
            (SHAP/TRUSTEE) ne sont pas encore entraînés. Les résultats
            affichés ici sont <strong>simulés</strong> et servent uniquement
            à valider le fonctionnement de l'interface.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "Cette application charge un fichier de flux réseau déjà nettoyé "
    "(colonnes = features attendues par le modèle), le découpe en fenêtres "
    "temporelles, puis affiche le verdict du NIDS et une explication pour "
    "chaque fenêtre analysée."
)

# ---------------------------------------------------------------------------
# 1. Chargement des données
# ---------------------------------------------------------------------------
st.header("1. Charger des données")

# Streamlit ré-exécute TOUT le script à chaque interaction (clic sur un
# bouton, déplacement d'un slider...). Une simple variable Python locale
# (comme "df = ...") est donc recréée à zéro, et perdue, à chaque rerun.
# st.session_state est un dictionnaire spécial qui, lui, survit d'un rerun à
# l'autre pour un même utilisateur : c'est là qu'on doit stocker tout ce qui
# doit "persister" (données chargées/générées, sélection en cours, etc.).
if "df" not in st.session_state:
    st.session_state.df = None
if "fichier_traite_id" not in st.session_state:
    st.session_state.fichier_traite_id = None

col_upload, col_demo = st.columns(2)

with col_upload:
    fichier = st.file_uploader(
        "Fichier CSV de flux réseau (déjà nettoyé)",
        type=["csv"],
        help=f"Le CSV doit contenir au moins les colonnes : {', '.join(FEATURES)}",
    )

with col_demo:
    st.write("Pas de fichier sous la main ?")
    generer_demo = st.button("Générer des données factices")

# On construit le DataFrame à analyser soit depuis le fichier uploadé, soit
# depuis des données factices générées à la volée. Le résultat est écrit
# dans st.session_state.df (pas dans une simple variable locale) pour rester
# disponible lors des reruns suivants (ex: clic sur "Analyser cette fenêtre").
if generer_demo:
    # Données purement aléatoires, juste pour pouvoir tester l'interface de
    # bout en bout sans avoir de vrai fichier CICFlowMeter sous la main.
    n_lignes = WINDOW_SIZE * 5  # de quoi faire 5 fenêtres
    donnees_aleatoires = np.random.default_rng().uniform(
        low=0.0, high=1000.0, size=(n_lignes, len(FEATURES))
    )
    st.session_state.df = pd.DataFrame(donnees_aleatoires, columns=FEATURES)
    st.session_state.fichier_traite_id = None  # on n'est plus sur un fichier uploadé
    st.success(f"{n_lignes} lignes factices générées ({n_lignes // WINDOW_SIZE} fenêtres).")

elif fichier is not None and fichier.file_id != st.session_state.fichier_traite_id:
    # Le composant d'upload garde le fichier "présent" à chaque rerun tant
    # qu'on ne le retire pas. Sans cette comparaison sur file_id, ce bloc
    # s'exécuterait à CHAQUE interaction (même sans changement de fichier)
    # et écraserait par exemple des données de démo générées après coup.
    df_uploade = pd.read_csv(fichier)

    # On vérifie que toutes les colonnes attendues (définies dans
    # config.FEATURES) sont bien présentes dans le fichier chargé.
    colonnes_manquantes = [f for f in FEATURES if f not in df_uploade.columns]
    if colonnes_manquantes:
        st.error(
            f"Colonnes manquantes dans le CSV : {colonnes_manquantes}. "
            f"Colonnes attendues : {FEATURES}"
        )
    else:
        st.session_state.df = df_uploade
        st.session_state.fichier_traite_id = fichier.file_id
        st.success(f"Fichier chargé : {len(df_uploade)} lignes.")

df = st.session_state.df

if df is not None:
    st.dataframe(df[FEATURES].head(20))

    # ---------------------------------------------------------------------
    # 2. Sélection de la fenêtre à analyser
    # ---------------------------------------------------------------------
    st.header("2. Analyser une fenêtre de trafic")

    nb_fenetres = len(df) // WINDOW_SIZE

    if nb_fenetres == 0:
        st.error(
            f"Pas assez de lignes pour former une fenêtre complète "
            f"(il faut au moins {WINDOW_SIZE} lignes, {len(df)} fournies)."
        )
    else:
        index_fenetre = st.slider(
            "Choisir la fenêtre à analyser",
            min_value=0,
            max_value=nb_fenetres - 1,
            value=0,
            help=f"Chaque fenêtre couvre {WINDOW_SIZE} lignes consécutives du fichier.",
            # Une "key" explicite indique à Streamlit de conserver la valeur
            # de ce slider dans st.session_state["index_fenetre"] entre les
            # reruns, plutôt que de la redéfinir à "value=0" à chaque fois.
            key="index_fenetre",
        )

        debut = index_fenetre * WINDOW_SIZE
        fin = debut + WINDOW_SIZE
        fenetre_df = df[FEATURES].iloc[debut:fin]

        st.caption(f"Lignes {debut} à {fin - 1} du fichier.")
        st.dataframe(fenetre_df)

        if st.button("🔍 Analyser cette fenêtre", type="primary"):
            fenetre = fenetre_df.to_numpy(dtype=float)

            resultat = predict(fenetre)
            explication = explain(fenetre, resultat)

            # -----------------------------------------------------------
            # 3. Affichage du verdict
            # -----------------------------------------------------------
            st.header("3. Résultat")

            score = resultat["score"]
            verdict = resultat["verdict"]
            classe_verdict = "normal" if verdict == "normal" else "anomalie"
            icone_verdict = "✅" if verdict == "normal" else "🚨"

            col_card, col_gauge = st.columns(2)

            with col_card:
                st.markdown(
                    f"""
                    <div class="alert-card alert-card--{classe_verdict}">
                        <div class="alert-card__label">Verdict NIDS</div>
                        <div class="alert-card__verdict alert-card__verdict--{classe_verdict}">
                            {icone_verdict} {verdict.upper()}
                        </div>
                        <div class="alert-card__score">{score:.3f}</div>
                        <div class="alert-card__score-label">
                            SCORE D'ANOMALIE (seuil : {SEUIL_ANOMALIE:.2f})
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_gauge:
                st.plotly_chart(
                    _construire_jauge(score, SEUIL_ANOMALIE),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

            # -----------------------------------------------------------
            # 4. Affichage de l'explication
            # -----------------------------------------------------------
            st.header("4. Explication")
            st.caption(f"Méthode : {explication['methode']}")

            importances = explication["importances"]
            df_importances = pd.DataFrame(
                {
                    "feature": list(importances.keys()),
                    "importance": list(importances.values()),
                }
            ).sort_values("importance", key=abs, ascending=False)

            # Sens de la contribution, pour un code couleur strict et
            # cohérent avec le reste de l'appli (vert = pousse vers
            # "normal", rouge = pousse vers "anomalie"). Voir la convention
            # de signe documentée dans explainability.py.
            df_importances["direction"] = df_importances["importance"].apply(
                lambda poids: "anomalie" if poids > 0 else "normal"
            )

            graphique = (
                alt.Chart(df_importances)
                .mark_bar()
                .encode(
                    x=alt.X("importance:Q", title="Contribution (signée)"),
                    y=alt.Y("feature:N", title=None, sort=df_importances["feature"].tolist()),
                    color=alt.Color(
                        "direction:N",
                        scale=alt.Scale(
                            domain=["normal", "anomalie"],
                            range=[COULEUR_NORMAL, COULEUR_ANOMALIE],
                        ),
                        legend=alt.Legend(title="Pousse vers"),
                    ),
                    tooltip=["feature", "importance"],
                )
                .configure_axis(
                    labelColor=COULEUR_TEXTE_ATTENUE,
                    titleColor=COULEUR_TEXTE_ATTENUE,
                    gridColor="#22304a",
                )
                .configure_legend(labelColor=COULEUR_TEXTE_ATTENUE, titleColor=COULEUR_TEXTE_ATTENUE)
                .configure_view(strokeWidth=0)
                .properties(background="transparent")
            )

            st.altair_chart(graphique, use_container_width=True)
            st.dataframe(df_importances[["feature", "importance"]], hide_index=True)
else:
    st.info("Charge un fichier CSV ou génère des données factices pour commencer.")
