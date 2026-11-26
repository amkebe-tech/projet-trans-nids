"""
Application Streamlit — NIDS comportemental, moteur V7.

Parcours en trois étapes, qui suit le fonctionnement réel du système :

  ① Calibration : 24 h de trafic NORMAL du réseau surveillé (journal Zeek
     conn.log). Le système y apprend « le normal de ce réseau » et fixe ses
     seuils de fausses alarmes. Aucune étiquette d'attaque n'est nécessaire.
  ② Analyse     : un journal Zeek à surveiller. Chaque flux reçoit un score
     V7, une alerte au seuil choisi (1 % ou 5 % de fausses alarmes) et le
     détecteur qui l'explique.
  ③ Détail      : un flux en particulier, avec la jauge, le verdict et la
     décomposition de la décision.

Ce fichier ne contient que de l'interface : le modèle est dans
model_interface.py (et moteur_v7.py), l'explication dans explainability.py.

Lancer :  streamlit run app/streamlit_app.py
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import model_interface as mi
from config import (DEMO_ANALYSE, DEMO_CALIBRATION, FEATURES, FEATURES_CONTEXTE, MAX_ALERTES_AFFICHEES,
                    OBJECTIFS, SEUIL_ANOMALIE, SEUIL_PAR_DEFAUT, SEUILS)
from explainability import explain

COULEUR_NORMAL = "#22c55e"
COULEUR_ANOMALIE = "#ef4444"
COULEUR_ACCENT = "#22d3ee"
COULEUR_TEXTE = "#e6edf3"
COULEUR_TEXTE_ATTENUE = "#8b98ac"
COULEUR_GRILLE = "#22304a"
TYPES_JOURNAL = ["log", "csv", "parquet", "labeled"]


def _charger_css() -> str:
    return (Path(__file__).parent / "assets" / "style.css").read_text(encoding="utf-8")


def _jauge(niveau: float, seuil: float) -> go.Figure:
    couleur = COULEUR_ANOMALIE if niveau >= seuil else COULEUR_NORMAL
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=niveau,
        number={"valueformat": ".3f", "font": {"family": "JetBrains Mono, monospace", "color": COULEUR_TEXTE, "size": 34}},
        gauge={
            "axis": {"range": [0, 1], "tickcolor": COULEUR_TEXTE_ATTENUE,
                     "tickfont": {"color": COULEUR_TEXTE_ATTENUE, "size": 10}},
            "bar": {"color": couleur, "thickness": 0.35}, "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [{"range": [0, seuil], "color": "rgba(34, 197, 94, 0.18)"},
                      {"range": [seuil, 1], "color": "rgba(239, 68, 68, 0.18)"}],
            "threshold": {"line": {"color": COULEUR_TEXTE, "width": 2}, "thickness": 0.85, "value": seuil},
        },
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": COULEUR_TEXTE},
                      height=220, margin=dict(l=30, r=30, t=30, b=10))
    return fig


def _style_graphique(g):
    return (g.configure_axis(labelColor=COULEUR_TEXTE_ATTENUE, titleColor=COULEUR_TEXTE_ATTENUE, gridColor=COULEUR_GRILLE)
             .configure_legend(labelColor=COULEUR_TEXTE_ATTENUE, titleColor=COULEUR_TEXTE_ATTENUE)
             .configure_view(strokeWidth=0).properties(background="transparent"))


def _barres_signees(valeurs: dict, titre_x: str, legende: tuple = ("normal", "anomalie")):
    df = pd.DataFrame({"élément": list(valeurs), "valeur": list(valeurs.values())}).sort_values("valeur", key=abs, ascending=False)
    df["sens"] = np.where(df["valeur"] > 0, legende[1], legende[0])
    g = alt.Chart(df).mark_bar().encode(
        x=alt.X("valeur:Q", title=titre_x),
        y=alt.Y("élément:N", title=None, sort=df["élément"].tolist()),
        color=alt.Color("sens:N", scale=alt.Scale(domain=list(legende), range=[COULEUR_NORMAL, COULEUR_ANOMALIE]),
                        legend=alt.Legend(title="Pousse vers")),
        tooltip=["élément", alt.Tooltip("valeur:Q", format=".3f")],
    )
    return _style_graphique(g)


def _progression():
    barre = st.progress(0.0, text="Démarrage…")
    return lambda texte, v: barre.progress(min(max(float(v), 0.0), 1.0), text=texte)


@st.cache_resource(show_spinner="Chargement du modèle V7…")
def _modele():
    return mi.charger()


st.set_page_config(page_title="NIDS Comportemental — V7", page_icon="📡", layout="wide")
st.markdown(f"<style>{_charger_css()}</style>", unsafe_allow_html=True)

for cle, defaut in {"etat": None, "nom_reseau": None, "flux_analyse": None, "resultats": None,
                    "erreurs_vae": None, "nom_analyse": None}.items():
    st.session_state.setdefault(cle, defaut)

try:
    MODELE = _modele()
except Exception as e:   # modèle absent ou incompatible : on l'explique au lieu de planter
    st.error(f"Impossible de charger le modèle V7 : {e}")
    st.stop()

etat = st.session_state.etat
st.markdown(
    f"""
    <div class="nids-header">
        <div class="nids-header__title">NIDS — Console de Détection Comportementale</div>
        <div class="nids-header__subtitle">Moteur V7 · détection non supervisée · zero-day</div>
        <div class="nids-header__divider"></div>
        <div class="nids-badge-row">
            <span class="nids-badge">DÉTECTEURS : VAE ×5 · CONTEXTE · kNN · RARETÉ</span>
            <span class="nids-badge">VARIABLES : {len(FEATURES)} FLUX + {len(FEATURES_CONTEXTE)} CONTEXTE</span>
            <span class="nids-badge">RÉSEAU : {(st.session_state.nom_reseau or "NON CALIBRÉ").upper()}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if etat is None:
    st.markdown(
        """<div class="nids-banner"><div class="nids-banner__tag">Étape 1 requise : calibration</div>
        <div class="nids-banner__text">Le moteur V7 n'apprend que du trafic normal. Avant d'analyser un réseau,
        il doit en apprendre le comportement habituel sur <strong>24 h de trafic normal</strong> (onglet ①).
        Sans cette étape, il produirait 89 à 99 % de fausses alarmes sur un réseau nouveau.</div></div>""",
        unsafe_allow_html=True,
    )

onglet_cal, onglet_ana, onglet_det, onglet_info = st.tabs(
    ["① Calibration", "② Analyse", "③ Détail d'un flux", "ℹ️ Le modèle"])

# =============================================================================
# ① CALIBRATION
# =============================================================================
with onglet_cal:
    st.subheader("Apprendre le trafic normal du réseau surveillé")
    st.caption("Journal Zeek `conn.log` (ou CSV / parquet aux colonnes Zeek) contenant environ 24 h de trafic "
               "**normal**. Aucune étiquette n'est nécessaire. Le système ajuste ses 5 VAE au réseau, construit "
               "ses références et fixe les seuils à 1 % et 5 % de fausses alarmes.")

    col_a, col_b = st.columns([3, 2])
    with col_a:
        nom = st.text_input("Nom du réseau", value=st.session_state.nom_reseau or "mon-reseau")
        fichier_cal = st.file_uploader("Journal de calibration (trafic normal)", type=TYPES_JOURNAL, key="up_cal")
        c1, c2 = st.columns(2)
        lancer = c1.button("Calibrer avec ce fichier", type="primary", disabled=fichier_cal is None)
        demo = c2.button("Démonstration : 24 h normales d'UWF 2024", disabled=not DEMO_CALIBRATION.exists(),
                         help=None if DEMO_CALIBRATION.exists() else "Fichier de démonstration absent (src/creer_demo.py).")
    with col_b:
        existants = mi.etats_disponibles()
        st.markdown("**Ou reprendre un réseau déjà calibré**")
        choix = st.selectbox("États enregistrés", existants, index=None, placeholder="aucun" if not existants else "choisir…")
        if st.button("Charger cet état", disabled=choix is None):
            st.session_state.etat, st.session_state.nom_reseau = mi.charger_etat(choix), choix
            st.session_state.resultats = None
            st.rerun()

    if lancer or demo:
        try:
            with st.spinner("Lecture du journal et calcul des variables…"):
                if demo:
                    flux = mi.lire(DEMO_CALIBRATION)
                    nom = "UWF-2024-demo"
                else:
                    flux = mi.lire(fichier_cal, fichier_cal.name)
            st.session_state.etat = mi.calibrer(flux, _progression())
            st.session_state.nom_reseau = nom
            st.session_state.resultats = None
            chemin = mi.sauver_etat(st.session_state.etat, nom)
            st.success(f"Réseau « {nom} » calibré et enregistré ({chemin.name}).")
            st.rerun()
        except Exception as e:
            st.error(str(e))

    if st.session_state.etat is not None:
        r = st.session_state.etat.resume
        st.markdown(f"#### Réseau calibré : {st.session_state.nom_reseau}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Flux de calibration", f"{r['flux de calibration']:,}".replace(",", " "))
        m2.metric("Détecteurs actifs", f"{len(r['détecteurs actifs'])} / 4")
        m3.metric("Seuil à 1 %", f"{r['seuils']['1 %']:.2f}")
        m4.metric("Seuil à 5 %", f"{r['seuils']['5 %']:.2f}")
        st.caption(f"Période : {r['période']} · protocoles : "
                   + ", ".join(f"{k} {v:,}".replace(",", " ") for k, v in r["répartition des protocoles"].items())
                   + ("" if r["contexte disponible"] else " · ⚠ contexte indisponible (pas d'IP ou d'horodatage)"))

# =============================================================================
# ② ANALYSE
# =============================================================================
with onglet_ana:
    st.subheader("Analyser un journal de trafic")
    if st.session_state.etat is None:
        st.info("Commencez par calibrer un réseau (onglet ①).")
    else:
        col_a, col_b = st.columns([3, 2])
        with col_a:
            fichier_ana = st.file_uploader("Journal à analyser", type=TYPES_JOURNAL, key="up_ana")
            c1, c2 = st.columns(2)
            lancer_a = c1.button("Analyser ce fichier", type="primary", disabled=fichier_ana is None)
            demo_a = c2.button("Démonstration : trafic UWF 2024 avec attaques", disabled=not DEMO_ANALYSE.exists())
        with col_b:
            seuil = st.radio("Seuil de fausses alarmes", SEUILS, index=SEUILS.index(SEUIL_PAR_DEFAUT), horizontal=True,
                             help="1 % : peu de fausses alarmes, mais les attaques discrètes peuvent passer. "
                                  "5 % : beaucoup plus de détections, 5 fois plus de fausses alarmes.")
            st.session_state.seuil = seuil

        if lancer_a or demo_a:
            try:
                with st.spinner("Lecture du journal et calcul des variables…"):
                    flux = mi.lire(DEMO_ANALYSE) if demo_a else mi.lire(fichier_ana, fichier_ana.name)
                res, err = mi.analyser(flux, st.session_state.etat, _progression())
                st.session_state.flux_analyse, st.session_state.resultats, st.session_state.erreurs_vae = flux, res, err
                st.session_state.nom_analyse = DEMO_ANALYSE.name if demo_a else fichier_ana.name
                st.rerun()
            except Exception as e:
                st.error(str(e))

        res = st.session_state.resultats
        if res is not None:
            alerte = res[f"alerte {seuil}"].to_numpy()
            st.markdown(f"#### Résultats — {st.session_state.nom_analyse} (seuil {seuil})")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Flux analysés", f"{len(res):,}".replace(",", " "))
            m2.metric("Alertes", f"{int(alerte.sum()):,}".replace(",", " "), f"{alerte.mean():.1%} des flux",
                      delta_color="off")
            m3.metric("Machines sources en alerte", f"{res.loc[alerte, 'source'].nunique():,}".replace(",", " "))
            m4.metric("Période", f"{res['horodatage'].min():%d/%m %H:%M} → {res['horodatage'].max():%d/%m %H:%M}")

            met = mi.metriques(res, seuil)
            if met is not None:
                st.markdown("**Le journal est étiqueté : comparaison avec la vérité terrain**")
                cols = st.columns(6)
                for col, (k, v) in zip(cols, met.items()):
                    obj = OBJECTIFS.get(k)
                    ok = "" if obj is None or not np.isfinite(v) else (" ✓" if v > obj else " ✗")
                    col.metric(k + ok, "—" if not np.isfinite(v) else (f"{v:.1%}" if k in ("Fausses alarmes", "% d'attaques") else f"{v:.3f}"),
                               None if obj is None else f"objectif > {obj:.2f}", delta_color="off")

            # Chronologie des alertes
            t = res[["horodatage"]].copy()
            t["statut"] = np.where(alerte, "alerte", "normal")
            pas = "10min" if (res["horodatage"].max() - res["horodatage"].min()) < pd.Timedelta(days=2) else "1h"
            chrono = t.groupby([pd.Grouper(key="horodatage", freq=pas), "statut"]).size().reset_index(name="flux")
            g = alt.Chart(chrono).mark_bar().encode(
                x=alt.X("horodatage:T", title=None), y=alt.Y("flux:Q", title=f"flux par tranche de {pas}"),
                color=alt.Color("statut:N", scale=alt.Scale(domain=["normal", "alerte"], range=[COULEUR_NORMAL, COULEUR_ANOMALIE]),
                                legend=alt.Legend(title=None)),
                tooltip=["horodatage:T", "statut", "flux"]).properties(height=220)
            st.altair_chart(_style_graphique(g), width="stretch")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Machines sources les plus signalées**")
                top = (res[alerte].groupby("source").agg(alertes=("score V7", "size"), score_max=("score V7", "max"),
                                                         destinations=("destination", "nunique"), ports=("port", "nunique"))
                       .sort_values("alertes", ascending=False).head(15))
                st.dataframe(top, width="stretch")
            with c2:
                st.markdown("**Pourquoi les alertes ? (détecteur dominant)**")
                if alerte.any():
                    st.dataframe(res.loc[alerte, "détecteur dominant"].value_counts().rename("alertes"),
                                 width="stretch")
                if "type" in res:
                    st.markdown("**Détection par type (vérité terrain)**")
                    par_type = res.groupby("type").agg(flux=("score V7", "size"), détectés=(f"alerte {seuil}", "mean"))
                    par_type["détectés"] = (100 * par_type["détectés"]).round(1).astype(str) + " %"
                    st.dataframe(par_type, width="stretch")

            st.markdown(f"**Alertes, des plus fortes aux plus faibles** ({min(int(alerte.sum()), MAX_ALERTES_AFFICHEES)} affichées)")
            colonnes = ["horodatage", "source", "destination", "port", "protocole", "état", "score V7", "détecteur dominant"] \
                       + (["étiquette", "type"] if "type" in res else [])
            liste = res[alerte].sort_values("score V7", ascending=False).head(MAX_ALERTES_AFFICHEES)[colonnes]
            st.dataframe(liste, width="stretch", height=320)
            st.caption("Le numéro de ligne (à gauche) permet d'ouvrir le flux dans l'onglet ③.")
            st.download_button("Télécharger toutes les alertes (CSV)",
                               res[alerte].sort_values("score V7", ascending=False).to_csv(index=True).encode("utf-8"),
                               file_name=f"alertes_{seuil.replace(' %', 'pc')}.csv", mime="text/csv")

# =============================================================================
# ③ DÉTAIL D'UN FLUX
# =============================================================================
with onglet_det:
    res = st.session_state.resultats
    if res is None:
        st.info("Analysez d'abord un journal (onglet ②).")
    else:
        seuil = st.session_state.get("seuil", SEUIL_PAR_DEFAUT)
        alerte = res[f"alerte {seuil}"]
        defaut = int(res.loc[alerte, "score V7"].idxmax()) if alerte.any() else 0
        i = int(st.number_input("Numéro de ligne du flux", min_value=0, max_value=len(res) - 1, value=defaut,
                                help="Par défaut : l'alerte la plus forte."))
        ligne = res.iloc[i]
        resultat = mi.predict(ligne, st.session_state.etat, seuil)
        classe = resultat["verdict"]
        icone = "🚨" if classe == "anomalie" else "✅"

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"""<div class="alert-card alert-card--{classe}">
                <div class="alert-card__label">Verdict NIDS · seuil {seuil}</div>
                <div class="alert-card__verdict alert-card__verdict--{classe}">{icone} {classe.upper()}</div>
                <div class="alert-card__score">{resultat['score']:.3f}</div>
                <div class="alert-card__score-label">NIVEAU D'ANOMALIE (0,5 = seuil) · score V7 {ligne['score V7']:.2f}
                pour un seuil de {st.session_state.etat.seuils[seuil]:.2f}</div>
                <div class="alert-card__score-label">{ligne['source']} → {ligne['destination']}:{ligne['port']} ·
                {ligne['protocole']} · {ligne['état']} · {ligne['horodatage']:%Y-%m-%d %H:%M:%S}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with c2:
            st.plotly_chart(_jauge(resultat["score"], SEUIL_ANOMALIE), width="stretch",
                            config={"displayModeBar": False})
        if "étiquette" in res:
            st.caption(f"Vérité terrain : **{ligne['étiquette']}** ({ligne['type']}).")

        flux = st.session_state.flux_analyse
        exp = explain(ligne, st.session_state.etat, seuil, st.session_state.erreurs_vae[i], flux.XC[i])
        st.markdown("#### Pourquoi ce verdict ?")
        st.caption("Contribution de chaque détecteur : (−log₁₀ p − seuil) / nombre de détecteurs. La somme vaut "
                   "exactement « score − seuil » : positive → alerte.")
        st.altair_chart(_barres_signees(exp["importances"], "contribution au score"), width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Variables du flux que le VAE reconstruit le plus mal**")
            v = pd.Series(exp["variables_vae"]).sort_values(ascending=False).head(6)
            st.dataframe((100 * v).round(1).astype(str).add(" %").rename("part de l'erreur"), width="stretch")
        with c2:
            st.markdown("**Comportement de la machine source (écart au normal du réseau)**")
            if exp["contexte"]:
                cz = pd.Series(exp["contexte"]).sort_values(key=abs, ascending=False).head(6)
                st.dataframe(cz.round(1).rename("écart (en écarts-types)"), width="stretch")
            else:
                st.caption("Contexte indisponible pour ce journal.")
        with st.expander("Toutes les informations du flux"):
            st.dataframe(ligne.to_frame("valeur").astype(str), width="stretch")

# =============================================================================
# ℹ️ LE MODÈLE
# =============================================================================
with onglet_info:
    st.subheader("Le moteur V7")
    st.markdown(
        """
Le moteur V7 est le meilleur système obtenu dans le projet (notebook `NIDS-v7-kaggle.ipynb`).
Il n'apprend **que du trafic normal** : toute attaque est pour lui une attaque jamais vue (logique *zero-day*).

| Détecteur | Question posée |
|---|---|
| **VAE ajusté** (5 graines) | Ce flux ressemble-t-il à un flux normal ? |
| **Contexte** (Isolation Forest, 1 min / 10 min / 1 h) | La machine source se comporte-t-elle normalement ? |
| **kNN local** | Ce flux est-il loin de tous les flux normaux de *ce* réseau ? |
| **Rareté** | La combinaison protocole / état / port est-elle courante ici ? |

Chaque score devient une **p-valeur** (par protocole) ; le score V7 est la moyenne des −log₁₀ p.

**Résultats mesurés (entraînement CIC-2017 + CIC-2018 + UWF 2022, calibré) :**

| Test | Seuil 1 % | Seuil 5 % |
|---|---|---|
| UWF 2022 → 2025 (même réseau, 4 années) | **4 objectifs sur 4** chaque année | 4/4 |
| UNSW-NB15 (réseau jamais vu) | F1 0,897, AUC 0,992 | F1 0,841 |
| IoT-23 (juge aveugle, réseau inconnu) | AUC 0,887, rappel 0,13 | rappel 0,55, précision 0,96 |

**Limites à connaître** : sur un réseau totalement inconnu, le classement est bon mais le rappel est faible au
seuil de 1 % ; les balayages discrets et le C&C sont les plus difficiles. La calibration suppose 24 h de trafic
sans attaque. Détails : `documentation/V7-documentation-complete.md`.
        """
    )
    st.json(MODELE.infos, expanded=False)
