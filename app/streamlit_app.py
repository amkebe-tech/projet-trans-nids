"""
Application Streamlit — NIDS comportemental, moteur V7.

Parcours en trois étapes, qui suit le fonctionnement réel du système :

  1. Calibration : 24 h de trafic NORMAL du réseau surveillé (journal Zeek
     conn.log). Le système y apprend « le normal de ce réseau » et fixe ses
     seuils de fausses alarmes. Aucune étiquette d'attaque n'est nécessaire.
  2. Analyse     : un journal Zeek à surveiller. Chaque flux reçoit un score
     V7, une alerte au seuil choisi (1 % ou 5 % de fausses alarmes) et le
     détecteur qui l'explique.
  3. Détail      : un flux en particulier, avec la jauge, le verdict et la
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

COULEUR_NORMAL = "#30a46c"
COULEUR_ANOMALIE = "#e5484d"
COULEUR_TEXTE = "#d5d9e0"
COULEUR_TEXTE_ATTENUE = "#7c8594"
COULEUR_GRILLE = "#20252e"
TYPES_JOURNAL = ["log", "csv", "parquet", "labeled"]


def _charger_css() -> str:
    return (Path(__file__).parent / "assets" / "style.css").read_text(encoding="utf-8")


def _n(x) -> str:
    """Entier avec espace comme séparateur de milliers."""
    return f"{x:,}".replace(",", " ")


def _jauge(niveau: float, seuil: float) -> go.Figure:
    couleur = COULEUR_ANOMALIE if niveau >= seuil else COULEUR_NORMAL
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=niveau,
        number={"valueformat": ".3f", "font": {"family": "JetBrains Mono, monospace", "color": COULEUR_TEXTE, "size": 26}},
        gauge={
            "axis": {"range": [0, 1], "tickcolor": COULEUR_TEXTE_ATTENUE, "tickwidth": 1,
                     "tickfont": {"color": COULEUR_TEXTE_ATTENUE, "size": 10}},
            "bar": {"color": couleur, "thickness": 0.25}, "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [{"range": [0, seuil], "color": "rgba(48, 164, 108, 0.10)"},
                      {"range": [seuil, 1], "color": "rgba(229, 72, 77, 0.10)"}],
            "threshold": {"line": {"color": COULEUR_TEXTE_ATTENUE, "width": 1}, "thickness": 0.8, "value": seuil},
        },
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": COULEUR_TEXTE},
                      height=180, margin=dict(l=24, r=24, t=16, b=0))
    return fig


def _style_graphique(g):
    return (g.configure_axis(labelColor=COULEUR_TEXTE_ATTENUE, titleColor=COULEUR_TEXTE_ATTENUE, gridColor=COULEUR_GRILLE,
                             domainColor=COULEUR_GRILLE, tickColor=COULEUR_GRILLE, titleFontWeight="normal",
                             labelFontSize=11, titleFontSize=11)
             .configure_legend(labelColor=COULEUR_TEXTE_ATTENUE, titleColor=COULEUR_TEXTE_ATTENUE, orient="top",
                               labelFontSize=11)
             .configure_view(strokeWidth=0).properties(background="transparent"))


def _barres_signees(valeurs: dict, titre_x: str, legende: tuple = ("normal", "anomalie")):
    df = pd.DataFrame({"élément": list(valeurs), "valeur": list(valeurs.values())}).sort_values("valeur", key=abs, ascending=False)
    df["sens"] = np.where(df["valeur"] > 0, legende[1], legende[0])
    g = alt.Chart(df).mark_bar(cornerRadius=2, height=18).encode(
        x=alt.X("valeur:Q", title=titre_x),
        y=alt.Y("élément:N", title=None, sort=df["élément"].tolist(), axis=alt.Axis(labelLimit=280)),
        color=alt.Color("sens:N", scale=alt.Scale(domain=list(legende), range=[COULEUR_NORMAL, COULEUR_ANOMALIE]),
                        legend=alt.Legend(title=None)),
        tooltip=["élément", alt.Tooltip("valeur:Q", format=".3f")],
    ).properties(height=40 * max(len(df), 1))
    return _style_graphique(g)


def _progression():
    barre = st.progress(0.0, text="Démarrage…")
    return lambda texte, v: barre.progress(min(max(float(v), 0.0), 1.0), text=texte)


@st.cache_resource(show_spinner="Chargement du modèle V7…")
def _modele():
    return mi.charger()


st.set_page_config(page_title="NIDS · V7", page_icon=":material/shield:", layout="wide")
st.html(f"<style>{_charger_css()}</style>")

for cle, defaut in {"etat": None, "nom_reseau": None, "flux_analyse": None, "resultats": None,
                    "erreurs_vae": None, "nom_analyse": None}.items():
    st.session_state.setdefault(cle, defaut)

try:
    MODELE = _modele()
except Exception as e:   # modèle absent ou incompatible : on l'explique au lieu de planter
    st.error(f"Modèle V7 indisponible : {e}")
    st.stop()

# -----------------------------------------------------------------------------
# En-tête : titre + état sur une ligne
# -----------------------------------------------------------------------------
etat = st.session_state.etat
with st.container(horizontal=True, vertical_alignment="center", gap="medium"):
    st.html('<div class="nids-title">NIDS <span>· Moteur V7 · détection non supervisée</span></div>', width="content")
    st.space("stretch")
    st.badge(f"{len(FEATURES)} flux + {len(FEATURES_CONTEXTE)} contexte", color="gray",
             help="Variables du modèle")
    st.badge("VAE · Contexte · kNN · Rareté", color="gray", help="Détecteurs")
    if etat is None:
        st.badge("Non calibré", color="orange", help="Calibration sur 24 h de trafic normal requise")
    else:
        st.badge(f"Réseau : {st.session_state.nom_reseau}", color="green")

onglet_cal, onglet_ana, onglet_det, onglet_info = st.tabs(["Calibration", "Analyse", "Détail flux", "Modèle"])

# =============================================================================
# CALIBRATION
# =============================================================================
with onglet_cal:
    col_a, col_b = st.columns([3, 2], gap="large")
    with col_a:
        st.markdown("##### Nouvelle calibration")
        nom = st.text_input("Nom du réseau", value=st.session_state.nom_reseau or "mon-reseau")
        fichier_cal = st.file_uploader("Trafic normal (24 h)", type=TYPES_JOURNAL, key="up_cal",
                                       help="Zeek conn.log, ou CSV / parquet aux colonnes Zeek. Sans étiquettes.")
        with st.container(horizontal=True):
            lancer = st.button("Calibrer", type="primary", disabled=fichier_cal is None)
            demo = st.button("Démo UWF 2024", disabled=not DEMO_CALIBRATION.exists(),
                             help="24 h normales d'UWF 2024" if DEMO_CALIBRATION.exists()
                             else "Fichier de démonstration absent (src/creer_demo.py).")
    with col_b:
        st.markdown("##### Réseau enregistré")
        existants = mi.etats_disponibles()
        with st.container(horizontal=True, vertical_alignment="bottom"):
            choix = st.selectbox("États enregistrés", existants, index=None,
                                 placeholder="aucun" if not existants else "choisir…")
            if st.button("Charger", disabled=choix is None):
                st.session_state.etat, st.session_state.nom_reseau = mi.charger_etat(choix), choix
                st.session_state.resultats = None
                st.rerun()

    if lancer or demo:
        try:
            with st.spinner("Lecture du journal…"):
                if demo:
                    flux = mi.lire(DEMO_CALIBRATION)
                    nom = "UWF-2024-demo"
                else:
                    flux = mi.lire(fichier_cal, fichier_cal.name)
            st.session_state.etat = mi.calibrer(flux, _progression())
            st.session_state.nom_reseau = nom
            st.session_state.resultats = None
            chemin = mi.sauver_etat(st.session_state.etat, nom)
            st.toast(f"Réseau « {nom} » calibré ({chemin.name})")
            st.rerun()
        except Exception as e:
            st.error(str(e))

    if st.session_state.etat is not None:
        r = st.session_state.etat.resume
        st.divider()
        st.markdown(f"##### {st.session_state.nom_reseau}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Flux de calibration", _n(r["flux de calibration"]), border=True)
        m2.metric("Détecteurs actifs", f"{len(r['détecteurs actifs'])} / 4", border=True)
        m3.metric("Seuil 1 %", f"{r['seuils']['1 %']:.2f}", border=True)
        m4.metric("Seuil 5 %", f"{r['seuils']['5 %']:.2f}", border=True)
        st.caption(f"{r['période']} · "
                   + " · ".join(f"{k} {_n(v)}" for k, v in r["répartition des protocoles"].items()))
        if not r["contexte disponible"]:
            st.badge("Contexte indisponible", color="orange", help="Pas d'IP ou d'horodatage dans le journal")

# =============================================================================
# ANALYSE
# =============================================================================
with onglet_ana:
    if st.session_state.etat is None:
        st.caption("Calibration requise.")
    else:
        col_a, col_b = st.columns([3, 2], gap="large")
        with col_a:
            fichier_ana = st.file_uploader("Journal à analyser", type=TYPES_JOURNAL, key="up_ana")
            with st.container(horizontal=True):
                lancer_a = st.button("Analyser", type="primary", disabled=fichier_ana is None)
                demo_a = st.button("Démo UWF 2024 (attaques)", disabled=not DEMO_ANALYSE.exists())
        with col_b:
            seuil = st.segmented_control(
                "Taux de fausses alarmes", SEUILS, default=SEUIL_PAR_DEFAUT, required=True,
                help="1 % : peu de fausses alarmes, attaques discrètes possibles. "
                     "5 % : plus de détections, 5× plus de fausses alarmes.")
            st.session_state.seuil = seuil

        if lancer_a or demo_a:
            try:
                with st.spinner("Lecture du journal…"):
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
            st.divider()
            with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown(f"##### {st.session_state.nom_analyse}")
                st.badge(f"seuil {seuil}", color="gray")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Flux analysés", _n(len(res)), border=True)
            m2.metric("Alertes", _n(int(alerte.sum())), f"{alerte.mean():.1%}", delta_color="red" if alerte.any() else "off",
                      delta_arrow="off", border=True)
            m3.metric("Sources en alerte", _n(res.loc[alerte, "source"].nunique()), border=True)
            m4.metric("Période", f"{res['horodatage'].min():%d/%m %H:%M} → {res['horodatage'].max():%d/%m %H:%M}",
                      border=True)

            met = mi.metriques(res, seuil)
            if met is not None:
                st.markdown("###### Vérité terrain", help="Journal étiqueté : comparaison avec les labels.")
                cols = st.columns(6)
                for col, (k, v) in zip(cols, met.items()):
                    obj = OBJECTIFS.get(k)
                    valeur = "—" if not np.isfinite(v) else (f"{v:.1%}" if k in ("Fausses alarmes", "% d'attaques") else f"{v:.3f}")
                    if obj is None:
                        col.metric(k, valeur, border=True)
                    else:
                        atteint = np.isfinite(v) and v > obj
                        col.metric(k, valeur, f"obj. > {obj:.2f}", delta_color="green" if atteint else "red",
                                   delta_arrow="off", border=True)

            # Chronologie des alertes
            t = res[["horodatage"]].copy()
            t["statut"] = np.where(alerte, "alerte", "normal")
            pas = "10min" if (res["horodatage"].max() - res["horodatage"].min()) < pd.Timedelta(days=2) else "1h"
            chrono = t.groupby([pd.Grouper(key="horodatage", freq=pas), "statut"]).size().reset_index(name="flux")
            st.markdown("###### Chronologie", help=f"Flux par tranche de {pas}")
            g = alt.Chart(chrono).mark_bar().encode(
                x=alt.X("horodatage:T", title=None), y=alt.Y("flux:Q", title=None),
                color=alt.Color("statut:N", scale=alt.Scale(domain=["normal", "alerte"], range=[COULEUR_NORMAL, COULEUR_ANOMALIE]),
                                legend=alt.Legend(title=None)),
                tooltip=["horodatage:T", "statut", "flux"]).properties(height=200)
            st.altair_chart(_style_graphique(g), width="stretch")

            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown("###### Top sources")
                top = (res[alerte].groupby("source").agg(alertes=("score V7", "size"), score_max=("score V7", "max"),
                                                         destinations=("destination", "nunique"), ports=("port", "nunique"))
                       .sort_values("alertes", ascending=False).head(15))
                st.dataframe(top, width="stretch")
            with c2:
                st.markdown("###### Détecteur dominant")
                if alerte.any():
                    st.dataframe(res.loc[alerte, "détecteur dominant"].value_counts().rename("alertes"),
                                 width="stretch")
                if "type" in res:
                    st.markdown("###### Détection par type", help="Vérité terrain")
                    par_type = res.groupby("type").agg(flux=("score V7", "size"), détectés=(f"alerte {seuil}", "mean"))
                    par_type["détectés"] = (100 * par_type["détectés"]).round(1).astype(str) + " %"
                    st.dataframe(par_type, width="stretch")

            with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown("###### Alertes", help="Triées par score. N° de ligne → onglet Détail flux.")
                st.badge(f"{min(int(alerte.sum()), MAX_ALERTES_AFFICHEES)} affichées", color="gray")
                st.space("stretch")
                st.download_button("Export CSV",
                                   res[alerte].sort_values("score V7", ascending=False).to_csv(index=True).encode("utf-8"),
                                   file_name=f"alertes_{seuil.replace(' %', 'pc')}.csv", mime="text/csv",
                                   icon=":material/download:")
            colonnes = ["horodatage", "source", "destination", "port", "protocole", "état", "score V7", "détecteur dominant"] \
                       + (["étiquette", "type"] if "type" in res else [])
            liste = res[alerte].sort_values("score V7", ascending=False).head(MAX_ALERTES_AFFICHEES)[colonnes]
            st.dataframe(liste, width="stretch", height=320)

# =============================================================================
# DÉTAIL D'UN FLUX
# =============================================================================
with onglet_det:
    res = st.session_state.resultats
    if res is None:
        st.caption("Aucune analyse.")
    else:
        seuil = st.session_state.get("seuil", SEUIL_PAR_DEFAUT)
        alerte = res[f"alerte {seuil}"]
        defaut = int(res.loc[alerte, "score V7"].idxmax()) if alerte.any() else 0
        i = int(st.number_input("N° de ligne", min_value=0, max_value=len(res) - 1, value=defaut,
                                help="Par défaut : l'alerte la plus forte.", width=200))
        ligne = res.iloc[i]
        resultat = mi.predict(ligne, st.session_state.etat, seuil)
        classe = resultat["verdict"]

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.html(
                f"""<div class="verdict verdict--{classe}">
                <div class="verdict__label">Verdict · seuil {seuil}</div>
                <div class="verdict__value verdict__value--{classe}">{classe.upper()}</div>
                <div class="verdict__score">{resultat['score']:.3f}</div>
                <div class="verdict__meta">score V7 {ligne['score V7']:.2f} / seuil {st.session_state.etat.seuils[seuil]:.2f}</div>
                <div class="verdict__meta">{ligne['source']} → {ligne['destination']}:{ligne['port']} ·
                {ligne['protocole']} · {ligne['état']} · {ligne['horodatage']:%Y-%m-%d %H:%M:%S}</div>
                </div>"""
            )
            if "étiquette" in res:
                st.caption(f"Vérité terrain : {ligne['étiquette']} ({ligne['type']})")
        with c2:
            st.plotly_chart(_jauge(resultat["score"], SEUIL_ANOMALIE), width="stretch",
                            config={"displayModeBar": False})

        flux = st.session_state.flux_analyse
        exp = explain(ligne, st.session_state.etat, seuil, st.session_state.erreurs_vae[i], flux.XC[i])
        st.markdown("###### Contribution des détecteurs",
                    help="(−log₁₀ p − seuil) / n. La somme vaut score − seuil : positive → alerte.")
        st.altair_chart(_barres_signees(exp["importances"], None), width="stretch")

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("###### Erreur VAE par variable", help="Part de l'erreur de reconstruction")
            v = pd.Series(exp["variables_vae"]).sort_values(ascending=False).head(6)
            st.dataframe((100 * v).round(1).astype(str).add(" %").rename("part"), width="stretch")
        with c2:
            st.markdown("###### Écart comportemental source", help="En écarts-types par rapport au normal du réseau")
            if exp["contexte"]:
                cz = pd.Series(exp["contexte"]).sort_values(key=abs, ascending=False).head(6)
                st.dataframe(cz.round(1).rename("σ"), width="stretch")
            else:
                st.caption("Contexte indisponible.")
        with st.expander("Flux brut"):
            st.dataframe(ligne.to_frame("valeur").astype(str), width="stretch")

# =============================================================================
# LE MODÈLE
# =============================================================================
with onglet_info:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("###### Détecteurs", help="Apprentissage sur trafic normal uniquement (zero-day). "
                                               "Score V7 = moyenne des −log₁₀ p par protocole.")
        st.table(pd.DataFrame({
            "Détecteur": ["VAE ajusté (×5)", "Contexte IF (1 min / 10 min / 1 h)", "kNN local", "Rareté"],
            "Signal": ["Flux atypique", "Source atypique", "Distance au normal", "Combinaison proto/état/port rare"],
        }).set_index("Détecteur"))
    with c2:
        st.markdown("###### Résultats mesurés", help="Entraînement CIC-2017 + CIC-2018 + UWF 2022, calibré.")
        st.table(pd.DataFrame({
            "Test": ["UWF 2022 → 2025", "UNSW-NB15 (inconnu)", "IoT-23 (aveugle)"],
            "Seuil 1 %": ["4/4 objectifs", "F1 0,897 · AUC 0,992", "AUC 0,887 · rappel 0,13"],
            "Seuil 5 %": ["4/4 objectifs", "F1 0,841", "rappel 0,55 · préc. 0,96"],
        }).set_index("Test"))
    st.caption("Limites : rappel faible à 1 % sur réseau inconnu · balayages discrets et C&C difficiles · "
               "calibration sans attaque requise. Voir documentation/V7-documentation-complete.md")
    with st.expander("Métadonnées du modèle"):
        st.json(MODELE.infos, expanded=False)
