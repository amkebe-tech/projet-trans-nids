"""
Module d'explicabilité (XAI) — moteur V7.

Le score V7 d'un flux est la MOYENNE de quatre contributions, une par
détecteur : −log10(p-valeur). L'alerte se déclenche si cette moyenne dépasse
le seuil. On peut donc décomposer exactement la décision :

    score − seuil = Σ_détecteurs ( −log10 p_d − seuil ) / nombre de détecteurs

Chaque terme est la contribution signée d'un détecteur :
    positive -> pousse vers l'anomalie (rouge),
    négative -> pousse vers la normalité (vert).
C'est la même convention de signe que SHAP, reprise par l'affichage.

En complément, pour le VAE, on indique quelles variables du flux il reconstruit
le plus mal (part de l'erreur de reconstruction), et pour le contexte, les
variables de comportement de la machine les plus éloignées du normal du réseau.

Contrat (inchangé par rapport au prototype) :
    explain(ligne, etat, seuil, erreurs_vae, contexte) ->
        {"methode": str, "importances": {nom: poids signé}, "variables_vae": {...}, "contexte": {...}}
"""
from __future__ import annotations

import numpy as np

from config import FEATURES, FEATURES_CONTEXTE, NOMS_DETECTEURS

LIBELLES_VARIABLES = {
    "log_duree_ms": "durée", "log_pkts_aller": "paquets envoyés", "log_pkts_retour": "paquets reçus",
    "log_octets_aller": "octets envoyés", "log_octets_retour": "octets reçus",
    "log_octets_par_pkt_aller": "taille des paquets envoyés", "log_octets_par_pkt_retour": "taille des paquets reçus",
    "log_ratio_pkts": "asymétrie en paquets", "log_ratio_octets": "asymétrie en octets",
    "a_reponse": "réponse reçue", "proto_tcp": "TCP", "proto_udp": "UDP", "proto_autre": "autre protocole",
}
LIBELLES_CONTEXTE = {
    "ctx_log_flux_src": "flux de la source (1 min)", "ctx_log_ports_src": "ports visés (1 min)",
    "ctx_log_dest_src": "destinations visées (1 min)", "ctx_part_sans_rep_src": "part sans réponse (1 min)",
    "ctx_log_flux_dst": "flux reçus par la destination (1 min)", "ctx_log_src_dst": "sources de la destination (1 min)",
    "ctx600_log_flux_src": "flux de la source (10 min)", "ctx600_log_ports_src": "ports visés (10 min)",
    "ctx600_log_dest_src": "destinations visées (10 min)", "ctx600_part_sans_rep_src": "part sans réponse (10 min)",
    "ctx3600_log_flux_src": "flux de la source (1 h)", "ctx3600_log_ports_src": "ports visés (1 h)",
    "ctx3600_log_dest_src": "destinations visées (1 h)", "ctx3600_part_sans_rep_src": "part sans réponse (1 h)",
    "ctx600_log_flux_dst": "flux reçus par la destination (10 min)", "ctx600_log_src_dst": "sources de la destination (10 min)",
    "ctx3600_log_flux_paire": "flux de la paire source → destination (1 h)",
    "ctx3600_cv_intervalles": "irrégularité des intervalles (1 h, bas = balise de bot)",
}


def explain(ligne, etat, seuil: str, erreurs_vae: np.ndarray | None = None,
            contexte_brut: np.ndarray | None = None) -> dict:
    s = etat.seuils[seuil]
    dets = [d for d in etat.refs_p if np.isfinite(ligne.get(f"−log p {d}", np.nan))]
    n = max(len(dets), 1)
    importances = {NOMS_DETECTEURS[d]: float((ligne[f"−log p {d}"] - s) / n) for d in dets}

    variables_vae = {}
    if erreurs_vae is not None and np.sum(erreurs_vae) > 0:
        part = erreurs_vae / np.sum(erreurs_vae)
        variables_vae = {LIBELLES_VARIABLES.get(f, f): float(v) for f, v in zip(FEATURES, part)}

    contexte = {}
    if contexte_brut is not None and etat.st_c is not None and np.isfinite(contexte_brut).all():
        z = (contexte_brut - etat.st_c[0]) / etat.st_c[1]
        contexte = {LIBELLES_CONTEXTE.get(f, f): float(v) for f, v in zip(FEATURES_CONTEXTE, z)}

    return {
        "methode": "décomposition exacte du score V7 par détecteur (−log10 p − seuil) / n",
        "importances": importances,
        "variables_vae": variables_vae,
        "contexte": contexte,
    }
