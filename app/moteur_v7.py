"""
Moteur de détection V7 (NIDS comportemental, non supervisé).

Ce module reprend fidèlement le code du notebook `NIDS-v7-kaggle.ipynb`
(sections 1.1, 2.1 et 2.2) pour l'exécuter dans l'application, sur processeur.
Documentation complète : documentation/V7-documentation-complete.md.

Le système fonctionne en deux phases :

1. calibrer(journal_sain) -> EtatCalibre
   24 h de trafic NORMAL du réseau surveillé, sans aucune étiquette d'attaque.
   On y calcule la normalisation du réseau, on ajuste les 5 VAE (3 passes),
   on construit la base kNN et la table de rareté, puis les p-valeurs par
   protocole et les seuils à 1 % et 5 % de fausses alarmes.

2. analyser(journal, etat) -> DataFrame (un flux par ligne)
   Pour chaque flux : 4 détecteurs -> 4 p-valeurs -> score V7 (moyenne des
   -log10 p) -> alerte si le score dépasse le seuil choisi. Le détecteur
   dominant (plus petite p-valeur) explique l'alerte.

Entrée attendue : un journal Zeek `conn.log` (format TSV avec la ligne
`#fields`), ou un CSV / parquet avec les colonnes Zeek (noms standard
`id.orig_h`… ou noms UWF `src_ip_zeek`…). Une colonne d'étiquette est
facultative ; si elle est présente, l'application affiche aussi les métriques.
"""

from __future__ import annotations

import copy
import io
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Paramètres (identiques au notebook V7)
# ---------------------------------------------------------------------------
SEED = 42
GRAINES = [42, 43, 44, 45, 46]
LOT = 4096
BETA = 0.1
DUREE_MAX_S = 120.0
N_CAL_MAX = 200_000
N_KNN_REF = 50_000
DETECTEURS_V7 = ["VAE ajusté", "IF contexte multi-échelle", "kNN local", "rareté"]
NOMS_COURTS = {
    "VAE ajusté": "flux atypique (VAE)",
    "IF contexte multi-échelle": "comportement de la machine (contexte)",
    "kNN local": "loin des flux normaux du réseau (kNN)",
    "rareté": "combinaison protocole / état / port rare",
}
PROTOS = ["TCP", "UDP", "autre"]
ETATS = ["établi", "sans réponse", "refusé", "réinitialisé", "autre", "inconnu"]
ZEEK_ETAT = {"SF": 0, "S1": 0, "S2": 0, "S3": 0, "S0": 1, "REJ": 2,
             "RSTO": 3, "RSTR": 3, "RSTOS0": 3, "RSTRH": 3}
LABELS_SAINS = {"benign", "normal", "none", "0", "background", "nan", "", "-"}

CHEMIN_MODELE = Path(__file__).resolve().parent.parent / "models" / "systeme_v7.joblib"

dev = torch.device("cpu")


def _graine(g: int) -> None:
    random.seed(g)
    np.random.seed(g)
    torch.manual_seed(g)


# ---------------------------------------------------------------------------
# 1. Variables (notebook § 1.1)
# ---------------------------------------------------------------------------
def features_v3(dur_s, fp, bp, fb, bb, proto) -> pd.DataFrame:
    dur_s, fp, bp, fb, bb, proto = (np.asarray(v, dtype="float64") for v in (dur_s, fp, bp, fb, bb, proto))
    dur_ms = np.clip(dur_s, 0, DUREE_MAX_S) * 1e3
    tcp, udp = proto == 6, proto == 17
    X = {
        "log_duree_ms": np.log1p(dur_ms),
        "log_pkts_aller": np.log1p(fp),
        "log_pkts_retour": np.log1p(bp),
        "log_octets_aller": np.log1p(fb),
        "log_octets_retour": np.log1p(bb),
        "log_octets_par_pkt_aller": np.log1p(fb / np.maximum(fp, 1)),
        "log_octets_par_pkt_retour": np.log1p(bb / np.maximum(bp, 1)),
        "log_ratio_pkts": np.log1p(bp) - np.log1p(fp),
        "log_ratio_octets": np.log1p(bb) - np.log1p(fb),
        "a_reponse": bp > 0,
        "proto_tcp": tcp,
        "proto_udp": udp,
        "proto_autre": ~(tcp | udp),
    }
    return pd.DataFrame({k: np.asarray(v, dtype="float32") for k, v in X.items()})


CTX5 = ["ctx_log_flux_src", "ctx_log_ports_src", "ctx_log_dest_src", "ctx_part_sans_rep_src",
        "ctx_log_flux_dst", "ctx_log_src_dst"]
CTX = CTX5 + [f"ctx{w}_{k}" for w in (600, 3600)
              for k in ("log_flux_src", "log_ports_src", "log_dest_src", "part_sans_rep_src")] \
          + ["ctx600_log_flux_dst", "ctx600_log_src_dst", "ctx3600_log_flux_paire", "ctx3600_cv_intervalles"]


def _code(v):
    return pd.factorize(pd.Series(v).astype(str), sort=False)[0].astype("int64")


def _compte(cle_, valeur):
    u, inv, n = np.unique(cle_, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    m = int(valeur.max()) + 1
    up = np.unique(inv.astype("int64") * m + valeur)
    return inv, n, np.bincount(up // m, minlength=len(u))


def _fenetre(s, d, p, ts, sans_rep, w):
    b = np.floor(ts / w).astype("int64")
    b -= b.min()
    nb = int(b.max()) + 1
    inv, n, n_ports = _compte(s * nb + b, p)
    _, _, n_dest = _compte(s * nb + b, d)
    part = np.bincount(inv, weights=sans_rep) / n
    inv_d, n_d, n_src = _compte(d * nb + b, s)
    return (np.log1p(n[inv]), np.log1p(n_ports[inv]), np.log1p(n_dest[inv]), part[inv],
            np.log1p(n_d[inv_d]), np.log1p(n_src[inv_d]))


def _regularite(s, d, ts, w=3600):
    b = np.floor(ts / w).astype("int64")
    b -= b.min()
    nb = int(b.max()) + 1
    paire = np.unique(s * (int(d.max()) + 1) + d, return_inverse=True)[1].reshape(-1).astype("int64")
    u, inv, n = np.unique(paire * nb + b, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    o = np.lexsort((ts, inv))
    io_, dt = inv[o], np.diff(ts[o])
    meme = io_[1:] == io_[:-1]
    g, dtm = io_[1:][meme], dt[meme]
    m = np.maximum(n - 1, 1)
    moy = np.bincount(g, dtm, len(u)) / m
    var = np.maximum(np.bincount(g, dtm ** 2, len(u)) / m - moy ** 2, 0)
    cv = np.where(moy > 0, np.sqrt(var) / np.where(moy > 0, moy, 1), 0.0)
    cv = np.where(n >= 3, np.minimum(cv, 10.0), 1.0)
    return np.log1p(n[inv]), cv[inv]


def contexte(src, dst, dport, ts, a_reponse) -> pd.DataFrame:
    s, d, p = _code(src), _code(dst), _code(dport)
    ts = np.asarray(ts, dtype="float64")
    sans = 1.0 - np.asarray(a_reponse, dtype="float64")
    X = dict(zip(CTX5, _fenetre(s, d, p, ts, sans, 60)))
    for w in (600, 3600):
        f = _fenetre(s, d, p, ts, sans, w)
        for k, v in zip(("log_flux_src", "log_ports_src", "log_dest_src", "part_sans_rep_src"), f[:4]):
            X[f"ctx{w}_{k}"] = v
        if w == 600:
            X["ctx600_log_flux_dst"], X["ctx600_log_src_dst"] = f[4], f[5]
    X["ctx3600_log_flux_paire"], X["ctx3600_cv_intervalles"] = _regularite(s, d, ts)
    return pd.DataFrame({k: np.asarray(X[k], dtype="float32") for k in CTX})


def port_num(serie):
    s = pd.Series(serie).astype(str).str.strip().str.lower()
    hexa = s.str.startswith("0x")
    v = pd.to_numeric(s.where(~hexa), errors="coerce")
    if hexa.any():
        v[hexa] = s[hexa].map(lambda x: int(x, 16) if re.fullmatch(r"0x[0-9a-f]+", x) else np.nan)
    return v.fillna(-1).clip(-1, 65535).astype("int32").to_numpy()


def etat_depuis(serie):
    return serie.astype(str).str.strip().str.upper().map(ZEEK_ETAT).fillna(4).astype("int8").to_numpy()


# ---------------------------------------------------------------------------
# 2. Lecture d'un journal Zeek (conn.log TSV, CSV ou parquet)
# ---------------------------------------------------------------------------
ALIAS = {
    "ts": ["ts"],
    "src": ["id.orig_h", "src_ip_zeek", "src_ip", "orig_h"],
    "dst": ["id.resp_h", "dest_ip_zeek", "dst_ip", "resp_h"],
    "dport": ["id.resp_p", "dest_port_zeek", "dst_port", "resp_p"],
    "proto": ["proto"],
    "duration": ["duration"],
    "orig_bytes": ["orig_bytes", "src_bytes"],
    "resp_bytes": ["resp_bytes", "dst_bytes"],
    "orig_pkts": ["orig_pkts", "src_pkts"],
    "resp_pkts": ["resp_pkts", "dst_pkts"],
    "orig_ip_bytes": ["orig_ip_bytes", "src_ip_bytes"],
    "resp_ip_bytes": ["resp_ip_bytes", "dst_ip_bytes"],
    "conn_state": ["conn_state"],
}
ALIAS_LABEL = ["label", "label_tactic", "attack", "étiquette"]
ALIAS_TYPE = ["detailed-label", "detailed_label", "type", "label_technique"]
OBLIGATOIRES = ["ts", "src", "dst", "dport", "proto", "duration", "orig_bytes", "resp_bytes",
                "orig_pkts", "resp_pkts", "conn_state"]


def _lire_tsv_zeek(contenu: bytes) -> pd.DataFrame:
    texte = contenu.decode("utf-8", errors="replace")
    champs = None
    for ligne in texte.splitlines():
        if ligne.startswith("#fields"):
            champs = ligne.rstrip("\n").split("\t")[1:]
            break
        if not ligne.startswith("#"):
            break
    if champs is None:
        raise ValueError("Ligne `#fields` introuvable : ce n'est pas un conn.log Zeek au format TSV.")
    df = pd.read_csv(io.StringIO(texte), sep="\t", comment="#", header=None, dtype=str,
                     na_filter=False, on_bad_lines="skip")
    if df.shape[1] > len(champs):
        df = df.iloc[:, :len(champs)]
    df.columns = champs[:df.shape[1]]
    # IoT-23 : la dernière colonne réunit « tunnel_parents label detailed-label »
    if "tunnel_parents" in df.columns and "label" not in df.columns:
        et = df["tunnel_parents"].str.extract(r"(benign|malicious)\s+(\S+)\s*$", flags=re.I)
        if et[0].notna().any():
            df["label"], df["detailed-label"] = et[0].fillna(""), et[1].fillna("-")
    return df


def lire_journal(source, nom: str | None = None) -> pd.DataFrame:
    """Lit un journal Zeek depuis un chemin, des octets ou un fichier téléversé (Streamlit)."""
    if hasattr(source, "read"):
        nom = nom or getattr(source, "name", "")
        contenu = source.read()
    elif isinstance(source, (bytes, bytearray)):
        contenu = bytes(source)
    else:
        p = Path(source)
        nom = nom or p.name
        contenu = p.read_bytes()
    nom = (nom or "").lower()
    if nom.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(contenu))
    tete = contenu[:4096].decode("utf-8", errors="replace")
    if tete.startswith("#separator") or "#fields" in tete:
        return _lire_tsv_zeek(contenu)
    return pd.read_csv(io.BytesIO(contenu), low_memory=False)


@dataclass
class Flux:
    """Flux préparés : variables du modèle + informations d'affichage."""
    XF: np.ndarray                 # (n, 13)
    XC: np.ndarray                 # (n, 18)
    proto: np.ndarray              # 0 TCP, 1 UDP, 2 autre
    cle_rare: np.ndarray
    infos: pd.DataFrame            # ts, src, dst, dport, proto, état, volumes
    etiquette: np.ndarray | None = None   # 1 = attaque, 0 = sain (si fournie)
    type_attaque: np.ndarray | None = None
    rapport: dict = field(default_factory=dict)


def _colonne(df, cle):
    for a in ALIAS[cle]:
        if a in df.columns:
            return df[a]
    return None


def preparer(df: pd.DataFrame) -> Flux:
    """Journal Zeek -> variables V7 (mêmes conventions que le notebook pour UWF et IoT-23)."""
    rapport = {"lignes lues": len(df)}
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # UWF : un même flux recopié sous plusieurs tactiques -> une ligne par uid
    if "uid" in df.columns and df["uid"].duplicated().any():
        avant = len(df)
        df = df.drop_duplicates("uid", keep="first").reset_index(drop=True)
        rapport["doublons (uid) retirés"] = avant - len(df)
    manquantes = [c for c in OBLIGATOIRES if _colonne(df, c) is None]
    if manquantes:
        raise ValueError(
            "Colonnes Zeek manquantes : " + ", ".join(manquantes)
            + ". Attendu : un conn.log Zeek (ts, id.orig_h, id.resp_h, id.resp_p, proto, duration, "
              "orig_bytes, resp_bytes, orig_pkts, resp_pkts, conn_state)."
        )
    num = {}
    for c in ["ts", "duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes"]:
        col = _colonne(df, c)
        num[c] = (pd.to_numeric(col, errors="coerce").fillna(0).astype("float64").to_numpy()
                  if col is not None else None)
    ts = num["ts"]
    if not np.isfinite(ts).all() or (ts <= 0).all():
        raise ValueError("Horodatages `ts` invalides : le contexte (1 min / 10 min / 1 h) ne peut pas être calculé.")
    fb, bb = num["orig_bytes"], num["resp_bytes"]
    if num["orig_ip_bytes"] is not None:
        fb = np.minimum(fb, num["orig_ip_bytes"])
    if num["resp_ip_bytes"] is not None:
        bb = np.minimum(bb, num["resp_ip_bytes"])
    proto_txt = _colonne(df, "proto").astype(str).str.strip().str.lower()
    proto = proto_txt.map({"tcp": 6, "udp": 17, "icmp": 1}).fillna(0).to_numpy()
    f = features_v3(num["duration"], num["orig_pkts"], num["resp_pkts"], fb, bb, proto)
    src = _colonne(df, "src").astype(str).str.strip()
    dst = _colonne(df, "dst").astype(str).str.strip()
    dport = port_num(_colonne(df, "dport"))
    etat = etat_depuis(_colonne(df, "conn_state"))
    c = contexte(src, dst, dport, ts, f["a_reponse"])
    p3 = np.where(proto == 6, 0, np.where(proto == 17, 1, 2)).astype("int8")
    cle = (p3.astype("int64") * 6 + etat) * 65537 + (dport.astype("int64") + 1)

    etiquette = type_attaque = None
    lab = next((df[a] for a in ALIAS_LABEL if a in df.columns), None)
    if lab is not None:
        t = lab.astype(str).str.strip()
        etiquette = (~t.str.lower().isin(LABELS_SAINS)).astype("int8").to_numpy()
        det = next((df[a] for a in ALIAS_TYPE if a in df.columns), None)
        if det is not None and lab.name != "label_tactic":
            type_attaque = np.where(etiquette == 1, det.astype(str).str.strip().to_numpy(), "sain")
        else:
            type_attaque = np.where(etiquette == 1, t.to_numpy(), "sain")
        rapport["étiquettes trouvées"] = f"{int(etiquette.sum()):,} attaques / {len(etiquette):,} flux"

    infos = pd.DataFrame({
        "horodatage": pd.to_datetime(ts, unit="s"),
        "source": src.to_numpy(), "destination": dst.to_numpy(), "port": dport,
        "protocole": np.array(PROTOS)[p3], "état": np.array(ETATS)[etat],
        "durée (s)": np.round(num["duration"], 3),
        "paquets aller": num["orig_pkts"].astype("int64"), "paquets retour": num["resp_pkts"].astype("int64"),
        "octets aller": fb.astype("int64"), "octets retour": bb.astype("int64"),
    })
    rapport["flux préparés"] = len(f)
    rapport["période"] = f"{infos['horodatage'].min():%Y-%m-%d %H:%M} → {infos['horodatage'].max():%Y-%m-%d %H:%M}"
    return Flux(XF=f.to_numpy("float32"), XC=c.to_numpy("float32"), proto=p3, cle_rare=cle,
                infos=infos, etiquette=etiquette, type_attaque=type_attaque, rapport=rapport)


# ---------------------------------------------------------------------------
# 3. Le modèle entraîné (notebook § 2.1)
# ---------------------------------------------------------------------------
class VAE(nn.Module):
    def __init__(self, d, h=64, z=4):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h // 2), nn.ReLU())
        self.mu, self.logvar = nn.Linear(h // 2, z), nn.Linear(h // 2, z)
        self.dec = nn.Sequential(nn.Linear(z, h // 2), nn.ReLU(), nn.Linear(h // 2, h), nn.ReLU(), nn.Linear(h, d))

    def forward(self, x):
        e = self.enc(x)
        mu, lv = self.mu(e), self.logvar(e).clamp(-10, 10)
        return self.dec(mu + torch.randn_like(mu) * torch.exp(0.5 * lv)), mu, lv


@dataclass
class Modele:
    vae: list
    if_ctx: object
    SD_MIN: float
    CLIP: float
    K: int
    EPOCHS_AJUST: int
    LR_AJUST: float
    MIN_GROUPE: int
    SEUILS_FA: dict
    FEATURES: list
    infos: dict


def charger_modele(chemin: Path | str = CHEMIN_MODELE) -> Modele:
    E = joblib.load(chemin)
    if list(E["FEATURES"]) != list(features_v3(*[np.zeros(1)] * 6).columns) or list(E["CTX"]) != CTX:
        raise ValueError("Le fichier du modèle ne correspond pas aux variables de ce moteur.")
    a = E["vae_archi"]
    vaes = []
    for sd in E["vae_state_dicts"]:
        m = VAE(a["d"], a["h"], a["z"])
        m.load_state_dict({k: v.float() for k, v in sd.items()})
        m.eval()
        vaes.append(m)
    infos = {"système": E["système V7"], "détecteurs": E["détecteurs"], "fusion": E["fusion"],
             "sources d'entraînement": E["sources"], "VAE": f"{len(vaes)} graines, {a}"}
    return Modele(vae=vaes, if_ctx=E["if_ctx"], SD_MIN=E["SD_MIN"], CLIP=E["CLIP"], K=E["K_VOISINS"],
                  EPOCHS_AJUST=E["EPOCHS_AJUST"], LR_AJUST=E["LR_AJUST"], MIN_GROUPE=E["MIN_GROUPE"],
                  SEUILS_FA=E["SEUILS_FA"], FEATURES=list(E["FEATURES"]), infos=infos)


# ---------------------------------------------------------------------------
# 4. Détecteurs, p-valeurs, fusion (notebook § 2.2)
# ---------------------------------------------------------------------------
def _stats(M, sd_min):
    M = M[np.isfinite(M).all(axis=1)]
    return M.mean(0).astype("float32"), np.maximum(M.std(0), sd_min).astype("float32")


def _norm(M, st, clip):
    return np.clip((M - st[0]) / st[1], -clip, clip).astype("float32")


@torch.no_grad()
def _erreurs_vae(m, Xn):
    """Erreur de reconstruction par variable (n, 13)."""
    out = []
    for i in range(0, len(Xn), 262144):
        x = torch.from_numpy(np.ascontiguousarray(Xn[i:i + 262144]))
        out.append(((m.dec(m.mu(m.enc(x))) - x) ** 2).numpy())
    return np.concatenate(out) if out else np.zeros((0, Xn.shape[1]), dtype="float32")


def _ajuster(m, Xt, epochs, lr, graine):
    _graine(graine)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), LOT):
            xr, mu, lv = m(Xt[perm[i:i + LOT]])
            x = Xt[perm[i:i + LOT]]
            rec = ((xr - x) ** 2).sum(1)
            kl = -0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)
            loss = (rec + BETA * kl).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    m.eval()
    return m


@torch.no_grad()
def _knn(Xref, X, k):
    R = torch.from_numpy(np.ascontiguousarray(Xref))
    k = min(k, len(Xref))
    out = []
    for i in range(0, len(X), 2048):
        x = torch.from_numpy(np.ascontiguousarray(X[i:i + 2048]))
        out.append(torch.cdist(x, R).topk(k, largest=False).values[:, -1].numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype="float32")


def _table_rarete(cles):
    u, c = np.unique(cles, return_counts=True)
    return u, c, len(cles)


def _rarete(tab, cles):
    u, c, n = tab
    j = np.minimum(np.searchsorted(u, cles), len(u) - 1)
    cnt = np.where(u[j] == cles, c[j], 0)
    return -np.log((cnt + 1.0) / (n + 1.0))


def _pvaleurs(s_cal, g_cal, s_x, g_x, min_groupe):
    p = np.full(len(s_x), np.nan)
    ok = np.isfinite(s_x)
    tout = np.sort(s_cal)
    for g in np.unique(g_x[ok]):
        m = ok & (g_x == g)
        ref = np.sort(s_cal[g_cal == g]) if (g_cal == g).sum() >= min_groupe else tout
        p[m] = (1 + len(ref) - np.searchsorted(ref, s_x[m], side="left")) / (len(ref) + 1)
    return p


@dataclass
class EtatCalibre:
    """Tout ce que la calibration a appris du réseau surveillé."""
    st_f: tuple
    st_c: tuple | None
    vae_ajustes: list           # state_dicts des 5 VAE ajustés
    Xknn: np.ndarray
    table_rarete: tuple
    refs_p: dict                # détecteur -> (scores de référence, protocole)
    seuils: dict                # "1 %" / "5 %" -> seuil sur le score V7
    resume: dict


def _scores(M: Modele, F: Flux, idx, st_f, st_c, vaes, Xknn, table):
    Xf = _norm(F.XF[idx], st_f, M.CLIP)
    err = np.mean([_erreurs_vae(m, Xf) for m in vaes], axis=0)
    out = {"VAE ajusté": err.sum(1), "kNN local": _knn(Xknn, Xf, M.K), "rareté": _rarete(table, F.cle_rare[idx])}
    c = F.XC[idx]
    ok = np.isfinite(c).all(axis=1)
    s = np.full(len(idx), np.nan)
    if ok.any() and st_c is not None:
        s[ok] = -M.if_ctx.score_samples(_norm(c[ok], st_c, M.CLIP))
    out["IF contexte multi-échelle"] = s
    return out, err


def calibrer(M: Modele, F: Flux, progression=None) -> EtatCalibre:
    """Phase 1 : 24 h de trafic NORMAL du réseau -> état calibré (notebook : evaluer_systeme, régime calibré)."""
    def etape(txt, v):
        if progression:
            progression(txt, v)

    n = len(F.XF)
    if n < 2_000:
        raise ValueError(f"Calibration trop courte : {n:,} flux (au moins 2 000 recommandés ; 24 h de trafic normal).")
    if F.etiquette is not None and F.etiquette.any():
        raise ValueError(f"Le journal de calibration contient {int(F.etiquette.sum()):,} flux étiquetés comme "
                         "attaques. La calibration doit se faire sur du trafic NORMAL uniquement.")
    ic = np.arange(n)
    if n > N_CAL_MAX:
        ic = np.sort(np.random.default_rng(SEED).choice(ic, size=N_CAL_MAX, replace=False))
    perm = np.random.default_rng(SEED + 1).permutation(len(ic))
    ic_ref, ic_cal = np.sort(ic[perm[: len(ic) // 2]]), np.sort(ic[perm[len(ic) // 2:]])

    etape("Normalisation du réseau", 0.05)
    st_f = _stats(F.XF[ic], M.SD_MIN)
    st_c = _stats(F.XC[ic], M.SD_MIN) if np.isfinite(F.XC[ic]).all() else None

    Xr = _norm(F.XF[ic_ref], st_f, M.CLIP)
    Xt = torch.from_numpy(Xr)
    vaes = []
    for i, (m, g) in enumerate(zip(M.vae, GRAINES)):
        etape(f"Ajustement du VAE {i + 1}/5 au réseau", 0.10 + 0.12 * i)
        vaes.append(_ajuster(copy.deepcopy(m), Xt, M.EPOCHS_AJUST, M.LR_AJUST, g))
    etape("Base kNN et table de rareté", 0.72)
    Xknn = Xr[np.random.default_rng(SEED).choice(len(Xr), size=min(N_KNN_REF, len(Xr)), replace=False)]
    table = _table_rarete(F.cle_rare[ic_ref])

    etape("p-valeurs par protocole et seuils", 0.80)
    C, _ = _scores(M, F, ic_cal, st_f, st_c, vaes, Xknn, table)
    gC = F.proto[ic_cal]
    nC = len(gC)
    p2 = np.random.default_rng(SEED).permutation(nC)
    A, B = p2[: nC // 2], p2[nC // 2:]
    refs = {}
    for d in DETECTEURS_V7:
        s = C[d][A]
        ok = np.isfinite(s)
        if ok.sum() >= 100:
            refs[d] = (s[ok], gC[A][ok])
    L = np.vstack([-np.log10(_pvaleurs(*refs[d], C[d][B], gC[B], M.MIN_GROUPE)) for d in refs])
    fB = np.nanmean(L, axis=0)
    seuils = {k: float(np.quantile(fB, 1 - v)) for k, v in M.SEUILS_FA.items()}
    etape("Calibration terminée", 1.0)
    resume = {
        "flux de calibration": n, "utilisés": len(ic),
        "référence (ajustement, kNN, rareté)": len(ic_ref), "p-valeurs": len(A), "seuils": len(B),
        "détecteurs actifs": list(refs), "contexte disponible": st_c is not None,
        "répartition des protocoles": {PROTOS[k]: int((F.proto[ic] == k).sum()) for k in range(3)},
        "période": F.rapport.get("période", ""), "seuils": seuils,
    }
    return EtatCalibre(st_f=st_f, st_c=st_c, vae_ajustes=[m.state_dict() for m in vaes], Xknn=Xknn,
                       table_rarete=table, refs_p=refs, seuils=seuils, resume=resume)


def _vaes_ajustes(M: Modele, E: EtatCalibre):
    out = []
    for sd in E.vae_ajustes:
        m = copy.deepcopy(M.vae[0])
        m.load_state_dict(sd)
        m.eval()
        out.append(m)
    return out


def analyser(M: Modele, E: EtatCalibre, F: Flux, progression=None) -> tuple[pd.DataFrame, np.ndarray]:
    """Phase 2 : score V7 de chaque flux. Renvoie (tableau des flux, erreurs du VAE par variable)."""
    idx = np.arange(len(F.XF))
    vaes = _vaes_ajustes(M, E)
    morceaux, erreurs = [], []
    pas = 100_000
    for i in range(0, len(idx), pas):
        if progression:
            progression(f"Analyse des flux {i:,} à {min(i + pas, len(idx)):,}", i / max(len(idx), 1))
        j = idx[i:i + pas]
        S, err = _scores(M, F, j, E.st_f, E.st_c, vaes, E.Xknn, E.table_rarete)
        L = {d: -np.log10(_pvaleurs(*E.refs_p[d], S[d], F.proto[j], M.MIN_GROUPE)) for d in E.refs_p}
        morceaux.append(pd.DataFrame(L))
        erreurs.append(err)
    L = pd.concat(morceaux, ignore_index=True)
    score = np.nanmean(L.to_numpy(), axis=1)
    res = F.infos.copy()
    res["score V7"] = score
    for d in E.refs_p:
        res[f"−log p {d}"] = L[d].to_numpy()
    dom = L.fillna(-np.inf).to_numpy().argmax(axis=1)
    res["détecteur dominant"] = np.array([NOMS_COURTS[d] for d in L.columns])[dom]
    for k, s in E.seuils.items():
        res[f"alerte {k}"] = score >= s
    if F.etiquette is not None:
        res["étiquette"] = np.where(F.etiquette == 1, "attaque", "sain")
        res["type"] = F.type_attaque
    if progression:
        progression("Analyse terminée", 1.0)
    return res, np.concatenate(erreurs)


def niveau(score, seuil):
    """Score V7 ramené entre 0 et 1, égal à 0,5 exactement au seuil (pour la jauge)."""
    score = np.maximum(np.asarray(score, dtype="float64"), 0)
    return score / (score + seuil)


def metriques(res: pd.DataFrame, cle_seuil: str) -> dict | None:
    """Précision, rappel, F1, AUC et fausses alarmes si le journal est étiqueté."""
    if "étiquette" not in res:
        return None
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    y = (res["étiquette"] == "attaque").to_numpy()
    pred = res[f"alerte {cle_seuil}"].to_numpy()
    s = res["score V7"].to_numpy()
    deux = 0 < y.sum() < len(y)
    return {
        "Précision": precision_score(y, pred, zero_division=0) if deux else np.nan,
        "Rappel": recall_score(y, pred, zero_division=0) if y.any() else np.nan,
        "F1": f1_score(y, pred, zero_division=0) if deux else np.nan,
        "AUC-ROC": roc_auc_score(y, s) if deux else np.nan,
        "Fausses alarmes": float(pred[~y].mean()) if (~y).any() else np.nan,
        "% d'attaques": float(y.mean()),
    }


def sauver_etat(E: EtatCalibre, chemin) -> None:
    joblib.dump(E, chemin, compress=3)


def charger_etat(chemin) -> EtatCalibre:
    return joblib.load(chemin)
