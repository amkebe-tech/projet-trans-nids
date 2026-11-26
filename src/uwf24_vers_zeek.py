"""
Outils partagés : lire les parquet UWF-ZeekData24 comme dans le notebook V7
(§ 1.4, `charger_fichier_zeek`) et les écrire au format Zeek `conn.log`.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

COLS_Z = ["uid", "ts", "proto", "service", "duration", "orig_pkts", "resp_pkts", "orig_bytes", "resp_bytes",
          "orig_ip_bytes", "resp_ip_bytes", "conn_state", "label_tactic", "label_technique",
          "src_ip_zeek", "src_port_zeek", "dest_ip_zeek", "dest_port_zeek"]


def lire_fichier_uwf(p: Path) -> pd.DataFrame | None:
    """Un parquet UWF -> une ligne par flux (uid), comme le notebook."""
    noms = set(pq.read_schema(p).names)
    zk = pd.read_parquet(p, columns=[c for c in COLS_Z if c in noms])
    if len(zk) == 0:
        return None
    zk["label_tactic"] = zk["label_tactic"].astype(str).str.strip()
    dup = zk["uid"].duplicated(keep=False).to_numpy()
    if dup.any():
        d = zk.loc[dup].copy()
        multi = d.groupby("uid")["label_tactic"].agg(lambda s: " + ".join(sorted(set(s))))
        d["_copie"] = ((d["label_technique"].astype(str).str.strip() == "Duplicate").astype("int8")
                       if "label_technique" in d else 0)
        d = d.sort_values(["uid", "_copie"], kind="stable").drop_duplicates("uid", keep="first").drop(columns="_copie")
        d["label_tactic"] = d["uid"].map(multi)
        zk = pd.concat([zk.loc[~dup], d], ignore_index=True)
    return zk


def lire_uwf24(dossier: Path) -> pd.DataFrame:
    fichiers = sorted(Path(dossier).glob("*.parquet"))
    return pd.concat([x for x in (lire_fichier_uwf(p) for p in fichiers) if x is not None], ignore_index=True)


def vers_conn_log(df: pd.DataFrame, chemin: Path) -> None:
    """Écrit un journal au format Zeek conn.log (TSV), avec une colonne `label` en plus."""
    out = pd.DataFrame({
        "ts": df["ts"].map(lambda v: f"{float(v):.6f}"),
        "uid": df["uid"].astype(str),
        "id.orig_h": df["src_ip_zeek"].astype(str),
        "id.orig_p": df.get("src_port_zeek", pd.Series("-", index=df.index)).astype(str),
        "id.resp_h": df["dest_ip_zeek"].astype(str),
        "id.resp_p": df["dest_port_zeek"].astype(str),
        "proto": df["proto"].astype(str).str.lower(),
        "service": df.get("service", pd.Series("-", index=df.index)).astype(str),
        "duration": df["duration"], "orig_bytes": df["orig_bytes"], "resp_bytes": df["resp_bytes"],
        "conn_state": df["conn_state"].astype(str),
        "orig_pkts": df["orig_pkts"], "orig_ip_bytes": df["orig_ip_bytes"],
        "resp_pkts": df["resp_pkts"], "resp_ip_bytes": df["resp_ip_bytes"],
        "label": df["label_tactic"].astype(str).where(df["label_tactic"].astype(str).str.lower() != "none", "benign"),
    })
    for c in ["duration", "orig_bytes", "resp_bytes", "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"]:
        v = pd.to_numeric(out[c], errors="coerce")
        out[c] = np.where(v.isna(), "-", v.map(lambda x: f"{x:.6f}" if c == "duration" else f"{int(x)}" if x == x else "-"))
    out = out.replace({"nan": "-", "None": "-", "": "-"})
    with open(chemin, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("#separator \\x09\n#set_separator\t,\n#empty_field\t(empty)\n#unset_field\t-\n#path\tconn\n")
        fh.write("#fields\t" + "\t".join(out.columns) + "\n")
        out.to_csv(fh, sep="\t", header=False, index=False, lineterminator="\n")
        fh.write("#close\n")
