"""
Fabrique les deux journaux Zeek de démonstration à partir d'UWF-ZeekData24 (parquet) :

  donnees_demo/uwf24_calibration_24h.conn.log
      les 24 premières heures de trafic NORMAL (exactement la calibration du notebook V7)
  donnees_demo/uwf24_analyse.conn.log
      6 h de trafic normal qui suivent + 6 h d'attaques du 17 mars 2024 (avec étiquettes)

Usage :  python src/creer_demo.py <dossier des parquet UWF-ZeekData24>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from uwf24_vers_zeek import lire_uwf24, vers_conn_log

SORTIE = Path(__file__).resolve().parent.parent / "donnees_demo"


def main(dossier):
    df = lire_uwf24(Path(dossier))
    ts = df["ts"].astype(float).to_numpy()
    sain = (df["label_tactic"].astype(str).str.lower() == "none").to_numpy()
    debut = ts[sain].min()
    coupe = min(debut + 24 * 3600, np.quantile(ts[sain], 0.20))          # règle du notebook (24 h ou 20 %)
    cal = df[sain & (ts < coupe)].sort_values("ts")
    suite = df[sain & (ts >= coupe) & (ts < coupe + 6 * 3600)]
    t17 = pd.Timestamp("2024-03-17 06:00").timestamp()
    attaques = df[~sain & (ts >= t17) & (ts < t17 + 6 * 3600)]
    ana = pd.concat([attaques, suite]).sort_values("ts")

    SORTIE.mkdir(exist_ok=True)
    vers_conn_log(cal, SORTIE / "uwf24_calibration_24h.conn.log")
    vers_conn_log(ana, SORTIE / "uwf24_analyse.conn.log")
    print(f"Calibration : {len(cal):,} flux normaux ({pd.to_datetime(cal.ts.min(), unit='s')} → "
          f"{pd.to_datetime(cal.ts.max(), unit='s')})")
    print(f"Analyse : {len(ana):,} flux, dont {len(attaques):,} attaques :")
    print(attaques["label_tactic"].value_counts().to_string())


if __name__ == "__main__":
    main(sys.argv[1])
