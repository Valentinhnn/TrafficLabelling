# wasserstein_check.py
import pandas as pd
from scipy.stats import wasserstein_distance
import json
import os

REAL_CSV = 'data/Friday-WorkingHours-Morning.pcap_ISCX.csv'
SYNTH_CSV = 'flux_synthetiques_benign.csv'
WANTED = ['Flow Duration', 'Packet Length Mean']
QS = [0.01, 0.05, 0.5, 0.95, 0.99]

def normalize(cols):
    out = []
    for c in cols:
        c2 = str(c).strip().lstrip('\ufeff').replace('\t', ' ')
        c2 = " ".join(c2.split())
        out.append(c2)
    return out

def load_and_normalize(path):
    encs = ['utf-8', 'latin1', 'cp1252']
    for e in encs:
        try:
            df = pd.read_csv(path, encoding=e, nrows=0)
            cols = normalize(df.columns.tolist())
            df_full = pd.read_csv(path, encoding=e, low_memory=False)
            df_full.columns = normalize(df_full.columns.tolist())
            return df_full
        except Exception:
            continue
    with open(path, 'rb') as f:
        head = f.readline()
    try:
        head_dec = head.decode('utf-8', errors='replace')
    except Exception:
        head_dec = head.decode('latin1', errors='replace')
    cols = [c.strip().lstrip('\ufeff') for c in head_dec.split(',')]
    df_full = pd.read_csv(path, header=0, names=cols, encoding='latin1', skiprows=1)
    df_full.columns = normalize(df_full.columns.tolist())
    return df_full

def main():
    if not os.path.exists(REAL_CSV):
        print("Fichier réel introuvable:", REAL_CSV); return
    if not os.path.exists(SYNTH_CSV):
        print("Fichier synthétique introuvable:", SYNTH_CSV); return

    real_df = load_and_normalize(REAL_CSV)
    synth_df = load_and_normalize(SYNTH_CSV)

    print("Colonnes réelles (extrait):", real_df.columns.tolist()[:20])
    print("Colonnes synth (extrait):", synth_df.columns.tolist()[:20])
    print()

    missing_real = [c for c in WANTED if c not in real_df.columns]
    missing_synth = [c for c in WANTED if c not in synth_df.columns]
    if missing_real:
        print("Colonnes manquantes dans le réel:", missing_real)
    if missing_synth:
        print("Colonnes manquantes dans le synthétique:", missing_synth)
    if missing_real or missing_synth:
        print("Si les noms diffèrent, adapte WANTED ou renomme les colonnes.")
        return

    r = real_df[WANTED].dropna()
    s = synth_df[WANTED].dropna()

    for col in WANTED:
        w = wasserstein_distance(r[col], s[col])
        rq = r[col].quantile(QS).to_dict()
        sq = s[col].quantile(QS).to_dict()
        print(f"{col}  —  Wasserstein = {w:.6g}")
        print("  Quantiles réel :", json.dumps({str(k): float(v) for k,v in rq.items()}))
        print("  Quantiles synth:", json.dumps({str(k): float(v) for k,v in sq.items()}))
        print()

if __name__ == "__main__":
    main()
