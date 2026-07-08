import pandas as pd
from scipy.stats import wasserstein_distance

def norm_cols(df):
    df.columns = [str(c).strip().lstrip('\ufeff').replace('\t',' ') for c in df.columns]
    df.columns = [" ".join(c.split()) for c in df.columns]
    return df

real_path = 'data/Friday-WorkingHours-Morning.pcap_ISCX.csv'
synth_path = 'flux_synthetiques_benign.csv'
cols = ['Flow Duration','Packet Length Mean']

r = pd.read_csv(real_path, low_memory=False)
s = pd.read_csv(synth_path, low_memory=False)
r = norm_cols(r)
s = norm_cols(s)

missing_real = [c for c in cols if c not in r.columns]
missing_synth = [c for c in cols if c not in s.columns]
if missing_real:
    print("Colonnes manquantes dans réel:", missing_real)
if missing_synth:
    print("Colonnes manquantes dans synth:", missing_synth)
if missing_real or missing_synth:
    raise SystemExit("Corrige les noms de colonnes ou adapte 'cols' dans le script.")

r = r[cols].dropna()
s = s[cols].dropna()

for col in cols:
    W = wasserstein_distance(r[col], s[col])
    med = r[col].median()
    iqr = r[col].quantile(0.75) - r[col].quantile(0.25)
    print(col, "W=", W, "W/median=", (W/med if med else None), "W/IQR=", (W/iqr if iqr else None))
