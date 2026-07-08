# wasserstein_norm.py
import pandas as pd
from scipy.stats import wasserstein_distance

real_path = 'data/Friday-WorkingHours-Morning.pcap_ISCX.csv'
synth_path = 'flux_synthetiques_benign.csv'
cols = ['Flow Duration','Packet Length Mean']

r = pd.read_csv(real_path, low_memory=False)[cols].dropna()
s = pd.read_csv(synth_path, low_memory=False)[cols].dropna()

for col in cols:
    W = wasserstein_distance(r[col], s[col])
    med = r[col].median()
    iqr = r[col].quantile(0.75) - r[col].quantile(0.25)
    print(col, "W=", W, "W/median=", (W/med if med else None), "W/IQR=", (W/iqr if iqr else None))
