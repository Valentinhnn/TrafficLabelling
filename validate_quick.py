# validate_quick.py
import pandas as pd
from scipy.stats import ks_2samp

REAL_CSV = 'data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'  # adapte si nécessaire
SYNTH_CSV = 'flux_synthetiques_benign.csv'
WANTED = ['Flow Duration', 'Packet Length Mean']

print("Fichier réel utilisé :", REAL_CSV)
r = pd.read_csv(REAL_CSV, low_memory=False)
s = pd.read_csv(SYNTH_CSV, low_memory=False)

print("\n=== KS tests univariés ===")
for col in WANTED:
    if col in r.columns and col in s.columns:
        stat, p = ks_2samp(r[col].dropna(), s[col].dropna())
        print(f"{col} : KS stat={stat:.4f}, p={p:.4f}")
    else:
        print(f"{col} : colonne manquante")

print("\n=== Corrélations (colonnes communes) ===")
common = [c for c in WANTED if c in r.columns and c in s.columns]
if common:
    print("Corr réel:")
    print(r[common].corr())
    print("\nCorr synth:")
    print(s[common].corr())
else:
    print("Aucune colonne commune pour corrélations.")
