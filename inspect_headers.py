# inspect_headers.py
import pandas as pd
import os

path = "data/Friday-WorkingHours-Morning.pcap_ISCX.csv"
print("Fichier inspecté:", path)
try:
    df = pd.read_csv(path, nrows=0, encoding="latin1")
    cols = df.columns.tolist()
    for i, c in enumerate(cols):
        print(f"{i:03d} -> {repr(c)}")
except Exception as e:
    print("Erreur lors de la lecture:", e)
    # fallback: lire la première ligne brute
    with open(path, "rb") as f:
        head = f.readline()
    print("Première ligne brute (bytes):", head[:400])
    try:
        print("Première ligne décodée utf-8:", head.decode("utf-8", errors="replace")[:400])
    except Exception:
        pass
