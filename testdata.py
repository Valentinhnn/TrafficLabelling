import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Nom du fichier à analyser (remplace par celui que tu veux)
file = "Wednesday-workingHours.pcap_ISCX.csv"

print(f"\n=== Analyse du fichier : {file} ===")

# Charger le dataset avec gestion d'encodage
try:
    df = pd.read_csv(file, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(file, encoding="latin1")

# Nettoyer les noms de colonnes
df.columns = df.columns.str.strip()

# Dimensions
print("Dimensions :", df.shape)

# Répartition des classes
if "Label" in df.columns:
    print("Répartition des classes :")
    print(df["Label"].value_counts())

    # Graphique (corrigé pour éviter le FutureWarning)
    plt.figure(figsize=(8,5))
    sns.countplot(x="Label", hue="Label", data=df,
                  order=df['Label'].value_counts().index,
                  palette="viridis", legend=False)
    plt.title(f"Distribution des classes - {file}")
    plt.xticks(rotation=45)
    plt.show()
else:
    print("Colonne 'Label' absente dans ce fichier.")
