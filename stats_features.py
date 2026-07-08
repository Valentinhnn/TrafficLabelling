#!/usr/bin/env python3
"""
stats_features.py
==================
Pipeline d'analyse statistique des features de flux réseau (CICIDS2017).

Usage:
    python stats_features.py --data-dir data --fig-dir figs --k-max 5
"""
import os
import warnings
import argparse
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# DEFAULT CONFIG
# ----------------------------------------------------------------------
DEFAULT_FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Packet Length Mean",
    "Flow IAT Mean",
]
LABEL_COL = "Label"
BENIGN_VALUE = "BENIGN"
DEFAULT_K_MAX = 5
DEFAULT_N_POINTS = 400
DEFAULT_MAX_SAMPLES = 8000
DEFAULT_RANDOM_STATE = 42

# ----------------------------------------------------------------------
# UTILITAIRES
# ----------------------------------------------------------------------
def safe_filename(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")

def load_data(path_or_dir: str, features: list, label_col: str, warnings_out: str = None) -> pd.DataFrame:
    """
    Charge soit un seul CSV, soit TOUS les CSV d'un dossier.
    Ne garde que les colonnes nécessaires, tolère fichiers manquants.
    Enregistre les fichiers ignorés dans warnings_out si fourni.
    """
    wanted = set(features) | {label_col}
    csv_files = []
    if os.path.isdir(path_or_dir):
        csv_files = sorted(f for f in os.listdir(path_or_dir) if f.lower().endswith(".csv"))
        if not csv_files:
            raise ValueError(f"Aucun fichier .csv trouvé dans {path_or_dir}")
        print(f"[INFO] {len(csv_files)} fichiers CSV trouvés, chargement...")
        base_dir = path_or_dir
    else:
        csv_files = [os.path.basename(path_or_dir)]
        base_dir = os.path.dirname(path_or_dir) or "."

    frames = []
    warnings_list = []
    for fname in csv_files:
        fpath = os.path.join(base_dir, fname)
        try:
            df_tmp = pd.read_csv(fpath, encoding="latin1", usecols=lambda c: c.strip() in wanted)
            df_tmp.columns = df_tmp.columns.str.strip()
            missing = wanted - set(df_tmp.columns)
            if missing:
                print(f"[WARN] {fname} : colonnes manquantes {missing} -> fichier ignoré")
                warnings_list.append({"file": fname, "missing": ";".join(sorted(missing))})
                continue
            df_tmp["__source_file"] = fname
            frames.append(df_tmp)
            print(f"[OK] {fname} : {len(df_tmp)} lignes")
        except Exception as e:
            print(f"[ERREUR] {fname} : {e} -> ignoré")
            warnings_list.append({"file": fname, "error": str(e)})

    if warnings_out and warnings_list:
        with open(warnings_out, "w", newline="", encoding="utf-8") as wf:
            writer = csv.DictWriter(wf, fieldnames=["file", "missing", "error"])
            writer.writeheader()
            for w in warnings_list:
                writer.writerow({"file": w.get("file", ""), "missing": w.get("missing", ""), "error": w.get("error", "")})

    if not frames:
        raise ValueError("Aucun fichier n'a pu être chargé correctement.")
    return pd.concat(frames, ignore_index=True)

def clean_feature(series: pd.Series) -> np.ndarray:
    data = series.replace([np.inf, -np.inf], np.nan).dropna()
    data = data[data >= 0]
    return data.values.reshape(-1, 1)

def subsample(data: np.ndarray, max_samples: int = DEFAULT_MAX_SAMPLES, random_state: int = DEFAULT_RANDOM_STATE) -> np.ndarray:
    if len(data) <= max_samples:
        return data
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(data), size=max_samples, replace=False)
    return data[idx]

def silverman_bandwidth(data_log: np.ndarray) -> float:
    n = len(data_log)
    std = data_log.std()
    iqr = np.subtract(*np.percentile(data_log, [75, 25]))
    sigma = min(std, iqr / 1.34) if iqr > 0 else std
    sigma = max(sigma, 1e-6)
    bw = 0.9 * sigma * n ** (-1 / 5)
    return max(bw, 1e-3)

def fit_kde_log(data_log: np.ndarray) -> KernelDensity:
    sample = subsample(data_log)
    bw = silverman_bandwidth(sample)
    kde = KernelDensity(kernel="gaussian", bandwidth=bw)
    kde.fit(sample)
    return kde

def fit_gmm_log(data_log: np.ndarray, k_max: int = DEFAULT_K_MAX):
    sample = subsample(data_log)
    bics, models = [], []
    n_unique = len(np.unique(sample))
    k_max_eff = max(1, min(k_max, n_unique - 1)) if n_unique > 1 else 1
    for k in range(1, k_max_eff + 1):
        gmm = GaussianMixture(n_components=k, random_state=DEFAULT_RANDOM_STATE, reg_covar=1e-6)
        gmm.fit(sample)
        bics.append(gmm.bic(sample))
        models.append(gmm)
    best_k = int(np.argmin(bics)) + 1
    return models[best_k - 1], best_k, bics

def normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    std = max(std, 1e-9)
    return (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

# ----------------------------------------------------------------------
# FIGURES
# ----------------------------------------------------------------------
def plot_models(feature: str, result: dict, suffix: str, fig_dir: str, n_points: int):
    data_log = result["data_log"]
    x = np.linspace(data_log.min(), data_log.max(), n_points).reshape(-1, 1)
    normal_y = normal_pdf(x.flatten(), result["mean_log"], result["std_log"])
    kde_y = np.exp(result["kde"].score_samples(x))
    gmm_y = np.exp(result["gmm"].score_samples(x))
    plt.figure(figsize=(8, 5))
    sns.histplot(data_log.flatten(), bins=50, stat="density", color="skyblue", alpha=0.4, label="Données (log1p)")
    plt.plot(x, normal_y, "r--", label="Normale")
    plt.plot(x, kde_y, "g-", label="KDE")
    plt.plot(x, gmm_y, "m-", label=f"GMM ({result['best_k']} comp.)")
    plt.title(f"{feature} ({suffix}) — échelle log1p")
    plt.xlabel("log(1 + valeur)")
    plt.ylabel("Densité")
    plt.legend()
    plt.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(f"{fig_dir}/{safe_filename(feature)}_{suffix}_models.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_benign_vs_attack(feature: str, res_benign: dict, res_attack: dict, fig_dir: str, n_points: int):
    lo = min(res_benign["data_log"].min(), res_attack["data_log"].min())
    hi = max(res_benign["data_log"].max(), res_attack["data_log"].max())
    x = np.linspace(lo, hi, n_points).reshape(-1, 1)
    kde_b = np.exp(res_benign["kde"].score_samples(x))
    kde_a = np.exp(res_attack["kde"].score_samples(x))
    plt.figure(figsize=(8, 5))
    plt.plot(x, kde_b, label="BENIGN", color="tab:blue")
    plt.fill_between(x.flatten(), kde_b, alpha=0.2, color="tab:blue")
    plt.plot(x, kde_a, label="Attaque", color="tab:red")
    plt.fill_between(x.flatten(), kde_a, alpha=0.2, color="tab:red")
    plt.title(f"{feature} — BENIGN vs Attaque (KDE, échelle log1p)")
    plt.xlabel("log(1 + valeur)")
    plt.ylabel("Densité")
    plt.legend()
    plt.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(f"{fig_dir}/{safe_filename(feature)}_benign_vs_attack.png", dpi=150, bbox_inches="tight")
    plt.close()

# ----------------------------------------------------------------------
# ANALYSE D'UNE FEATURE POUR UNE CLASSE DONNEE
# ----------------------------------------------------------------------
def analyze_class(data: np.ndarray, k_max: int, random_state: int):
    if len(data) < 5:
        return None
    data_log = np.log1p(data)
    mean_log, std_log = data_log.mean(), data_log.std()
    if std_log <= 0 or np.isnan(std_log):
        return None
    kde = fit_kde_log(data_log)
    gmm, best_k, bics = fit_gmm_log(data_log, k_max=k_max)
    return {
        "data": data,
        "data_log": data_log,
        "mean_log": mean_log,
        "std_log": std_log,
        "mean_raw": float(np.expm1(mean_log)),
        "kde": kde,
        "gmm": gmm,
        "best_k": best_k,
        "bics": bics,
    }

# ----------------------------------------------------------------------
# REPORT
# ----------------------------------------------------------------------
def write_report(tableau: pd.DataFrame, out_path: str = "rapport.md"):
    lines = []
    lines.append("# Rapport d'analyse statistique — CICIDS2017\n")
    lines.append("## Pourquoi la loi normale est une approximation insuffisante\n\n"
                 "Les features de flux réseau sont fortement asymétriques, bornées à zéro, et présentent une queue lourde.\n")
    lines.append("## Pourquoi KDE et GMM sont plus adaptés\n\n"
                 "Le KDE est non paramétrique; le GMM est génératif et permet d'échantillonner du trafic synthétique réaliste.\n")
    lines.append("## Tableau comparatif (résumé)\n\n")
    lines.append(tableau.to_markdown(index=False))
    lines.append("\n\n## Limites identifiées\n\n"
                 "- Le GMM peut sous-estimer les queues extrêmement lourdes.\n"
                 "- Le log1p facilite le fit mais complique l'interprétation directe des paramètres.\n")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    parser.add_argument("--label-col", default=LABEL_COL)
    parser.add_argument("--fig-dir", default="figs")
    parser.add_argument("--k-max", type=int, default=DEFAULT_K_MAX)
    parser.add_argument("--n-points", type=int, default=DEFAULT_N_POINTS)
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--warnings-out", default="warnings_stats_features.csv")
    args = parser.parse_args()

    # mettre à jour les valeurs globales sans déclarer de variables locales
    globals()['DEFAULT_MAX_SAMPLES'] = args.max_samples
    globals()['DEFAULT_RANDOM_STATE'] = args.random_state

    df = load_data(args.data_dir, args.features, args.label_col, warnings_out=args.warnings_out)
    if args.label_col not in df.columns:
        raise ValueError(f"Colonne '{args.label_col}' absente du fichier.")

    rows = []
    for feature in args.features:
        if feature not in df.columns:
            print(f"[SKIP] Feature absente : {feature}")
            continue
        benign_raw = clean_feature(df.loc[df[args.label_col] == BENIGN_VALUE, feature])
        attack_raw = clean_feature(df.loc[df[args.label_col] != BENIGN_VALUE, feature])
        res_benign = analyze_class(benign_raw, k_max=args.k_max, random_state=args.random_state)
        res_attack = analyze_class(attack_raw, k_max=args.k_max, random_state=args.random_state)
        if res_benign is None or res_attack is None:
            print(f"[SKIP] Pas assez de données ou variance nulle pour : {feature}")
            continue
        plot_models(feature, res_benign, "benign", args.fig_dir, args.n_points)
        plot_models(feature, res_attack, "attack", args.fig_dir, args.n_points)
        plot_benign_vs_attack(feature, res_benign, res_attack, args.fig_dir, args.n_points)
        gmm_b, gmm_a = res_benign["gmm"], res_attack["gmm"]
        rows.append({
            "feature": feature,
            "n_benign": len(benign_raw),
            "n_attack": len(attack_raw),
            "mu_log_benign": round(res_benign["mean_log"], 4),
            "sigma_log_benign": round(res_benign["std_log"], 4),
            "mu_log_attack": round(res_attack["mean_log"], 4),
            "sigma_log_attack": round(res_attack["std_log"], 4),
            "k_gmm_benign": res_benign["best_k"],
            "k_gmm_attack": res_attack["best_k"],
            "gmm_weights_benign": np.round(gmm_b.weights_, 3).tolist(),
            "gmm_means_log_benign": np.round(gmm_b.means_.flatten(), 3).tolist(),
            "gmm_weights_attack": np.round(gmm_a.weights_, 3).tolist(),
            "gmm_means_log_attack": np.round(gmm_a.means_.flatten(), 3).tolist(),
        })
        print(f"[OK] {feature} : k_benign={res_benign['best_k']}, k_attack={res_attack['best_k']}")

    tableau = pd.DataFrame(rows)
    tableau.to_csv("tableau_comparatif.csv", index=False)
    print("\nTableau comparatif exporté -> tableau_comparatif.csv")
    write_report(tableau)
    print("Rapport exporté -> rapport.md")

if __name__ == "__main__":
    main()
