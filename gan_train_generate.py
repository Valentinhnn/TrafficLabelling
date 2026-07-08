#!/usr/bin/env python3
"""
gan_train_generate.py
======================
Génération de trafic réseau synthétique (CICIDS2017) par CTGAN.
Pendant direct de synthetic_generation.py (GMM), avec la même logique
de chargement, preprocessing, validation et export.

Prérequis :
    pip install ctgan torch

Usage exemple :
    python gan_train_generate.py --data-dir data --features "Flow Duration" "Packet Length Mean" \
        --n-synthetic 5000 --epochs 300 --random-state 42

Sorties :
    flux_synthetiques_gan.csv       -> flux synthétiques générés
    ctgan_model.pkl                 -> modèle entraîné (rechargeable)
    comparaison_gmm_vs_gan.csv      -> tableau comparatif si GMM disponible
"""
from __future__ import annotations

import os
import sys
import argparse
import pickle
import warnings
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Génération synthétique CTGAN — CICIDS2017")
    p.add_argument("--data-dir", default="data",
                   help="Dossier contenant les CSV CICIDS2017 (défaut: data)")
    p.add_argument("--features", nargs="+",
                   default=["Flow Duration", "Packet Length Mean"],
                   help="Features continues à modéliser")
    p.add_argument("--label-col", default="Label",
                   help="Colonne label (défaut: Label)")
    p.add_argument("--n-synthetic", type=int, default=5000,
                   help="Nombre de flux synthétiques à générer (défaut: 5000)")
    p.add_argument("--epochs", type=int, default=300,
                   help="Epochs d'entraînement CTGAN (défaut: 300, min recommandé: 200)")
    p.add_argument("--batch-size", type=int, default=500,
                   help="Taille de batch (défaut: 500)")
    p.add_argument("--fit-clip-quantile", type=float, default=0.995,
                   help="Quantile de clip des données réelles avant fit (défaut: 0.995)")
    p.add_argument("--clip-max-output", type=float, default=None,
                   help="Clip absolu sur les sorties après génération (ex: 118000000 pour Flow Duration)")
    p.add_argument("--model-path", default="ctgan_model.pkl",
                   help="Chemin de sauvegarde du modèle (défaut: ctgan_model.pkl)")
    p.add_argument("--force-retrain", action="store_true",
                   help="Forcer le réentraînement même si modèle existant")
    p.add_argument("--random-state", type=int, default=42,
                   help="Seed pour reproductibilité (défaut: 42)")
    p.add_argument("--gmm-csv", default="flux_synthetiques_benign.csv",
                   help="CSV GMM existant pour comparaison (défaut: flux_synthetiques_benign.csv)")
    p.add_argument("--max-train-rows", type=int, default=100000,
                   help="Sous-échantillonnage max pour entraînement CTGAN (défaut: 100000). "
                        "CTGAN est lent sur 2M de lignes — 100k suffit largement.")
    return p.parse_args()


# ----------------------------------------------------------------------
# CHARGEMENT (identique à synthetic_generation.py)
# ----------------------------------------------------------------------

def normalize_cols(cols: List[str]) -> List[str]:
    out = []
    for c in cols:
        s = str(c).strip().lstrip("\ufeff").replace("\t", " ")
        s = " ".join(s.split())
        out.append(s)
    return out


def load_all_csvs(data_dir: str) -> pd.DataFrame:
    if not os.path.isdir(data_dir):
        print(f"[ERREUR] Dossier introuvable: {data_dir}", file=sys.stderr)
        sys.exit(1)

    csv_files = sorted(f for f in os.listdir(data_dir) if f.lower().endswith(".csv"))
    if not csv_files:
        print(f"[ERREUR] Aucun CSV dans {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] {len(csv_files)} fichiers CSV trouvés dans '{data_dir}', chargement...")
    frames = []
    for fname in csv_files:
        fpath = os.path.join(data_dir, fname)
        try:
            df = pd.read_csv(fpath, low_memory=False)
            df.columns = normalize_cols(df.columns.tolist())
            frames.append(df)
            print(f"[OK]   {fname} : {len(df)} lignes")
        except Exception as e:
            print(f"[ERREUR] {fname} : {e} -> ignoré")

    if not frames:
        print("[ERREUR] Aucun fichier lisible.", file=sys.stderr)
        sys.exit(1)

    full = pd.concat(frames, ignore_index=True, sort=False)
    print(f"[INFO] Total : {len(full)} lignes\n")
    return full


# ----------------------------------------------------------------------
# PREPROCESSING
# ----------------------------------------------------------------------

def preprocess(df: pd.DataFrame, features: List[str],
               fit_clip_quantile: float) -> tuple[pd.DataFrame, dict]:
    """
    1. Clip des valeurs extrêmes (queues) au quantile demandé
    2. log1p pour atténuer l'asymétrie (même logique que GMM)
    Retourne le DataFrame transformé + les caps utilisés (pour inversion).
    """
    data = df[features].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data[(data >= 0).all(axis=1)]

    caps = {}
    for col in features:
        cap = data[col].quantile(fit_clip_quantile)
        n_clipped = int((data[col] > cap).sum())
        data[col] = data[col].clip(upper=cap)
        caps[col] = float(cap)
        print(f"[FIT-CLIP] {col} : {n_clipped} valeurs > q{fit_clip_quantile} ({cap:.2f}) clippées")

    for col in features:
        data[col] = np.log1p(data[col])

    return data, caps


def postprocess(synth: pd.DataFrame, features: List[str],
                caps: dict, clip_max_output: float | None) -> pd.DataFrame:
    """
    Inversion : expm1 -> clip lower=0 -> clip_max_output si fourni.
    """
    out = synth[features].copy()
    SAFE_EXP_MAX = 700.0
    for col in features:
        out[col] = np.minimum(out[col].values, SAFE_EXP_MAX)
        with np.errstate(over="ignore", invalid="ignore"):
            out[col] = np.expm1(out[col])
        out[col] = out[col].clip(lower=0.0)
        if clip_max_output is not None:
            out[col] = out[col].clip(upper=clip_max_output)
    return out


# ----------------------------------------------------------------------
# ENTRAÎNEMENT CTGAN
# ----------------------------------------------------------------------

def train_ctgan(data: pd.DataFrame, features: List[str],
                epochs: int, batch_size: int,
                random_state: int, max_rows: int):
    try:
        from ctgan import CTGAN
    except ImportError:
        print("[ERREUR] ctgan non installé. Lancer : pip install ctgan", file=sys.stderr)
        sys.exit(1)

    # Sous-échantillonnage si trop de lignes (CTGAN est lent sur >100k)
    if len(data) > max_rows:
        data = data.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
        print(f"[INFO] Sous-échantillonnage : {max_rows} lignes utilisées pour l'entraînement")

    print(f"\n[CTGAN] Entraînement sur {len(data)} lignes, {epochs} epochs, batch={batch_size}")
    print(f"[CTGAN] Features : {features}\n")

    ctgan = CTGAN(
        epochs=epochs,
        batch_size=batch_size,
        verbose=True,
        cuda=_cuda_available(),
    )
    # discrete_columns=[] car toutes les features ici sont continues
    ctgan.fit(data[features], discrete_columns=[])
    return ctgan


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ----------------------------------------------------------------------
# VALIDATION (identique à synthetic_generation.py)
# ----------------------------------------------------------------------

def compute_quantiles(series: pd.Series, qs=(0.01, 0.05, 0.5, 0.95, 0.99)) -> dict:
    return {float(q): float(series.quantile(q)) for q in qs}


def validate(real: pd.DataFrame, synth: pd.DataFrame, features: List[str], label: str = "GAN"):
    print(f"\n{'='*60}")
    print(f"  Validation univariée — {label}")
    print(f"{'='*60}")
    results = []
    for col in features:
        if col not in real.columns or col not in synth.columns:
            print(f"{col} : colonne manquante")
            continue
        r = real[col].dropna()
        s = synth[col].dropna()
        if len(r) < 2 or len(s) < 2:
            print(f"{col} : pas assez de données")
            continue

        ks_stat, ks_p = ks_2samp(r, s)
        w = wasserstein_distance(r, s)
        med = float(r.median())
        iqr = float(r.quantile(0.75) - r.quantile(0.25))
        w_med = w / med if med != 0 else float("nan")
        w_iqr = w / iqr if iqr != 0 else float("nan")

        print(f"\n{col}")
        print(f"  KS stat       = {ks_stat:.4f}  (p={ks_p:.4g})")
        print(f"  Wasserstein   = {w:.6g}")
        print(f"  W/median      = {w_med:.4f}")
        print(f"  W/IQR         = {w_iqr:.4f}")
        print(f"  Quantiles réel : {compute_quantiles(r)}")
        print(f"  Quantiles synth: {compute_quantiles(s)}")

        results.append({
            "feature": col,
            "KS": round(ks_stat, 4),
            "Wasserstein": round(w, 2),
            "W/median": round(w_med, 4),
            "W/IQR": round(w_iqr, 4),
        })

    return pd.DataFrame(results)


def validate_correlations(real: pd.DataFrame, synth: pd.DataFrame, features: List[str]):
    common = [c for c in features if c in real.columns and c in synth.columns]
    if not common:
        return
    print(f"\n--- Corrélations ---")
    print("Corr réel:")
    print(real[common].corr().to_string())
    print("\nCorr synth (GAN):")
    print(synth[common].corr().to_string())


# ----------------------------------------------------------------------
# COMPARAISON GMM vs GAN
# ----------------------------------------------------------------------

def compare_gmm_vs_gan(real: pd.DataFrame, synth_gmm: pd.DataFrame,
                       synth_gan: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print("  Comparaison GMM vs GAN")
    print(f"{'='*60}")
    rows = []
    for col in features:
        if col not in real.columns:
            continue
        r = real[col].dropna()
        row = {"feature": col}
        for label, synth in [("GMM", synth_gmm), ("GAN", synth_gan)]:
            if col not in synth.columns:
                continue
            s = synth[col].dropna()
            ks_stat, _ = ks_2samp(r, s)
            w = wasserstein_distance(r, s)
            iqr = float(r.quantile(0.75) - r.quantile(0.25))
            w_iqr = w / iqr if iqr != 0 else float("nan")
            corr_real = float(real[features].corr().iloc[0, 1]) if len(features) == 2 else float("nan")
            corr_synth = float(synth[features].corr().iloc[0, 1]) if len(features) == 2 else float("nan")
            row[f"KS_{label}"] = round(ks_stat, 4)
            row[f"W/IQR_{label}"] = round(w_iqr, 4)
            row[f"Corr_{label}"] = round(corr_synth, 4)
        row["Corr_réel"] = round(corr_real, 4) if len(features) == 2 else float("nan")
        rows.append(row)

    df_cmp = pd.DataFrame(rows)

    # Verdict par feature
    print("\n" + df_cmp.to_string(index=False))
    print("\n--- Verdict ---")
    for _, row in df_cmp.iterrows():
        feat = row["feature"]
        if "W/IQR_GMM" in row and "W/IQR_GAN" in row:
            gmm_better = row["W/IQR_GMM"] < row["W/IQR_GAN"]
            winner = "GMM" if gmm_better else "GAN"
            print(f"  {feat} : W/IQR GMM={row.get('W/IQR_GMM','?')} | GAN={row.get('W/IQR_GAN','?')} -> {winner} meilleur")

    # Critère global (rapport de stage)
    if "W/IQR_GAN" in df_cmp.columns and "Corr_GAN" in df_cmp.columns:
        w_iqr_gan = df_cmp["W/IQR_GAN"].mean()
        corr_gan = df_cmp["Corr_GAN"].mean()
        corr_reel = df_cmp["Corr_réel"].mean() if "Corr_réel" in df_cmp.columns else 0.32
        print(f"\n  Critère GAN supérieur : W/IQR < 2.0 ET |Corr_GAN - Corr_réel| < |Corr_GMM - Corr_réel|")
        print(f"  W/IQR moyen GAN = {w_iqr_gan:.4f} {'✅' if w_iqr_gan < 2.0 else '❌'}")
        corr_gmm = df_cmp["Corr_GMM"].mean() if "Corr_GMM" in df_cmp.columns else float("nan")
        print(f"  |ΔCorr| GMM={abs(corr_gmm - corr_reel):.4f} | GAN={abs(corr_gan - corr_reel):.4f} "
              f"{'-> GAN meilleur sur corrélation ✅' if abs(corr_gan - corr_reel) < abs(corr_gmm - corr_reel) else '-> GMM meilleur sur corrélation'}")

    return df_cmp


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    args = parse_args()
    np.random.seed(args.random_state)

    # --- Chargement ---
    full = load_all_csvs(args.data_dir)
    full.columns = normalize_cols(full.columns.tolist())

    features = normalize_cols(args.features)
    missing = [c for c in features if c not in full.columns]
    if missing:
        print(f"[ERREUR] Colonnes manquantes: {missing}", file=sys.stderr)
        print("Colonnes disponibles (extrait):", full.columns.tolist()[:40])
        sys.exit(1)

    label_col = args.label_col.strip()
    if label_col in full.columns:
        full[label_col] = full[label_col].astype(str).str.strip()
        benign_df = full[full[label_col].str.upper().str.contains("BENIGN", na=False)]
    else:
        print(f"[WARN] Colonne label '{label_col}' introuvable — tout le dataset traité comme BENIGN")
        benign_df = full.copy()

    print(f"BENIGN : {len(benign_df)} lignes\n")

    # --- Preprocessing ---
    data_transformed, caps = preprocess(benign_df, features, args.fit_clip_quantile)

    # --- Entraînement ou chargement ---
    if not args.force_retrain and os.path.exists(args.model_path):
        print(f"\n[INFO] Modèle existant chargé depuis {args.model_path} (--force-retrain pour réentraîner)")
        with open(args.model_path, "rb") as f:
            ctgan = pickle.load(f)
    else:
        ctgan = train_ctgan(
            data_transformed, features,
            epochs=args.epochs,
            batch_size=args.batch_size,
            random_state=args.random_state,
            max_rows=args.max_train_rows,
        )
        with open(args.model_path, "wb") as f:
            pickle.dump(ctgan, f)
        print(f"\n[INFO] Modèle sauvegardé -> {args.model_path}")

    # --- Génération ---
    print(f"\n[INFO] Génération de {args.n_synthetic} flux synthétiques...")
    synth_log = ctgan.sample(args.n_synthetic)
    synth_real = postprocess(synth_log, features, caps, args.clip_max_output)

    out_path = "flux_synthetiques_gan.csv"
    synth_real.to_csv(out_path, index=False)
    print(f"[INFO] Synthétique exporté -> {out_path}")

    # --- Résumés ---
    print("\n-- Résumé synthétique (GAN) --")
    print(synth_real[features].describe().transpose().to_string())
    print("\n-- Résumé réel (BENIGN) --")
    print(benign_df[features].describe().transpose().to_string())

    # --- Validation ---
    val_results = validate(benign_df, synth_real, features, label="GAN")
    validate_correlations(benign_df, synth_real, features)
    val_results.to_csv("validation_gan.csv", index=False)
    print(f"\n[INFO] Résultats de validation exportés -> validation_gan.csv")

    # --- Comparaison GMM vs GAN ---
    gmm_path = args.gmm_csv
    if os.path.exists(gmm_path):
        print(f"\n[INFO] Chargement GMM depuis {gmm_path} pour comparaison...")
        synth_gmm = pd.read_csv(gmm_path, low_memory=False)
        synth_gmm.columns = normalize_cols(synth_gmm.columns.tolist())
        df_cmp = compare_gmm_vs_gan(benign_df, synth_gmm, synth_real, features)
        df_cmp.to_csv("comparaison_gmm_vs_gan.csv", index=False)
        print(f"[INFO] Tableau comparatif exporté -> comparaison_gmm_vs_gan.csv")
    else:
        print(f"\n[WARN] {gmm_path} introuvable — comparaison GMM vs GAN ignorée. "
              f"Lancer d'abord synthetic_generation.py.")

    print("\nTerminé.")


if __name__ == "__main__":
    main()