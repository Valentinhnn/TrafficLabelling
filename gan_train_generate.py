from __future__ import annotations

import os
import sys
import inspect
import argparse
import pickle
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

warnings.filterwarnings("ignore")


# Parses CLI arguments for this script.
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
                   help="Clip absolu sur les sorties après génération")
    p.add_argument("--model-path", default="ctgan_model.pkl",
                   help="Chemin de sauvegarde du modèle")
    p.add_argument("--force-retrain", action="store_true",
                   help="Forcer le réentraînement même si modèle existant")
    p.add_argument("--random-state", type=int, default=42,
                   help="Seed pour reproductibilité")
    p.add_argument("--gmm-csv", default="flux_synthetiques_benign.csv",
                   help="CSV GMM existant pour comparaison")
    p.add_argument("--max-train-rows", type=int, default=100000,
                   help="Sous-échantillonnage max pour CTGAN")
    return p.parse_args()


# Cleans column names: strips BOM/whitespace and collapses tabs/multiple spaces.
def normalize_cols(cols: List[str]) -> List[str]:
    out = []
    for c in cols:
        s = str(c).strip().lstrip("\ufeff").replace("\t", " ")
        s = " ".join(s.split())
        out.append(s)
    return out


# Loads and concatenates all CICIDS2017 CSVs found in a directory.
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
            df = pd.read_csv(fpath, low_memory=False, encoding="latin1")
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


# Cleans inf/NaN, clips outliers to quantile caps, and log1p-transforms features
# so CTGAN trains on a more Gaussian-like distribution.
def preprocess(df: pd.DataFrame, features: List[str],
               fit_clip_quantile: float) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    data = df[features].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data[(data >= 0).all(axis=1)]

    real_clean = data.copy()

    caps = {}
    for col in features:
        cap = data[col].quantile(fit_clip_quantile)
        n_clipped = int((data[col] > cap).sum())
        data[col] = data[col].clip(upper=cap)
        caps[col] = float(cap)
        print(f"[FIT-CLIP] {col} : {n_clipped} valeurs > q{fit_clip_quantile} ({cap:.2f}) clippées")

    for col in features:
        data[col] = np.log1p(data[col])

    return data, caps, real_clean


# Reverses the log1p transform on CTGAN's output and clips back to the caps.
def postprocess(synth: pd.DataFrame, features: List[str],
                caps: dict, clip_max_output: float | None) -> pd.DataFrame:
    out = synth[features].copy()
    SAFE_EXP_MAX = 700.0
    for col in features:
        out[col] = np.minimum(out[col].values, SAFE_EXP_MAX)
        with np.errstate(over="ignore", invalid="ignore"):
            out[col] = np.expm1(out[col])
        out[col] = out[col].clip(lower=0.0)
        if col in caps:
            out[col] = out[col].clip(upper=caps[col])
        if clip_max_output is not None:
            out[col] = out[col].clip(upper=clip_max_output)
    return out


# Sets random seeds (python/numpy/torch) for reproducibility.
def _seed_everything(random_state: int):
    np.random.seed(random_state)
    try:
        import torch
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)
    except ImportError:
        pass


# Trains a CTGAN model on the (log1p-transformed) real feature data.
def train_ctgan(data: pd.DataFrame, features: List[str],
                epochs: int, batch_size: int,
                random_state: int, max_rows: int):

    try:
        from ctgan import CTGAN
    except ImportError:
        print("[ERREUR] Le module 'ctgan' est introuvable.")
        print("[INFO] Installe-le dans ton venv : pip install ctgan")
        raise

    _seed_everything(random_state)

    if len(data) > max_rows:
        data = data.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
        print(f"[INFO] Sous-échantillonnage : {max_rows} lignes utilisées pour l'entraînement")

    print(f"\n[CTGAN] Entraînement sur {len(data)} lignes, {epochs} epochs, batch={batch_size}")
    print(f"[CTGAN] Features : {features}\n")

    ctgan_kwargs = dict(
        epochs=epochs,
        batch_size=batch_size,
        verbose=True,
        cuda=_cuda_available(),
    )
    sig_params = inspect.signature(CTGAN.__init__).parameters
    if "random_state" in sig_params:
        ctgan_kwargs["random_state"] = random_state
    elif "random_seed" in sig_params:
        ctgan_kwargs["random_seed"] = random_state
    else:
        print("[WARN] Cette version de ctgan n'expose pas de parametre de seed natif "
              "-> reproductibilite partielle (seeding numpy/torch global uniquement)")

    ctgan = CTGAN(**ctgan_kwargs)
    ctgan.fit(data[features], discrete_columns=[])
    return ctgan


# Checks whether a CUDA GPU is available for torch.
def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# Loads a previously saved CTGAN model + its quantile caps, if the file exists.
def load_cached_ctgan(model_path: str, features: List[str]) -> Tuple[Optional[object], Optional[dict], Optional[str]]:
    try:
        with open(model_path, "rb") as f:
            payload = pickle.load(f)
    except Exception as e:
        return None, None, f"lecture impossible ({e})"

    if not isinstance(payload, dict) or "ctgan" not in payload:
        return None, None, "format de modele incompatible (ancien format sans metadonnees)"

    if payload.get("features") != features:
        return None, None, (f"features incompatibles (modele: {payload.get('features')}, "
                             f"demande: {features})")

    return payload["ctgan"], payload.get("caps", {}), None


# Computes a few quantiles of a series, used for clipping/validation.
def compute_quantiles(series: pd.Series, qs=(0.01, 0.05, 0.5, 0.95, 0.99)) -> dict:
    return {float(q): float(series.quantile(q)) for q in qs}


# Prints a real-vs-synthetic comparison (KS test, Wasserstein distance, quantiles) per feature.
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


# Compares real vs synthetic feature correlation matrices (Frobenius distance).
def validate_correlations(real: pd.DataFrame, synth: pd.DataFrame, features: List[str]):
    common = [c for c in features if c in real.columns and c in synth.columns]
    if not common:
        return
    print(f"\n--- Corrélations ---")
    print("Corr réel:")
    print(real[common].corr().to_string())
    print("\nCorr synth (GAN):")
    print(synth[common].corr().to_string())


# Compares a GMM baseline against CTGAN on distribution similarity metrics.
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
        iqr = float(r.quantile(0.75) - r.quantile(0.25))
        med = float(r.median())
        use_median_fallback = (iqr == 0)
        row = {"feature": col, "metric": "W/median" if use_median_fallback else "W/IQR"}
        for label, synth in [("GMM", synth_gmm), ("GAN", synth_gan)]:
            if col not in synth.columns:
                continue
            s = synth[col].dropna()
            ks_stat, _ = ks_2samp(r, s)
            w = wasserstein_distance(r, s)
            if use_median_fallback:
                score = w / med if med != 0 else float("nan")
            else:
                score = w / iqr
            row[f"KS_{label}"] = round(ks_stat, 4)
            row[f"W_{label}"] = round(w, 2)
            row[f"score_{label}"] = round(score, 4)
        rows.append(row)

    df_cmp = pd.DataFrame(rows)
    print("\n" + df_cmp.to_string(index=False))

    print("\n--- Verdict ---")
    for _, row in df_cmp.iterrows():
        feat = row["feature"]
        metric = row.get("metric", "W/IQR")
        if "score_GMM" in row and "score_GAN" in row:
            v_gmm, v_gan = row["score_GMM"], row["score_GAN"]
            if pd.isna(v_gmm) and pd.isna(v_gan):
                print(f"  {feat} ({metric}) : GMM=nan | GAN=nan -> indéterminé")
            elif pd.isna(v_gmm):
                print(f"  {feat} ({metric}) : GMM=nan | GAN={v_gan} -> GAN meilleur (GMM invalide)")
            elif pd.isna(v_gan):
                print(f"  {feat} ({metric}) : GMM={v_gmm} | GAN=nan -> GMM meilleur (GAN invalide)")
            else:
                winner = "GMM" if v_gmm < v_gan else "GAN"
                print(f"  {feat} ({metric}) : GMM={v_gmm} | GAN={v_gan} -> {winner} meilleur")

    return df_cmp


# Compares GMM's and CTGAN's correlation matrices against the real one, side by side.
def compare_correlation_matrices(real: pd.DataFrame, synth_gmm: pd.DataFrame,
                                 synth_gan: pd.DataFrame, features: List[str]) -> Optional[pd.DataFrame]:
    if len(features) < 2:
        return None

    corr_real = real[features].corr()
    have_gmm = all(f in synth_gmm.columns for f in features)
    have_gan = all(f in synth_gan.columns for f in features)
    corr_gmm = synth_gmm[features].corr() if have_gmm else None
    corr_gan = synth_gan[features].corr() if have_gan else None

    print("\n--- Matrices de corrélation (réel / GMM / GAN) ---")
    print("Réel:")
    print(corr_real.round(4).to_string())
    if corr_gmm is not None:
        print("\nGMM:")
        print(corr_gmm.round(4).to_string())
    else:
        print("\nGMM: non disponible (features manquantes)")
    if corr_gan is not None:
        print("\nGAN:")
        print(corr_gan.round(4).to_string())
    else:
        print("\nGAN: non disponible (features manquantes)")

    rows = []
    for i, fi in enumerate(features):
        for j, fj in enumerate(features):
            if j <= i:
                continue
            row = {
                "feature_i": fi, "feature_j": fj,
                "corr_reel": round(float(corr_real.loc[fi, fj]), 4),
            }
            if corr_gmm is not None:
                row["corr_gmm"] = round(float(corr_gmm.loc[fi, fj]), 4)
            if corr_gan is not None:
                row["corr_gan"] = round(float(corr_gan.loc[fi, fj]), 4)
            rows.append(row)

    return pd.DataFrame(rows)


# Entry point: load real data, preprocess, train (or load) CTGAN, generate
# synthetic data, postprocess, validate, and save results.
def main():
    args = parse_args()
    np.random.seed(args.random_state)

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

    data_transformed, caps, real_clean = preprocess(benign_df, features, args.fit_clip_quantile)

    ctgan = None
    cached_caps = None
    if not args.force_retrain and os.path.exists(args.model_path):
        ctgan, cached_caps, err = load_cached_ctgan(args.model_path, features)
        if ctgan is not None:
            print(f"\n[INFO] Modèle existant chargé depuis {args.model_path} "
                  f"(--force-retrain pour réentraîner)")
            if cached_caps:
                caps = cached_caps
        else:
            print(f"\n[WARN] Modèle en cache ignoré : {err}")
            print("[INFO] Réentraînement du CTGAN...")

    if ctgan is None:
        ctgan = train_ctgan(
            data_transformed, features,
            epochs=args.epochs,
            batch_size=args.batch_size,
            random_state=args.random_state,
            max_rows=args.max_train_rows,
        )
        payload = {
            "ctgan": ctgan,
            "features": features,
            "caps": caps,
            "fit_clip_quantile": args.fit_clip_quantile,
            "random_state": args.random_state,
        }
        with open(args.model_path, "wb") as f:
            pickle.dump(payload, f)
        print(f"\n[INFO] Modèle sauvegardé -> {args.model_path}")

    print(f"\n[INFO] Génération de {args.n_synthetic} flux synthétiques...")
    try:
        synth_log = ctgan.sample(args.n_synthetic)
    except Exception as e:
        print(f"[ERREUR] Impossible de générer les données GAN : {e}")
        print("[INFO] Essaie de réentraîner avec --force-retrain")
        sys.exit(1)

    synth_real = postprocess(synth_log, features, caps, args.clip_max_output)

    out_path = "flux_synthetiques_gan.csv"
    synth_real.to_csv(out_path, index=False)
    print(f"[INFO] Synthétique exporté -> {out_path}")

    print("\n-- Résumé synthétique (GAN) --")
    print(synth_real[features].describe().transpose().to_string())
    print("\n-- Résumé réel (nettoyé, non clippé) --")
    print(real_clean[features].describe().transpose().to_string())

    val_results = validate(real_clean, synth_real, features, label="GAN")
    validate_correlations(real_clean, synth_real, features)
    val_results.to_csv("validation_gan.csv", index=False)
    print(f"\n[INFO] Résultats de validation exportés -> validation_gan.csv")

    gmm_path = args.gmm_csv
    if os.path.exists(gmm_path):
        print(f"\n[INFO] Chargement GMM depuis {gmm_path} pour comparaison...")
        try:
            synth_gmm = pd.read_csv(gmm_path, low_memory=False)
        except Exception as e:
            print(f"[WARN] Impossible de lire le CSV GMM ({e}).")
            print("[WARN] Comparaison GMM vs GAN ignorée.")
            print("\nTerminé.")
            return

        synth_gmm.columns = normalize_cols(synth_gmm.columns.tolist())
        df_cmp = compare_gmm_vs_gan(real_clean, synth_gmm, synth_real, features)
        df_cmp.to_csv("comparaison_gmm_vs_gan.csv", index=False)
        print(f"[INFO] Tableau comparatif exporté -> comparaison_gmm_vs_gan.csv")

        df_corr_cmp = compare_correlation_matrices(real_clean, synth_gmm, synth_real, features)
        if df_corr_cmp is not None and not df_corr_cmp.empty:
            df_corr_cmp.to_csv("correlation_matrices_comparaison.csv", index=False)
            print(f"[INFO] Comparaison des corrélations exportée -> correlation_matrices_comparaison.csv")
    else:
        print(f"\n[WARN] {gmm_path} introuvable — comparaison GMM vs GAN ignorée.")

    print("\nTerminé.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOP] Entraînement interrompu par l'utilisateur (Ctrl-C).")
        print("[STOP] Aucun fichier n'a été corrompu")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERREUR FATALE] {e}")
        sys.exit(1)