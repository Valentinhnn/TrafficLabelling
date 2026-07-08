#!/usr/bin/env python3
"""
synthetic_generation.py

Génération synthétique par GMM multivarié (version simplifiée et robuste).
Ce script :
- charge les CSV CICIDS2017
- filtre les flux BENIGN
- applique un prétraitement (log1p, clipping)
- fit un GMM multivarié
- corrige les composantes instables
- génère des flux synthétiques
- inverse la transformation log
- valide statistiquement les résultats
"""
from __future__ import annotations
import os
import argparse
import sys
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import joblib
from scipy.stats import ks_2samp, wasserstein_distance


# ------------------------------------------------------------
# PARSING DES ARGUMENTS DE LIGNE DE COMMANDE
# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Génération synthétique par GMM multivarié (simplifié)")

    # Dossier contenant les CSV CICIDS2017
    p.add_argument("--data-dir", default="data", help="Dossier contenant les CSV (défaut: data)")

    # Features multivariées à modéliser ensemble
    p.add_argument("--features-mv", nargs="+", default=["Flow Duration", "Packet Length Mean"],
                   help="Liste des features multivariées à modéliser")

    # Colonne du label (BENIGN / attaque)
    p.add_argument("--label-col", default="Label", help="Nom de la colonne label (défaut: Label)")

    # Nombre d'échantillons synthétiques à générer
    p.add_argument("--n-synthetic", type=int, default=5000, help="Nombre d'échantillons synthétiques à générer")

    # Nombre max de composantes GMM testées (BIC)
    p.add_argument("--k-max", type=int, default=10, help="Nombre max de composantes GMM à tester (BIC)")

    # Forcer le refit même si un modèle existe déjà
    p.add_argument("--force-refit", action="store_true", help="Forcer le refit même si modèle existant")

    # Chemin pour sauvegarder le modèle GMM
    p.add_argument("--model-path", default="gmm_mv_benign.joblib", help="Chemin pour sauvegarder le modèle GMM")

    # Seed pour reproductibilité
    p.add_argument("--random-state", type=int, default=0, help="Seed pour reproductibilité")

    # Régularisation des covariances (évite les matrices singulières)
    p.add_argument("--reg-covar", type=float, default=1e-6, help="Regularisation pour covariances du GMM")

    # Transformation log1p → expm1
    p.add_argument("--log-transform", action="store_true",
                   help="Appliquer log1p aux features avant fit et expm1 après génération")

    # Clip final des valeurs synthétiques
    p.add_argument("--clip-max", type=float, default=None,
                   help="Valeur maximale pour clipper les features après inversion (None = pas de clip)")

    # Type de covariance du GMM (full, diag, tied, spherical)
    p.add_argument("--covariance-type", default="full", choices=["full", "diag", "tied", "spherical"],
                   help="Type de covariance GMM (défaut: full)")

    # Clip des données réelles avant le fit (stabilise le GMM)
    p.add_argument("--fit-clip-quantile", type=float, default=None,
                   help="Quantile (ex 0.995) pour clipper les données réelles avant le fit GMM")

    # Variance max autorisée en log-espace (évite les composantes explosives)
    p.add_argument("--max-log-var", type=float, default=4.0,
                   help="Variance max autorisée en log-espace par composante GMM (défaut: 4.0)")

    return p.parse_args()


# ------------------------------------------------------------
# NORMALISATION DES NOMS DE COLONNES
# ------------------------------------------------------------
def normalize_cols(cols: List[str]) -> List[str]:
    out = []
    for c in cols:
        s = str(c)
        # Nettoyage des caractères parasites (BOM UTF-8, tabulations, espaces multiples)
        s = s.strip().lstrip("\ufeff").replace("\t", " ")
        s = " ".join(s.split())
        out.append(s)
    return out


# ------------------------------------------------------------
# LECTURE D'UN CSV AVEC NORMALISATION DES COLONNES
# ------------------------------------------------------------
def read_csv_simple(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = normalize_cols(df.columns.tolist())
    return df


# ------------------------------------------------------------
# TROUVER TOUS LES CSV DANS UN DOSSIER
# ------------------------------------------------------------
def find_csv_files(data_dir: str) -> List[str]:
    files = []
    for f in sorted(os.listdir(data_dir)):
        if f.lower().endswith(".csv"):
            files.append(os.path.join(data_dir, f))
    return files


# ------------------------------------------------------------
# SÉLECTION DU MEILLEUR GMM VIA LE BIC
# ------------------------------------------------------------
def select_best_gmm(X: np.ndarray, k_max: int, reg_covar: float, random_state: int,
                     covariance_type: str = "full") -> Tuple[object,int,List[float]]:

    # BIC minimal trouvé
    best_bic = float("inf")
    best_gmm = None
    bics = []

    # On teste k = 1 à k_max composantes
    for k in range(1, max(1, k_max) + 1):

        # Fit du GMM
        gmm = GaussianMixture(n_components=k, covariance_type=covariance_type,
                              reg_covar=reg_covar, random_state=random_state)
        gmm.fit(X)

        # Calcul du BIC
        bic = gmm.bic(X)
        bics.append(bic)
        print(f"  k={k} -> BIC={bic:.2f}")

        # Mise à jour du meilleur modèle
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm

    optimal_k = best_gmm.n_components
    return best_gmm, optimal_k, bics


# ------------------------------------------------------------
# CALCUL DES QUANTILES POUR LA VALIDATION
# ------------------------------------------------------------
def compute_quantiles(df: pd.DataFrame, cols: List[str], qs=[0.01,0.05,0.5,0.95,0.99]):
    out = {}
    for c in cols:
        if c in df.columns:
            q = df[c].dropna().quantile(qs).to_dict()
            out[c] = {float(k): float(v) for k,v in q.items()}
    return out


# ------------------------------------------------------------
# VALIDATION UNIVARIÉE : KS, Wasserstein, quantiles
# ------------------------------------------------------------
def print_validation(real: pd.DataFrame, synth: pd.DataFrame, features: List[str]):
    print("\n--- Validation univariée ---")
    for col in features:

        # Vérification que la colonne existe
        if col not in real.columns or col not in synth.columns:
            print(f"{col} : colonne manquante dans réel ou synthétique")
            continue

        r = real[col].dropna()
        s = synth[col].dropna()

        # Pas assez de données
        if len(r) < 2 or len(s) < 2:
            print(f"{col} : pas assez de données pour test")
            continue

        # KS test
        ks_stat, ks_p = ks_2samp(r, s)

        # Distance de Wasserstein
        w = wasserstein_distance(r, s)

        # Normalisation
        med = float(r.median()) if r.median() != 0 else None
        iqr = float(r.quantile(0.75) - r.quantile(0.25)) if (r.quantile(0.75) - r.quantile(0.25)) != 0 else None

        print(f"{col}  —  KS={ks_stat:.4f} (p={ks_p:.4g}), Wasserstein={w:.6g}")
        if med:
            print(f"    W/median = {w/med:.4f}")
        if iqr:
            print(f"    W/IQR = {w/iqr:.4f}")

        # Quantiles
        rq = compute_quantiles(real, [col])[col]
        sq = compute_quantiles(synth, [col])[col]
        print(f"    Quantiles réel: {rq}")
        print(f"    Quantiles synth: {sq}")


# ------------------------------------------------------------
# VALIDATION DES CORRÉLATIONS
# ------------------------------------------------------------
def print_correlations(real: pd.DataFrame, synth: pd.DataFrame, features: List[str]):
    common = [c for c in features if c in real.columns and c in synth.columns]
    if not common:
        print("Aucune colonne commune pour corrélations.")
        return

    print("\n--- Corrélations ---")
    print("Corr réel:")
    print(real[common].corr())

    print("\nCorr synth:")
    print(synth[common].corr())


# ------------------------------------------------------------
# PIPELINE PRINCIPAL
# ------------------------------------------------------------
def main():
    args = parse_args()
    np.random.seed(args.random_state)

    # Vérification du dossier
    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        print(f"[ERREUR] Dossier introuvable: {data_dir}", file=sys.stderr)
        sys.exit(1)

    # Recherche des CSV
    csvs = find_csv_files(data_dir)
    if not csvs:
        print(f"[ERREUR] Aucun CSV trouvé dans {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] {len(csvs)} fichiers CSV trouvés dans '{data_dir}', chargement...")
    dfs = []
    total_rows = 0

    # Chargement des CSV
    for path in csvs:
        try:
            df = read_csv_simple(path)
            print(f"[OK]   {os.path.basename(path)} : {len(df)} lignes")
            dfs.append(df)
            total_rows += len(df)
        except Exception as e:
            print(f"[ERREUR] {os.path.basename(path)} : {e} -> ignoré")

    if not dfs:
        print("[ERREUR] Aucun fichier CSV lisible.", file=sys.stderr)
        sys.exit(1)

    # Concaténation de tous les CSV
    full = pd.concat(dfs, ignore_index=True, sort=False)
    print(f"[INFO] Total chargé : {total_rows} lignes\n")

    # Normalisation des colonnes
    full.columns = normalize_cols(full.columns.tolist())

    # Vérification des features demandées
    features_mv = normalize_cols([c for c in args.features_mv])
    missing = [c for c in features_mv if c not in full.columns]
    if missing:
        print(f"[ERREUR] Colonnes demandées manquantes: {missing}", file=sys.stderr)
        print("Colonnes disponibles (extrait):", full.columns.tolist()[:40])
        sys.exit(1)

    # Séparation BENIGN / attaque
    label_col = args.label_col.strip()
    if label_col in full.columns:
        full[label_col] = full[label_col].astype(str).str.strip()
        benign_df = full[full[label_col].str.upper().str.contains("BENIGN", na=False)]
        attack_df = full[~full[label_col].str.upper().str.contains("BENIGN", na=False)]
    else:
        print(f"[WARN] Colonne label '{label_col}' introuvable — tout le dataset sera traité comme BENIGN")
        benign_df = full.copy()
        attack_df = full.iloc[0:0]

    print(f"\nBENIGN  : {len(benign_df)} lignes")
    print(f"Attaque : {len(attack_df)} lignes\n")

    # Extraction des features multivariées
    df_X = benign_df[features_mv].dropna().copy()
    if df_X.shape[0] < 10:
        print("[ERREUR] Pas assez de données BENIGN pour ajuster le GMM.", file=sys.stderr)
        sys.exit(1)

    # Clip des données réelles avant fit
    if args.fit_clip_quantile is not None:
        for col in features_mv:
            cap = df_X[col].quantile(args.fit_clip_quantile)
            n_clipped = int((df_X[col] > cap).sum())
            df_X[col] = df_X[col].clip(upper=cap)
            print(f"[FIT-CLIP] {col} : {n_clipped} valeurs > q{args.fit_clip_quantile} ({cap:.2f}) clippées avant fit")

    # Transformation log1p
    offsets: Dict[str, float] = {}
    if args.log_transform:
        for col in features_mv:
            df_X[col] = df_X[col].clip(lower=0)
            df_X[col] = np.log1p(df_X[col])
            offsets[col] = 0.0

    X = df_X.values

    # Fit du GMM multivarié
    print(f"=== GMM multivarié BENIGN — features : {features_mv} ===")
    gmm, optimal_k, bics = select_best_gmm(X, args.k_max, args.reg_covar, args.random_state,
                                            covariance_type=args.covariance_type)
    print(f"[GMM] k optimal = {optimal_k}\n")

    # Clip des variances log-espace
    max_log_var = args.max_log_var
    if hasattr(gmm, 'covariances_'):
        for i, cov in enumerate(gmm.covariances_):
            if cov.ndim == 2:
                diag = np.diag(cov).copy()
                if np.any(diag > max_log_var):
                    print(f"[VAR-CLIP] Composante {i} : variance log-espace {np.round(diag,3)} "
                          f"-> clippée à max {max_log_var}")
                    np.fill_diagonal(gmm.covariances_[i], np.clip(diag, None, max_log_var))
            elif cov.ndim == 1:  # covariance_type='diag'
                if np.any(cov > max_log_var):
                    print(f"[VAR-CLIP] Composante {i} : variance log-espace {np.round(cov,3)} "
                          f"-> clippée à max {max_log_var}")
                    gmm.covariances_[i] = np.clip(cov, None, max_log_var)

    # Sauvegarde du modèle
    model_payload = {"gmm": gmm, "features": features_mv, "reg_covar": args.reg_covar,
                     "log_transform": bool(args.log_transform), "offsets": offsets}
    joblib.dump(model_payload, args.model_path)
    print(f"[INFO] Modèle sauvegardé -> {args.model_path}")

    # Génération synthétique
    n = args.n_synthetic
    samples, _ = gmm.sample(n)
    synth_df = pd.DataFrame(samples, columns=features_mv)

    # Inversion log + clipping
    if args.log_transform:
        SAFE_EXP_MAX = 700.0
        for col in features_mv:
            synth_df[col] = np.minimum(synth_df[col], SAFE_EXP_MAX)
            with np.errstate(over='ignore', invalid='ignore'):
                synth_df[col] = np.expm1(synth_df[col])
            synth_df[col] = synth_df[col].clip(lower=0.0)
            if args.clip_max is not None:
                synth_df[col] = synth_df[col].clip(upper=args.clip_max)

    # Export CSV synthétique
    synth_out = "flux_synthetiques_benign.csv"
    synth_df.to_csv(synth_out, index=False)
    print(f"=== {n} flux synthétiques BENIGN générés -> {synth_out} ===\n")

    # Résumés statistiques
    print("-- Résumé synthétique --")
    print(synth_df[features_mv].describe().transpose().to_string())
    print("\n-- Résumé réel (BENIGN) --")
    print(benign_df[features_mv].describe().transpose().to_string())

    # Validation statistique
    try:
        print_validation(benign_df, synth_df, features_mv)
        print_correlations(benign_df, synth_df, features_mv)
    except Exception as e:
        print("[WARN] Erreur lors de la validation :", e)

    print("\nTerminé.")


if __name__ == "__main__":
    main()
