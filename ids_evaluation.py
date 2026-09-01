#!/usr/bin/env python3
"""
ids_evaluation.py
==================
Évaluation applicative : est-ce que le trafic synthétique améliore
la détection d'intrusion (IDS) ?

Principe :
1. Charger les données réelles CICIDS2017 (BENIGN + attaques)
2. Entraîner un classifieur (Random Forest) sur données réelles seules
3. Ré-entraîner sur données réelles + synthétiques (data augmentation)
4. Comparer les performances (accuracy, F1, précision, rappel)
5. Si synthétique + réel > réel seul → les données synthétiques sont utiles

Le classifieur prédit : BENIGN (0) ou ATTAQUE (1)

Usage :
    # Évaluation de base (réel seul vs réel + GMM)
    python ids_evaluation.py --data-dir data \
        --synth-csvs flux_synthetiques_benign.csv flux_synthetiques_ddos.csv \
        --features "Flow Duration" "Packet Length Mean" \
        --random-state 42

    # Avec tous les synthétiques disponibles
    python ids_evaluation.py --data-dir data \
        --synth-dir . \
        --features "Flow Duration" "Packet Length Mean" \
        --random-state 42

Sorties :
    ids_evaluation_results.csv    -> tableau comparatif des performances
    ids_confusion_matrices.png    -> matrices de confusion côte à côte
    ids_metrics_comparison.png    -> graphique comparatif des métriques

CORRECTIF (vs version initiale) :
    - Fuite de données supprimée : la matrice de confusion du scénario
      "Réel + Synthétique" est désormais entraînée sur X_real_train + X_synth
      uniquement (jamais sur X_real_test), au lieu de X_real complet.
    - load_synth_csvs() traite désormais les +/-inf comme NA (cohérence avec
      load_real_data), important après un expm1() sur du GMM/CTGAN mal clippé.
"""
from __future__ import annotations

import os
import sys
import argparse
import warnings
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Évaluation IDS : réel vs réel+synthétique")
    p.add_argument("--data-dir", default="data",
                   help="Dossier contenant les CSV CICIDS2017")
    p.add_argument("--features", nargs="+",
                   default=["Flow Duration", "Packet Length Mean"],
                   help="Features à utiliser pour la classification")
    p.add_argument("--label-col", default="Label")
    p.add_argument("--synth-csvs", nargs="*", default=None,
                   help="Liste de CSV synthétiques à ajouter (ex: flux_synthetiques_benign.csv)")
    p.add_argument("--synth-dir", type=str, default=None,
                   help="Dossier contenant tous les CSV flux_synthetiques_*.csv à charger")
    p.add_argument("--n-estimators", type=int, default=100,
                   help="Nombre d'arbres Random Forest (défaut: 100)")
    p.add_argument("--cv-folds", type=int, default=5,
                   help="Nombre de folds pour validation croisée (défaut: 5)")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--output-dir", default=".",
                   help="Dossier de sortie (défaut: .)")
    p.add_argument("--max-rows-real", type=int, default=None,
                   help="Sous-échantillonnage des données réelles (None = tout)")
    return p.parse_args()


# ----------------------------------------------------------------------
# CHARGEMENT
# ----------------------------------------------------------------------
def normalize_cols(cols: List[str]) -> List[str]:
    return [" ".join(str(c).strip().lstrip("\ufeff").replace("\t", " ").split()) for c in cols]


def load_real_data(data_dir: str, features: List[str], label_col: str) -> pd.DataFrame:
    if not os.path.isdir(data_dir):
        print(f"[ERREUR] Dossier introuvable: {data_dir}", file=sys.stderr)
        sys.exit(1)
    csv_files = sorted(f for f in os.listdir(data_dir) if f.lower().endswith(".csv"))
    print(f"[INFO] {len(csv_files)} CSV trouvés dans '{data_dir}', chargement...")
    frames = []
    for fname in csv_files:
        try:
            # CORRECTIF : latin1 au lieu du defaut utf-8 -> certains fichiers
            # CICIDS2017 (ex: Thursday-...-WebAttacks) ne sont pas UTF-8
            # valides et etaient silencieusement exclus (perte d'une classe
            # d'attaque entiere pour l'evaluation IDS).
            df = pd.read_csv(os.path.join(data_dir, fname), low_memory=False, encoding="latin1")
            df.columns = normalize_cols(df.columns.tolist())
            frames.append(df)
            print(f"[OK]   {fname} : {len(df)} lignes")
        except Exception as e:
            print(f"[ERREUR] {fname} : {e} -> ignoré")
    if not frames:
        sys.exit(1)
    full = pd.concat(frames, ignore_index=True, sort=False)
    full.columns = normalize_cols(full.columns.tolist())
    print(f"[INFO] Total : {len(full)} lignes\n")

    needed = features + [label_col]
    missing = [c for c in needed if c not in full.columns]
    if missing:
        print(f"[ERREUR] Colonnes manquantes dans les données réelles: {missing}", file=sys.stderr)
        sys.exit(1)

    df = full[needed].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[(df[features] >= 0).all(axis=1)]

    df["label_binary"] = df[label_col].apply(
        lambda x: 0 if "BENIGN" in str(x).upper() else 1
    )
    return df


def load_synth_csvs(paths: List[str], features: List[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not os.path.exists(path):
            print(f"[WARN] Fichier synthétique introuvable: {path} -> ignoré")
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
            df.columns = normalize_cols(df.columns.tolist())

            if "Label_synthetic" in df.columns:
                label_val = df["Label_synthetic"].iloc[0]
            elif "benign" in os.path.basename(path).lower():
                label_val = "BENIGN"
            else:
                label_val = os.path.basename(path).replace("flux_synthetiques_", "").replace(".csv", "")

            df["label_binary"] = 0 if "BENIGN" in str(label_val).upper() else 1

            feat_present = [f for f in features if f in df.columns]
            if not feat_present:
                print(f"[WARN] {path} : aucune feature trouvée -> ignoré")
                continue

            df = df[feat_present + ["label_binary"]]
            # CORRECTIF : traiter +/-inf comme NA avant dropna, comme pour le réel
            # (important après expm1() sur des queues mal clippées)
            df[feat_present] = df[feat_present].replace([np.inf, -np.inf], np.nan)
            df = df.dropna()
            df = df[(df[feat_present] >= 0).all(axis=1)]

            frames.append(df)
            print(f"[OK]   {os.path.basename(path)} : {len(df)} lignes (label_binary={df['label_binary'].iloc[0]})")
        except Exception as e:
            print(f"[ERREUR] {path} : {e} -> ignoré")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def find_synth_csvs(synth_dir: str) -> List[str]:
    return [
        os.path.join(synth_dir, f)
        for f in sorted(os.listdir(synth_dir))
        if f.startswith("flux_synthetiques_") and f.endswith(".csv")
    ]


# ----------------------------------------------------------------------
# CLASSIFICATION
# ----------------------------------------------------------------------
def build_classifier(n_estimators: int, random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )


def evaluate_cv(X: np.ndarray, y: np.ndarray,
                clf, cv_folds: int, random_state: int) -> Dict:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = ["accuracy", "f1", "precision", "recall"]
    scores = cross_validate(clf, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        "accuracy":  round(float(np.mean(scores["test_accuracy"])), 4),
        "f1":        round(float(np.mean(scores["test_f1"])), 4),
        "precision": round(float(np.mean(scores["test_precision"])), 4),
        "recall":    round(float(np.mean(scores["test_recall"])), 4),
        "accuracy_std":  round(float(np.std(scores["test_accuracy"])), 4),
        "f1_std":        round(float(np.std(scores["test_f1"])), 4),
    }


def train_and_get_cm(X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray,
                     clf) -> np.ndarray:
    """Entraîne et retourne la matrice de confusion sur le jeu de test."""
    clf_copy = type(clf)(**clf.get_params())
    clf_copy.fit(X_train, y_train)
    y_pred = clf_copy.predict(X_test)
    return confusion_matrix(y_test, y_pred)


# ----------------------------------------------------------------------
# FIGURES
# ----------------------------------------------------------------------
def plot_confusion_matrices(cms: Dict[str, np.ndarray], output_path: str):
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    labels = ["BENIGN", "ATTAQUE"]
    for ax, (name, cm) in zip(axes, cms.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=labels, yticklabels=labels)
        ax.set_title(f"{name}")
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
    plt.suptitle("Matrices de confusion — Comparaison des scénarios", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Matrices de confusion -> {output_path}")


def plot_metrics_comparison(results_df: pd.DataFrame, output_path: str):
    metrics = ["accuracy", "f1", "precision", "recall"]
    x = np.arange(len(results_df))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, metric in enumerate(metrics):
        if metric in results_df.columns:
            ax.bar(x + i * width, results_df[metric], width, label=metric.capitalize())
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results_df["scenario"], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Comparaison des performances IDS — Réel vs Réel+Synthétique")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Graphique comparatif -> {output_path}")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    args = parse_args()
    np.random.seed(args.random_state)
    os.makedirs(args.output_dir, exist_ok=True)

    features = normalize_cols(args.features)
    label_col = args.label_col.strip()

    # --- Chargement données réelles ---
    print("=== Chargement données réelles ===")
    real_df = load_real_data(args.data_dir, features, label_col)

    if args.max_rows_real is not None and len(real_df) > args.max_rows_real:
        real_df = real_df.sample(n=args.max_rows_real, random_state=args.random_state)
        print(f"[INFO] Sous-échantillonnage réel : {args.max_rows_real} lignes")

    print(f"\nDonnées réelles : {len(real_df)} lignes")
    print(f"  BENIGN  : {(real_df['label_binary'] == 0).sum()}")
    print(f"  Attaque : {(real_df['label_binary'] == 1).sum()}\n")

    # --- Chargement données synthétiques ---
    synth_paths = []
    if args.synth_dir is not None:
        synth_paths = find_synth_csvs(args.synth_dir)
        print(f"[INFO] {len(synth_paths)} CSV synthétiques trouvés dans '{args.synth_dir}'")
    if args.synth_csvs:
        synth_paths += args.synth_csvs

    synth_df = pd.DataFrame()
    if synth_paths:
        print("\n=== Chargement données synthétiques ===")
        synth_df = load_synth_csvs(synth_paths, features)
        if not synth_df.empty:
            print(f"\nDonnées synthétiques : {len(synth_df)} lignes")
            print(f"  BENIGN synth  : {(synth_df['label_binary'] == 0).sum()}")
            print(f"  Attaque synth : {(synth_df['label_binary'] == 1).sum()}\n")

    # --- Split fixe réel (utilisé pour CM dans tous les scénarios) ---
    X_real = real_df[features].values
    y_real = real_df["label_binary"].values

    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
        X_real, y_real, test_size=0.2, random_state=args.random_state, stratify=y_real
    )

    # --- Préparation des scénarios (pour la CV, sur l'ensemble des données dispo) ---
    scenarios = {"Réel seul": (X_real, y_real)}

    X_synth, y_synth = None, None
    if not synth_df.empty:
        feat_synth = [f for f in features if f in synth_df.columns]
        if feat_synth:
            for f in features:
                if f not in synth_df.columns:
                    synth_df[f] = 0.0
            X_synth = synth_df[features].values
            y_synth = synth_df["label_binary"].values
            X_aug = np.vstack([X_real, X_synth])
            y_aug = np.concatenate([y_real, y_synth])
            scenarios["Réel + Synthétique"] = (X_aug, y_aug)
            print(f"[INFO] Scénario augmenté : {len(X_aug)} lignes total "
                  f"({len(X_real)} réels + {len(X_synth)} synthétiques)")

    # --- Évaluation ---
    clf = build_classifier(args.n_estimators, args.random_state)

    print(f"\n=== Évaluation par validation croisée ({args.cv_folds} folds) ===")
    results = []
    cms = {}

    for scenario_name, (X, y) in scenarios.items():
        print(f"\n--- {scenario_name} ---")
        print(f"    {len(X)} lignes, {y.sum()} attaques ({100*y.mean():.1f}%)")

        # Validation croisée (sur l'ensemble complet du scénario, pas de fuite :
        # cross_validate refait ses propres splits internes)
        metrics = evaluate_cv(X, y, clf, args.cv_folds, args.random_state)
        print(f"    Accuracy  : {metrics['accuracy']:.4f} (±{metrics['accuracy_std']:.4f})")
        print(f"    F1        : {metrics['f1']:.4f} (±{metrics['f1_std']:.4f})")
        print(f"    Précision : {metrics['precision']:.4f}")
        print(f"    Rappel    : {metrics['recall']:.4f}")

        metrics["scenario"] = scenario_name
        results.append(metrics)

        # --- Matrice de confusion : CORRECTIF fuite de données ---
        # On entraîne uniquement sur X_real_train (+ synth si dispo pour ce
        # scénario), jamais sur X_real_test, puis on teste sur X_real_test.
        if "Synthétique" in scenario_name and X_synth is not None:
            X_train_cm = np.vstack([X_real_train, X_synth])
            y_train_cm = np.concatenate([y_real_train, y_synth])
        else:
            X_train_cm = X_real_train
            y_train_cm = y_real_train

        cm = train_and_get_cm(X_train_cm, y_train_cm,
                               X_real_test, y_real_test, clf)
        cms[scenario_name] = cm
        print(f"    Matrice de confusion (test réel, hors fuite) :")
        print(f"      TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
        print(f"      FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")

    # --- Résultats ---
    results_df = pd.DataFrame(results)
    col_order = ["scenario", "accuracy", "accuracy_std", "f1", "f1_std",
                 "precision", "recall"]
    results_df = results_df[[c for c in col_order if c in results_df.columns]]

    out_csv = os.path.join(args.output_dir, "ids_evaluation_results.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\n[INFO] Résultats exportés -> {out_csv}")

    print("\n=== Tableau récapitulatif ===")
    print(results_df.to_string(index=False))

    # Verdict
    if len(results_df) >= 2:
        f1_reel = results_df.loc[results_df["scenario"] == "Réel seul", "f1"].values
        f1_aug = results_df.loc[results_df["scenario"] == "Réel + Synthétique", "f1"].values
        if len(f1_reel) > 0 and len(f1_aug) > 0:
            delta = float(f1_aug[0]) - float(f1_reel[0])
            print(f"\n=== Verdict ===")
            print(f"  ΔF1 (Réel+Synth vs Réel seul) = {delta:+.4f}")
            if delta > 0.005:
                print("  -> Le synthétique AMÉLIORE la détection d'intrusion ✅")
            elif delta < -0.005:
                print("  -> Le synthétique DÉGRADE la détection d'intrusion ❌")
            else:
                print("  -> Le synthétique n'a pas d'impact significatif (~0) ⚠️")

    # Figures
    cm_path = os.path.join(args.output_dir, "ids_confusion_matrices.png")
    plot_confusion_matrices(cms, cm_path)

    metrics_path = os.path.join(args.output_dir, "ids_metrics_comparison.png")
    plot_metrics_comparison(results_df, metrics_path)

    print("\nTerminé.")


if __name__ == "__main__":
    main()