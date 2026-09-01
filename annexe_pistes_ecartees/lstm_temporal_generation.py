#!/usr/bin/env python3
"""
lstm_temporal_generation.py
=============================
Étapes 5-7 du pipeline dépendances temporelles : entraîne un LSTM/GRU
autorégressif sur les séquences réelles (host_pair) pour capturer les
corrélations flow->flow et le timing (delta_t), puis génère des séquences
synthétiques en combinant ce modèle avec le CTGAN déjà entraîné en partie 1.

PRINCIPE (pourquoi combiner LSTM + CTGAN plutôt que l'un ou l'autre seul) :
    - CTGAN (gan_train_generate.py) génère des flows individuels réalistes
      au niveau marginal, mais SANS AUCUNE notion d'ordre ou de séquence :
      chaque ligne est indépendante. Il ne peut donc pas reproduire les
      corrélations flow->flow (objectif principal de cette partie du stage).
    - Le LSTM appris ici capture au contraire l'ordre et les dépendances
      (quel type de flow suit quel type de flow, quel delta_t est probable
      compte tenu de l'historique récent de la séquence).
    - Combinaison : à chaque pas de la génération, le LSTM prédit un profil
      cible (vecteur de features normalisé). Au lieu de réinjecter cette
      prédiction brute comme historique (risque connu de dérive/valeurs
      irréalistes en rollout autorégressif sur de longues séquences), on
      cherche le flow le plus proche de ce profil dans un STOCK DE FLOWS
      CTGAN rechargé (plus proche voisin dans l'espace log1p + standardisé,
      cf. NearestNeighbors matching plus bas), et c'est ce flow réel-plausible
      (déjà validé statistiquement par CTGAN en partie 1) qui est réinjecté
      comme historique pour le pas suivant. Le delta_t, lui, n'a pas
      d'équivalent CTGAN (CTGAN ne génère pas de timing) -> on garde
      directement la prédiction du LSTM pour cette composante.

    Résultat : chaque flow individuel de la séquence synthétique est un vrai
    flow généré par CTGAN (donc fidèle aux distributions marginales déjà
    validées), mais l'ORDRE dans lequel ils apparaissent et le TIMING entre
    eux viennent du LSTM (donc les corrélations flow->flow et les patterns
    burst/idle sont respectés).

Prérequis :
    Le modèle CTGAN doit déjà avoir été entraîné et sauvegardé par
    gan_train_generate.py (payload pickle : ctgan, features, caps,
    fit_clip_quantile, random_state).

Usage entraînement + génération (tout-en-un) :
    python3 lstm_temporal_generation.py \
        --sequences-csv out_sequences/sequences_host_pair.csv \
        --ctgan-model-path ctgan_model_mv.pkl \
        --epochs 50 --hidden-dim 64 --stock-size 50000 \
        --n-synthetic-sequences 2000 \
        --output-dir out_lstm_temporal | tee lstm_temporal.log

Sorties (dans --output-dir) :
    lstm_model.pt                    -> poids + métadonnées du LSTM (rechargeable)
    synthetic_sequences.csv          -> séquences synthétiques générées
    real_sequences_used.csv          -> séquences réelles retenues pour l'entraînement
    validation_temporal.log (stdout) -> comparaison réel vs synthétique
"""
from __future__ import annotations

import os
import sys
import argparse
import warnings
import pickle
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import ks_2samp, wasserstein_distance

warnings.filterwarnings("ignore")

FEATURES = [
    "Flow Duration",
    "Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "Packet Length Std",
]
DEFAULT_RANDOM_STATE = 42


# ----------------------------------------------------------------------
# NETTOYAGE INF/NAN (bug connu CICIDS2017, déjà géré dans le reste du
# pipeline : Flow Bytes/s et Flow Packets/s valent +inf quand Flow Duration
# = 0, sinon log1p(inf)=inf -> std=NaN au fit du scaler -> NaN partout
# (perte NaN dès epoch 1, crash du k-d tree "data must be finite").)
# ----------------------------------------------------------------------
def clean_inf_nan(df: pd.DataFrame, cols: List[str], label: str = "") -> pd.DataFrame:
    """Remplace +-inf par le max fini de la colonne, et NaN par 0.0, sur les
    colonnes indiquées. Log le nombre de valeurs corrigées pour traçabilité."""
    df = df.copy()
    n_inf_total = 0
    n_nan_total = 0
    for col in cols:
        if col not in df.columns:
            continue
        is_inf = np.isinf(df[col].values)
        n_inf = int(is_inf.sum())
        if n_inf:
            n_inf_total += n_inf
            finite_vals = df.loc[~is_inf, col]
            cap = float(finite_vals.max()) if len(finite_vals) and finite_vals.notna().any() else 0.0
            df.loc[is_inf, col] = cap

        n_nan = int(df[col].isna().sum())
        if n_nan:
            n_nan_total += n_nan
            df[col] = df[col].fillna(0.0)

    if n_inf_total or n_nan_total:
        print(f"[INFO] {label}{n_inf_total} valeurs infinies plafonnées au max fini de leur "
              f"colonne (Flow Bytes/s ou Flow Packets/s quand Flow Duration=0), "
              f"{n_nan_total} NaN remplacés par 0.0")
    return df


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="LSTM temporel + matching CTGAN — génération de séquences synthétiques")
    p.add_argument("--sequences-csv", required=True,
                   help="CSV produit par temporal_sequences.py (colonnes seq_id, position_in_seq, "
                        "timestamp, delta_t_seconds, + 7 features)")
    p.add_argument("--ctgan-model-path", required=True,
                   help="Chemin du .pkl CTGAN déjà entraîné (payload gan_train_generate.py)")
    p.add_argument("--min-seq-len", type=int, default=5,
                   help="Longueur minimale d'une séquence pour l'entraînement (défaut 5, "
                        "cohérent avec temporal_sequences.py)")
    p.add_argument("--max-train-seq-len", type=int, default=60,
                   help="Longueur max d'un chunk d'entraînement (défaut 60). Les séquences plus "
                        "longues (ex. le serveur central 192.168.10.3 avec 294075 flows) sont "
                        "découpées en chunks consécutifs non chevauchants de cette taille -> "
                        "évite l'explosion mémoire et l'instabilité d'un LSTM déroulé sur des "
                        "dizaines de milliers de pas. Continuité perdue aux frontières de chunk "
                        "seulement, acceptable pour l'objectif (corrélations court/moyen terme).")
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--stock-size", type=int, default=50000,
                   help="Nombre de flows à échantillonner du CTGAN pour constituer le stock de "
                        "matching (défaut 50000). Plus grand = meilleurs matchs mais plus lent "
                        "à construire le k-d tree et plus de mémoire.")
    p.add_argument("--n-synthetic-sequences", type=int, default=None,
                   help="Nombre de séquences synthétiques à générer (défaut : autant que de "
                        "séquences réelles retenues pour l'entraînement)")
    p.add_argument("--model-path", default=None,
                   help="Chemin de sauvegarde/chargement du LSTM (défaut : <output-dir>/lstm_model.pt)")
    p.add_argument("--force-retrain", action="store_true",
                   help="Réentraîner même si un modèle LSTM existe déjà à --model-path")
    p.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    p.add_argument("--output-dir", default="out_lstm_temporal")
    return p.parse_args()


# ----------------------------------------------------------------------
# CHARGEMENT + PRÉPARATION DES SÉQUENCES RÉELLES
# ----------------------------------------------------------------------
def load_real_sequences(csv_path: str, min_seq_len: int, max_train_seq_len: int) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """
    Charge sequences_host_pair.csv, filtre les séquences trop courtes, et
    découpe les séquences trop longues en chunks consécutifs non chevauchants
    de taille <= max_train_seq_len (cf. doc du paramètre CLI).
    Retourne (liste de DataFrames triés par position, DataFrame concaténé
    pour référence/export).
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in ["seq_id", "position_in_seq", "delta_t_seconds"] + FEATURES if c not in df.columns]
    if missing:
        print(f"[ERREUR] Colonnes manquantes dans {csv_path} : {missing}", file=sys.stderr)
        sys.exit(1)

    df = clean_inf_nan(df, FEATURES, label="[réel] ")
    df = df.sort_values(["seq_id", "position_in_seq"]).reset_index(drop=True)

    sequences = []
    n_raw_seqs = 0
    n_chunked = 0
    for seq_id, group in df.groupby("seq_id", sort=False):
        n_raw_seqs += 1
        if len(group) < min_seq_len:
            continue
        if len(group) <= max_train_seq_len:
            sequences.append(group.reset_index(drop=True))
        else:
            n_chunked += 1
            for start in range(0, len(group), max_train_seq_len):
                chunk = group.iloc[start:start + max_train_seq_len]
                if len(chunk) >= min_seq_len:
                    sequences.append(chunk.reset_index(drop=True))

    print(f"[INFO] {n_raw_seqs} séquences brutes dans {csv_path}")
    print(f"[INFO] {n_chunked} séquences découpées en chunks (> {max_train_seq_len} flows)")
    print(f"[INFO] {len(sequences)} séquences/chunks retenus pour l'entraînement "
          f"(longueur min={min(len(s) for s in sequences)}, "
          f"max={max(len(s) for s in sequences)}, "
          f"median={int(np.median([len(s) for s in sequences]))})")

    real_used = pd.concat(sequences, ignore_index=True)
    return sequences, real_used


# ----------------------------------------------------------------------
# NORMALISATION (log1p + standardisation), cohérente avec le reste du pipeline
# ----------------------------------------------------------------------
class Scaler:
    """
    Transforme les 7 features + delta_t en log1p puis standardise (z-score).
    Les stats (mean/std) sont calculées UNE FOIS sur les données réelles
    d'entraînement et réutilisées partout (LSTM, stock CTGAN, matching) pour
    que tout le monde vive dans le même espace normalisé.
    """
    def __init__(self):
        self.mean_: Dict[str, float] = {}
        self.std_: Dict[str, float] = {}
        self.cols = FEATURES + ["delta_t_seconds"]

    def fit(self, df: pd.DataFrame):
        for col in self.cols:
            vals = np.log1p(df[col].fillna(0.0).clip(lower=0.0).values)
            self.mean_[col] = float(vals.mean())
            self.std_[col] = float(vals.std()) if vals.std() > 1e-9 else 1.0
        return self

    def transform_row(self, row: dict) -> np.ndarray:
        """row : dict {col: valeur brute}. Retourne un vecteur standardisé dans l'ordre self.cols."""
        out = np.zeros(len(self.cols), dtype=np.float32)
        for i, col in enumerate(self.cols):
            v = max(float(row.get(col, 0.0) or 0.0), 0.0)
            logv = np.log1p(v)
            out[i] = (logv - self.mean_[col]) / self.std_[col]
        return out

    def transform_df(self, df: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(df), len(self.cols)), dtype=np.float32)
        for i, col in enumerate(self.cols):
            vals = np.log1p(df[col].fillna(0.0).clip(lower=0.0).values.astype(np.float64))
            out[:, i] = (vals - self.mean_[col]) / self.std_[col]
        return out

    def inverse_transform_value(self, col: str, z: float) -> float:
        logv = z * self.std_[col] + self.mean_[col]
        logv = min(logv, 50.0)  # garde-fou avant expm1
        return float(np.expm1(logv))

    def save(self) -> dict:
        return {"mean": self.mean_, "std": self.std_, "cols": self.cols}

    @staticmethod
    def load(payload: dict) -> "Scaler":
        s = Scaler()
        s.mean_ = payload["mean"]
        s.std_ = payload["std"]
        s.cols = payload["cols"]
        return s


def build_training_tensors(sequences: List[pd.DataFrame], scaler: Scaler):
    """
    Construit, pour chaque séquence, la matrice normalisée [L, 8] (7 features
    + delta_t). L'entrée du LSTM au temps t = flow t ; la cible au temps t
    = flow t+1 (teacher forcing classique). Le tout premier flow d'une
    séquence n'a pas de delta_t réel (NaN) -> traité comme condition
    initiale seulement (jamais utilisé comme cible), donc le NaN ne pose pas
    de problème (fillna(0) dans transform_df ne sert qu'à l'entrée, pas à
    une prédiction évaluée).
    """
    mats = []
    for seq in sequences:
        if len(seq) < 2:
            continue
        mat = scaler.transform_df(seq)  # [L, 8]
        mats.append(mat)
    return mats


# ----------------------------------------------------------------------
# MODÈLE LSTM (PyTorch)
# ----------------------------------------------------------------------
def _import_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError:
        print("[ERREUR] PyTorch introuvable (déjà requis par ctgan, devrait être installé).")
        print("[INFO] pip install torch")
        raise


def build_model(input_dim: int, hidden_dim: int, num_layers: int, torch, nn):
    class TemporalLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
            self.head = nn.Linear(hidden_dim, input_dim)

        def forward(self, x, lengths):
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            out_packed, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
            return self.head(out)  # [B, T, input_dim]

        def step(self, x_t, hidden):
            # x_t : [1, 1, input_dim] -- un seul pas, pour la génération autoregressive
            out, hidden = self.lstm(x_t, hidden)
            pred = self.head(out)
            return pred, hidden

    return TemporalLSTM()


def train_lstm(mats: List[np.ndarray], hidden_dim: int, num_layers: int,
                epochs: int, batch_size: int, lr: float, random_state: int):
    torch, nn = _import_torch()
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    input_dim = mats[0].shape[1]
    model = build_model(input_dim, hidden_dim, num_layers, torch, nn)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss(reduction="sum")

    # entrée = flow[0:-1], cible = flow[1:]
    inputs = [torch.tensor(m[:-1], dtype=torch.float32) for m in mats]
    targets = [torch.tensor(m[1:], dtype=torch.float32) for m in mats]
    lengths_all = [len(m) - 1 for m in mats]

    n = len(inputs)
    print(f"\n[LSTM] Entraînement sur {n} séquences/chunks, input_dim={input_dim}, "
          f"hidden_dim={hidden_dim}, num_layers={num_layers}, epochs={epochs}\n")

    rng = np.random.default_rng(random_state)
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        total_loss = 0.0
        total_steps = 0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            batch_inputs = [inputs[i] for i in idx]
            batch_targets = [targets[i] for i in idx]
            batch_lengths = [lengths_all[i] for i in idx]

            x_pad = nn.utils.rnn.pad_sequence(batch_inputs, batch_first=True)
            y_pad = nn.utils.rnn.pad_sequence(batch_targets, batch_first=True)

            optimizer.zero_grad()
            pred = model(x_pad, batch_lengths)

            mask = torch.zeros_like(y_pad)
            for bi, L in enumerate(batch_lengths):
                mask[bi, :L, :] = 1.0

            loss = loss_fn(pred * mask, y_pad * mask) / mask.sum().clamp(min=1.0)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * mask.sum().item()
            total_steps += mask.sum().item()

        if epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            print(f"[LSTM] epoch {epoch:>4}/{epochs} — MSE moyenne (espace standardisé) = "
                  f"{total_loss / max(total_steps, 1):.6f}")

    return model


def save_lstm(model, scaler: Scaler, hidden_dim: int, num_layers: int, input_dim: int,
              model_path: str, torch):
    torch.save({
        "state_dict": model.state_dict(),
        "scaler": scaler.save(),
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "input_dim": input_dim,
    }, model_path)
    print(f"[INFO] Modèle LSTM sauvegardé -> {model_path}")


def load_lstm(model_path: str, torch, nn):
    payload = torch.load(model_path, map_location="cpu")
    model = build_model(payload["input_dim"], payload["hidden_dim"], payload["num_layers"], torch, nn)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    scaler = Scaler.load(payload["scaler"])
    return model, scaler


# ----------------------------------------------------------------------
# STOCK CTGAN (rechargement du modèle entraîné en partie 1, ré-échantillonnage)
# ----------------------------------------------------------------------
def postprocess_ctgan(synth: pd.DataFrame, features: List[str], caps: dict) -> pd.DataFrame:
    """Identique à postprocess() de gan_train_generate.py (dupliqué ici pour
    garder ce script autonome, sans dépendance d'import croisé fragile)."""
    out = synth[features].copy()
    SAFE_EXP_MAX = 700.0
    for col in features:
        out[col] = np.minimum(out[col].values, SAFE_EXP_MAX)
        with np.errstate(over="ignore", invalid="ignore"):
            out[col] = np.expm1(out[col])
        out[col] = out[col].clip(lower=0.0)
        if col in caps:
            out[col] = out[col].clip(upper=caps[col])
    return out


def load_ctgan_stock(ctgan_model_path: str, stock_size: int, random_state: int) -> pd.DataFrame:
    """
    Recharge le CTGAN déjà entraîné (payload pickle de gan_train_generate.py)
    et ré-échantillonne un stock de flows SANS relancer l'entraînement.
    """
    with open(ctgan_model_path, "rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict) or "ctgan" not in payload:
        print(f"[ERREUR] {ctgan_model_path} n'est pas un payload CTGAN valide "
              f"(format attendu : dict avec clé 'ctgan', cf. gan_train_generate.py)")
        sys.exit(1)

    ctgan = payload["ctgan"]
    features = payload["features"]
    caps = payload.get("caps", {})

    if features != FEATURES:
        print(f"[WARN] Les features du modèle CTGAN ({features}) diffèrent de celles "
              f"attendues ici ({FEATURES}) -> vérifie la cohérence du pipeline.")

    print(f"[INFO] CTGAN rechargé depuis {ctgan_model_path} — génération d'un stock de "
          f"{stock_size} flows pour le matching (sans réentraînement)...")
    synth_log = ctgan.sample(stock_size)
    stock = postprocess_ctgan(synth_log, features, caps)
    stock = clean_inf_nan(stock, FEATURES, label="[stock CTGAN] ")
    print(f"[INFO] Stock CTGAN prêt : {len(stock)} flows")
    return stock


def build_matching_index(stock: pd.DataFrame, scaler: Scaler) -> Tuple[cKDTree, np.ndarray]:
    """
    Construit un k-d tree sur le stock CTGAN, dans le MÊME espace normalisé
    (log1p + standardisation par les stats des données réelles) que celui
    utilisé pour les prédictions du LSTM, afin que la distance euclidienne
    ait un sens comparable entre features d'échelles très différentes
    (ex. Flow Duration en microsecondes vs Packet Length Std en octets).
    """
    stock_z = np.zeros((len(stock), len(FEATURES)), dtype=np.float32)
    for i, col in enumerate(FEATURES):
        vals = np.log1p(stock[col].clip(lower=0.0).values.astype(np.float64))
        stock_z[:, i] = (vals - scaler.mean_[col]) / scaler.std_[col]
    tree = cKDTree(stock_z)
    return tree, stock.reset_index(drop=True).values


# ----------------------------------------------------------------------
# GÉNÉRATION (rollout autorégressif LSTM + matching CTGAN à chaque pas)
# ----------------------------------------------------------------------
def generate_synthetic_sequences(model, scaler: Scaler, tree: cKDTree, stock_raw: np.ndarray,
                                  real_sequences: List[pd.DataFrame], n_synthetic: int,
                                  random_state: int, torch, nn):
    """
    Pour chaque séquence synthétique :
      1. Tire une longueur cible (bootstrap sur les longueurs réelles).
      2. Initialise l'état avec le PREMIER flow d'une séquence réelle tirée
         au hasard (garantit un point de départ statistiquement plausible,
         cohérent avec le fait que le premier flow d'une séquence n'a pas de
         delta_t/contexte à prédire -- même logique qu'à l'entraînement).
      3. À chaque pas : le LSTM prédit le vecteur normalisé du flow suivant
         (7 features + delta_t). Le delta_t prédit est gardé tel quel
         (inverse-transformé). Les 7 features prédites servent de requête au
         k-d tree -> le flow CTGAN le plus proche est récupéré et devient le
         flow réellement généré à ce pas ; c'est CE flow (normalisé) qui est
         réinjecté comme entrée du pas suivant (pas la prédiction brute du
         LSTM) -- évite la dérive typique du rollout autorégressif pur.
    """
    rng = np.random.default_rng(random_state)
    real_lengths = [len(s) for s in real_sequences]
    real_first_rows = [s.iloc[0] for s in real_sequences]

    results = []
    nn_distances = []

    model.eval()
    with torch.no_grad():
        for seq_idx in range(n_synthetic):
            target_len = int(rng.choice(real_lengths))
            init_idx = rng.integers(0, len(real_first_rows))
            first_row = real_first_rows[init_idx]

            first_vec = scaler.transform_row({**first_row.to_dict(), "delta_t_seconds": 0.0})
            history_raw = [dict(zip(FEATURES, [first_row[c] for c in FEATURES])) | {"delta_t_seconds": np.nan}]

            hidden = None
            x_t = torch.tensor(first_vec, dtype=torch.float32).view(1, 1, -1)

            seq_rows = [{"position_in_seq": 0, "delta_t_seconds": np.nan,
                         **{c: float(first_row[c]) for c in FEATURES}}]

            for pos in range(1, target_len):
                pred, hidden = model.step(x_t, hidden)
                pred_vec = pred.view(-1).numpy()  # [8] standardisé : 7 features + delta_t

                # delta_t : on garde la prédiction du LSTM telle quelle
                delta_t_synth = max(scaler.inverse_transform_value("delta_t_seconds", pred_vec[-1]), 0.0)

                # 7 features : requête au k-d tree dans l'espace normalisé (les 7 premières
                # composantes de pred_vec, dans le même ordre que FEATURES/scaler.cols)
                query = pred_vec[:len(FEATURES)].reshape(1, -1)
                dist, match_idx = tree.query(query, k=1)
                nn_distances.append(float(dist[0]))
                matched_features = stock_raw[int(match_idx[0])]  # valeurs brutes, ordre FEATURES

                seq_rows.append({
                    "position_in_seq": pos,
                    "delta_t_seconds": delta_t_synth,
                    **{c: float(v) for c, v in zip(FEATURES, matched_features)},
                })

                # réinjection : le flow MATCHÉ (réaliste) + le delta_t prédit, normalisés
                next_row = {**dict(zip(FEATURES, matched_features)), "delta_t_seconds": delta_t_synth}
                x_t = torch.tensor(scaler.transform_row(next_row), dtype=torch.float32).view(1, 1, -1)

            seq_id = f"synth_seq_{seq_idx:05d}"
            for row in seq_rows:
                row["seq_id"] = seq_id
                results.append(row)

    out_df = pd.DataFrame(results)
    out_cols = ["seq_id", "position_in_seq", "delta_t_seconds"] + FEATURES
    out_df = out_df[out_cols]

    mean_dist = float(np.mean(nn_distances)) if nn_distances else float("nan")
    print(f"\n[INFO] Distance moyenne au plus proche voisin CTGAN (espace standardisé) : "
          f"{mean_dist:.4f} -> plus c'est bas, plus les flows matchés collent aux profils "
          f"demandés par le LSTM (diagnostic de qualité du matching, pas une erreur de fit)")

    return out_df


# ----------------------------------------------------------------------
# VALIDATION
# ----------------------------------------------------------------------
def validate_temporal(real_used: pd.DataFrame, synth: pd.DataFrame):
    print("\n=== Validation dépendances temporelles ===")

    real_dt = real_used["delta_t_seconds"].dropna().values
    synth_dt = synth["delta_t_seconds"].dropna().values
    if len(real_dt) > 1 and len(synth_dt) > 1:
        ks_stat, ks_p = ks_2samp(real_dt, synth_dt)
        w = wasserstein_distance(real_dt, synth_dt)
        print(f"delta_t_seconds : réel mean={real_dt.mean():.3f} synth mean={synth_dt.mean():.3f} | "
              f"KS={ks_stat:.4f} (p={ks_p:.4g}) | Wasserstein={w:.4f}")

    print("\n-- Fidélité marginale des features (réel vs synthétique matché) --")
    for col in FEATURES:
        r = real_used[col].dropna().values
        s = synth[col].dropna().values
        if len(r) < 2 or len(s) < 2:
            continue
        ks_stat, ks_p = ks_2samp(r, s)
        w = wasserstein_distance(r, s)
        print(f"{col:22s} : KS={ks_stat:.4f} (p={ks_p:.4g}) | Wasserstein={w:.4g}")

    print("\n-- Corrélations flow->flow (lag-1, au sein des séquences) --")
    print("   (objectif principal de cette étape : ce que CTGAN seul ne peut pas reproduire)")

    def lag1_corr(df: pd.DataFrame, col: str) -> float:
        vals = []
        for _, g in df.groupby("seq_id", sort=False):
            v = g[col].values
            if len(v) > 2:
                vals.append(np.corrcoef(v[:-1], v[1:])[0, 1])
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    for col in FEATURES:
        r_corr = lag1_corr(real_used, col)
        s_corr = lag1_corr(synth, col)
        print(f"{col:22s} : réel={r_corr:.4f} | synth={s_corr:.4f}")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = args.model_path or os.path.join(args.output_dir, "lstm_model.pt")

    torch, nn = _import_torch()

    sequences, real_used = load_real_sequences(args.sequences_csv, args.min_seq_len, args.max_train_seq_len)
    real_used_path = os.path.join(args.output_dir, "real_sequences_used.csv")
    real_used.to_csv(real_used_path, index=False)
    print(f"[INFO] Séquences réelles retenues exportées -> {real_used_path}")

    scaler = Scaler().fit(real_used)

    if not args.force_retrain and os.path.exists(model_path):
        print(f"\n[INFO] Modèle LSTM existant chargé depuis {model_path} (--force-retrain pour réentraîner)")
        model, scaler = load_lstm(model_path, torch, nn)
    else:
        mats = build_training_tensors(sequences, scaler)
        model = train_lstm(mats, args.hidden_dim, args.num_layers, args.epochs,
                            args.batch_size, args.lr, args.random_state)
        save_lstm(model, scaler, args.hidden_dim, args.num_layers, mats[0].shape[1], model_path, torch)

    stock = load_ctgan_stock(args.ctgan_model_path, args.stock_size, args.random_state)
    tree, stock_raw = build_matching_index(stock, scaler)

    n_synth = args.n_synthetic_sequences or len(sequences)
    print(f"\n[INFO] Génération de {n_synth} séquences synthétiques...")
    synth_df = generate_synthetic_sequences(model, scaler, tree, stock_raw, sequences,
                                             n_synth, args.random_state, torch, nn)

    synth_path = os.path.join(args.output_dir, "synthetic_sequences.csv")
    synth_df.to_csv(synth_path, index=False)
    print(f"[INFO] Séquences synthétiques exportées -> {synth_path} "
          f"({len(synth_df)} flows, {synth_df['seq_id'].nunique()} séquences)")

    validate_temporal(real_used, synth_df)

    print("\nTerminé.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOP] Interrompu par l'utilisateur (Ctrl-C).")
        sys.exit(0)