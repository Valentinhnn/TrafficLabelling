from __future__ import annotations

import os
import sys
import argparse
import warnings
import pickle
import contextlib
from typing import List, Tuple

import numpy as np
import pandas as pd
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


# Replaces +/-inf with the max finite value in the column, and NaN with 0.
def clean_inf_nan(df: pd.DataFrame, cols: List[str], label: str = "") -> pd.DataFrame:
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
        print(f"[INFO] {label}{n_inf_total} valeurs infinies plafonnées, {n_nan_total} NaN -> 0.0")
    return df


# Parses CLI arguments for this script.
def parse_args():
    p = argparse.ArgumentParser(description="Génération de séquences : features i.i.d. CTGAN + LSTM mono-variable sur delta_t")
    p.add_argument("--sequences-csv", required=True)
    p.add_argument("--ctgan-model-path", required=False, default=None,
                   help="Requis sauf en --diagnostic-only")
    p.add_argument("--min-seq-len", type=int, default=5)
    p.add_argument("--max-train-seq-len", type=int, default=60,
                   help="Découpage des séquences trop longues en chunks (même logique que "
                        "le script précédent -- évite l'explosion mémoire sur les hôtes très actifs)")
    p.add_argument("--hidden-dim", type=int, default=32,
                   help="Modèle mono-variable (delta_t seul) -> pas besoin d'un hidden_dim aussi "
                        "grand que la version multivariée précédente")
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--stock-size", type=int, default=50000,
                   help="Taille du stock CTGAN dans lequel piocher les features i.i.d.")
    p.add_argument("--n-synthetic-sequences", type=int, default=None)
    p.add_argument("--model-path", default=None)
    p.add_argument("--force-retrain", action="store_true")
    p.add_argument("--delta-t-mode", choices=["lstm", "iid"], default="lstm",
                   help="'lstm' (défaut) : delta_t généré par rollout autorégressif du LSTM "
                        "mono-variable (justifié si l'autocorrélation réelle affichée par le "
                        "diagnostic préalable est significative, ex. >0.1-0.15). "
                        "'iid' : pas d'entraînement, delta_t tiré par bootstrap direct dans la "
                        "distribution empirique réelle (indépendamment à chaque pas) -- à "
                        "utiliser si le diagnostic montre une autocorrélation quasi nulle "
                        "(comme pour les features), pour rester fidèle aux données sans "
                        "inventer de structure. Change juste ce flag, pas besoin de réentraîner "
                        "ni de repasser par un aller-retour.")
    p.add_argument("--delta-t-noise-std", type=float, default=0.75,
                   help="Écart-type du bruit gaussien ajouté (en espace standardisé) à la "
                        "prédiction du LSTM avant re-transformation en secondes (défaut 0.75, "
                        "proche de la racine de la MSE finale observée sur ce dataset). "
                        "JUSTIFICATION : une régression MSE prédit la MOYENNE conditionnelle en "
                        "espace log1p standardisé. Or expm1 est convexe -- repasser le point "
                        "moyen prédit en secondes SOUS-ESTIME fortement la vraie moyenne réelle "
                        "(inégalité de Jensen), d'autant plus que delta_t est très asymétrique. "
                        "Échantillonner un bruit autour de la prédiction avant expm1 restaure "
                        "une partie de cet écart, en générant des valeurs qui suivent la vraie "
                        "distribution plutôt que son seul point central. Mettre à 0 pour "
                        "retrouver le comportement déterministe d'avant (point estimate pur).")
    p.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    p.add_argument("--diagnostic-only", action="store_true",
                   help="Charge les séquences, affiche l'autocorrélation réelle du delta_t, "
                        "puis quitte SANS charger CTGAN ni entraîner de LSTM (rapide, quelques "
                        "secondes -- utile pour scripter la décision lstm/iid automatiquement, "
                        "cf. run_pipeline.sh)")
    p.add_argument("--output-dir", default="out_sequences_v2")
    args = p.parse_args()
    if not args.diagnostic_only and not args.ctgan_model_path:
        p.error("--ctgan-model-path est requis sauf en --diagnostic-only")
    return args


# Loads the sequences CSV, sorts by seq_id/position, drops too-short sequences,
# and splits too-long ones into chunks. Returns the list of sequence DataFrames
# plus a single concatenated DataFrame of everything kept.
def load_real_sequences(csv_path: str, min_seq_len: int, max_train_seq_len: int) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    df = pd.read_csv(csv_path)
    missing = [c for c in ["seq_id", "position_in_seq", "delta_t_seconds"] + FEATURES if c not in df.columns]
    if missing:
        print(f"[ERREUR] Colonnes manquantes : {missing}", file=sys.stderr)
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

    print(f"[INFO] {n_raw_seqs} séquences brutes, {n_chunked} découpées en chunks, "
          f"{len(sequences)} séquences/chunks retenus (longueur min={min(len(s) for s in sequences)}, "
          f"max={max(len(s) for s in sequences)}, median={int(np.median([len(s) for s in sequences]))})")

    real_used = pd.concat(sequences, ignore_index=True)
    return sequences, real_used


# Computes the average lag-1 autocorrelation of log1p(delta_t) across sequences,
# to check whether delta_t is worth modeling with a temporal model at all.
def diagnose_delta_t_autocorr(sequences: List[pd.DataFrame]) -> float:
    corrs = []
    for seq in sequences:
        dt = seq["delta_t_seconds"].values[1:]
        if len(dt) < 3:
            continue
        dt_log = np.log1p(np.clip(dt, 0, None))
        if dt_log.std() < 1e-9:
            continue
        c = np.corrcoef(dt_log[:-1], dt_log[1:])[0, 1]
        if not np.isnan(c):
            corrs.append(c)
    mean_corr = float(np.mean(corrs)) if corrs else float("nan")
    print(f"\n[DIAG] Autocorrélation lag-1 réelle de log1p(delta_t) au sein des séquences "
          f"host_pair : {mean_corr:.4f} (moyenne sur {len(corrs)} séquences valides)")
    if abs(mean_corr) < 0.05:
        print("[DIAG] ATTENTION : autocorrélation quasi nulle -> même le delta_t pourrait ne "
              "pas justifier un modèle temporel. Vérifier ce résultat avant d'investir dans "
              "l'entraînement du LSTM (envisager un tirage i.i.d. du delta_t si confirmé).")
    return mean_corr


# Standardizes delta_t in log1p space (mean/std computed from the real data).
class DeltaTScaler:
    def __init__(self):
        self.mean_ = 0.0
        self.std_ = 1.0

    # Computes mean/std of log1p(delta_t) over all sequences.
    def fit(self, sequences: List[pd.DataFrame]):
        all_dt = []
        for seq in sequences:
            dt = seq["delta_t_seconds"].values[1:]
            all_dt.append(np.clip(dt, 0, None))
        all_dt = np.concatenate(all_dt) if all_dt else np.array([0.0])
        log_dt = np.log1p(all_dt)
        self.mean_ = float(log_dt.mean())
        self.std_ = float(log_dt.std()) if log_dt.std() > 1e-9 else 1.0
        return self

    # Raw seconds -> standardized log space.
    def transform(self, raw_value: float) -> float:
        v = max(float(raw_value), 0.0)
        return (np.log1p(v) - self.mean_) / self.std_

    # Standardized log space -> raw seconds.
    def inverse_transform(self, z: float) -> float:
        logv = min(z * self.std_ + self.mean_, 50.0)
        return float(np.expm1(logv))

    # Exports mean/std for serialization.
    def save(self) -> dict:
        return {"mean": self.mean_, "std": self.std_}

    @staticmethod
    # Rebuilds a scaler from a saved mean/std dict.
    def load(payload: dict) -> "DeltaTScaler":
        s = DeltaTScaler()
        s.mean_ = payload["mean"]
        s.std_ = payload["std"]
        return s


# Builds one normalized delta_t vector per sequence, ready for LSTM training.
def build_training_tensors(sequences: List[pd.DataFrame], scaler: DeltaTScaler):
    mats = []
    for seq in sequences:
        dt = seq["delta_t_seconds"].values[1:]
        if len(dt) < 2:
            continue
        dt = np.clip(dt, 0, None)
        z = np.array([scaler.transform(v) for v in dt], dtype=np.float32)
        mats.append(z)
    return mats


# Imports torch/nn and caps CPU thread count for reasonable performance.
def _import_torch():
    try:
        import torch
        import torch.nn as nn
        n_threads = int(os.environ.get("TORCH_NUM_THREADS", "4"))
        torch.set_num_threads(n_threads)
        return torch, nn
    except ImportError:
        print("[ERREUR] PyTorch introuvable.")
        raise


# Builds a single-variable (delta_t only) LSTM: one LSTM layer + linear output head.
def build_model(hidden_dim: int, num_layers: int, torch, nn):
    class DeltaTLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden_dim, num_layers=num_layers, batch_first=True)
            self.head = nn.Linear(hidden_dim, 1)

        # Full-sequence forward pass (training), handles variable lengths via padding.
        def forward(self, x, lengths):
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            out_packed, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
            return self.head(out)

        # Single-step forward pass (autoregressive generation), keeps hidden state.
        def step(self, x_t, hidden):
            out, hidden = self.lstm(x_t, hidden)
            return self.head(out), hidden

    return DeltaTLSTM()


# Trains the delta_t LSTM with padded batches and a masked MSE loss (padding excluded).
def train_lstm(mats: List[np.ndarray], hidden_dim: int, num_layers: int,
                epochs: int, batch_size: int, lr: float, random_state: int):
    torch, nn = _import_torch()
    torch.manual_seed(random_state)

    model = build_model(hidden_dim, num_layers, torch, nn)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss(reduction="sum")

    inputs = [torch.tensor(m[:-1], dtype=torch.float32).unsqueeze(-1) for m in mats]
    targets = [torch.tensor(m[1:], dtype=torch.float32).unsqueeze(-1) for m in mats]
    lengths_all = [len(m) - 1 for m in mats]
    n = len(inputs)

    print(f"\n[LSTM] Entraînement sur {n} séquences/chunks (delta_t seul), "
          f"hidden_dim={hidden_dim}, epochs={epochs}\n")

    rng = np.random.default_rng(random_state)
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        total_loss, total_steps = 0.0, 0
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
            print(f"[LSTM] epoch {epoch:>4}/{epochs} — MSE moyenne (standardisé) = "
                  f"{total_loss / max(total_steps, 1):.6f}")

    return model


# Saves model weights + scaler + architecture params to disk.
def save_lstm(model, scaler: DeltaTScaler, hidden_dim: int, num_layers: int, model_path: str, torch):
    torch.save({"state_dict": model.state_dict(), "scaler": scaler.save(),
                "hidden_dim": hidden_dim, "num_layers": num_layers}, model_path)
    print(f"[INFO] Modèle LSTM (delta_t) sauvegardé -> {model_path}")


# Reloads a saved model + scaler, ready for inference.
def load_lstm(model_path: str, torch, nn):
    payload = torch.load(model_path, map_location="cpu")
    model = build_model(payload["hidden_dim"], payload["num_layers"], torch, nn)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    scaler = DeltaTScaler.load(payload["scaler"])
    return model, scaler


# Converts CTGAN's log1p output back to raw feature values and clips to the fit caps.
def postprocess_ctgan(synth: pd.DataFrame, features: List[str], caps: dict) -> pd.DataFrame:
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


# Loads a pretrained CTGAN model and samples a stock of i.i.d. synthetic feature rows.
def load_ctgan_stock(ctgan_model_path: str, stock_size: int, random_state: int) -> pd.DataFrame:
    with open(ctgan_model_path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "ctgan" not in payload:
        print(f"[ERREUR] {ctgan_model_path} n'est pas un payload CTGAN valide")
        sys.exit(1)

    ctgan = payload["ctgan"]
    features = payload["features"]
    caps = payload.get("caps", {})
    if features != FEATURES:
        print(f"[WARN] Features CTGAN ({features}) != attendues ({FEATURES})")

    print(f"[INFO] CTGAN rechargé depuis {ctgan_model_path} — échantillonnage de {stock_size} "
          f"flows i.i.d. (sans réentraînement)...")
    synth_log = ctgan.sample(stock_size)
    stock = postprocess_ctgan(synth_log, features, caps)
    stock = clean_inf_nan(stock, FEATURES, label="[stock CTGAN] ")
    print(f"[INFO] Stock CTGAN prêt : {len(stock)} flows")
    return stock.reset_index(drop=True)


# Generates n_synthetic sequences: features drawn i.i.d. from the CTGAN stock,
# delta_t either rolled out from the LSTM or bootstrapped from real values.
def generate_synthetic_sequences(mode: str, model, scaler: DeltaTScaler, stock: pd.DataFrame,
                                  real_sequences: List[pd.DataFrame], n_synthetic: int,
                                  random_state: int, torch=None, nn=None, delta_t_noise_std: float = 0.0):
    rng = np.random.default_rng(random_state)
    real_lengths = [len(s) for s in real_sequences]
    stock_arr = stock[FEATURES].values
    n_stock = len(stock_arr)

    real_dt_pool = np.concatenate([
        np.clip(s["delta_t_seconds"].values[1:], 0, None) for s in real_sequences if len(s) > 1
    ])
    real_dt_pool = real_dt_pool[~np.isnan(real_dt_pool)]

    results = []
    ctx = torch.no_grad() if mode == "lstm" else contextlib.nullcontext()
    if mode == "lstm":
        model.eval()

    with ctx:
        for seq_idx in range(n_synthetic):
            target_len = int(rng.choice(real_lengths))

            first_idx = rng.integers(0, n_stock)
            seq_rows = [{"position_in_seq": 0, "delta_t_seconds": np.nan,
                         **{c: float(stock_arr[first_idx, i]) for i, c in enumerate(FEATURES)}}]

            hidden = None
            x_t = None
            if mode == "lstm":
                init_seq = real_sequences[int(rng.integers(0, len(real_sequences)))]
                init_dt_series = init_seq["delta_t_seconds"].values[1:]
                init_dt = float(init_dt_series[0]) if len(init_dt_series) else 0.0
                x_t = torch.tensor([scaler.transform(init_dt)], dtype=torch.float32).view(1, 1, 1)

            for pos in range(1, target_len):
                if mode == "lstm":
                    pred, hidden = model.step(x_t, hidden)
                    z = float(pred.view(-1).item())
                    if delta_t_noise_std > 0:
                        z = z + rng.normal(0.0, delta_t_noise_std)
                    delta_t_synth = max(scaler.inverse_transform(z), 0.0)
                else:
                    delta_t_synth = float(rng.choice(real_dt_pool))

                feat_idx = rng.integers(0, n_stock)
                seq_rows.append({
                    "position_in_seq": pos,
                    "delta_t_seconds": delta_t_synth,
                    **{c: float(stock_arr[feat_idx, i]) for i, c in enumerate(FEATURES)},
                })

                if mode == "lstm":
                    x_t = torch.tensor([scaler.transform(delta_t_synth)], dtype=torch.float32).view(1, 1, 1)

            seq_id = f"synth_seq_{seq_idx:05d}"
            for row in seq_rows:
                row["seq_id"] = seq_id
                results.append(row)

    out_df = pd.DataFrame(results)[["seq_id", "position_in_seq", "delta_t_seconds"] + FEATURES]
    return out_df


# Average lag-1 correlation of `col` within each sequence (measures flow->flow dependency).
def lag1_corr_within_sequences(df: pd.DataFrame, col: str) -> float:
    vals = []
    for _, g in df.groupby("seq_id", sort=False):
        v = g[col].dropna().values
        if len(v) > 2 and np.std(v) > 1e-9:
            c = np.corrcoef(v[:-1], v[1:])[0, 1]
            if not np.isnan(c):
                vals.append(c)
    return float(np.mean(vals)) if vals else float("nan")


# Prints real-vs-synthetic comparison: delta_t distribution, feature distributions,
# and lag-1 autocorrelations, for a quick sanity check.
def validate(real_used: pd.DataFrame, synth: pd.DataFrame):
    print("\n=== Validation ===")

    real_dt = real_used["delta_t_seconds"].dropna().values
    synth_dt = synth["delta_t_seconds"].dropna().values
    if len(real_dt) > 1 and len(synth_dt) > 1:
        ks_stat, ks_p = ks_2samp(real_dt, synth_dt)
        w = wasserstein_distance(real_dt, synth_dt)
        print(f"delta_t_seconds : réel mean={real_dt.mean():.3f} synth mean={synth_dt.mean():.3f} | "
              f"KS={ks_stat:.4f} (p={ks_p:.4g}) | Wasserstein={w:.4f}")
    real_dt_corr = lag1_corr_within_sequences(real_used, "delta_t_seconds")
    synth_dt_corr = lag1_corr_within_sequences(synth, "delta_t_seconds")
    print(f"Autocorrélation lag-1 delta_t : réel={real_dt_corr:.4f} | synth={synth_dt_corr:.4f}")

    print("\n-- Fidélité marginale des features (doit ~= fidélité CTGAN de la partie 1, "
          "puisque tirage i.i.d. du même stock) --")
    for col in FEATURES:
        r = real_used[col].dropna().values
        s = synth[col].dropna().values
        if len(r) < 2 or len(s) < 2:
            continue
        ks_stat, ks_p = ks_2samp(r, s)
        w = wasserstein_distance(r, s)
        print(f"{col:22s} : KS={ks_stat:.4f} (p={ks_p:.4g}) | Wasserstein={w:.4g}")

    print("\n-- Corrélations flow->flow (lag-1, doit rester proche de 0 des deux côtés) --")
    for col in FEATURES:
        r_corr = lag1_corr_within_sequences(real_used, col)
        s_corr = lag1_corr_within_sequences(synth, col)
        print(f"{col:22s} : réel={r_corr:.4f} | synth={s_corr:.4f}")


# Entry point: load real sequences, diagnose delta_t autocorrelation, train/load
# the LSTM (or skip it in iid mode), generate synthetic sequences, validate, and save.
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = args.model_path or os.path.join(args.output_dir, "lstm_deltat_model.pt")

    sequences, real_used = load_real_sequences(args.sequences_csv, args.min_seq_len, args.max_train_seq_len)
    real_used_path = os.path.join(args.output_dir, "real_sequences_used.csv")
    real_used.to_csv(real_used_path, index=False)
    print(f"[INFO] Séquences réelles retenues exportées -> {real_used_path}")

    diagnose_delta_t_autocorr(sequences)

    if args.diagnostic_only:
        print("\n[INFO] --diagnostic-only : arrêt ici (pas de chargement CTGAN, pas d'entraînement).")
        sys.exit(0)

    model, scaler, torch, nn = None, None, None, None

    if args.delta_t_mode == "lstm":
        torch, nn = _import_torch()
        scaler = DeltaTScaler().fit(sequences)

        if not args.force_retrain and os.path.exists(model_path):
            print(f"\n[INFO] Modèle LSTM existant chargé depuis {model_path} (--force-retrain pour réentraîner)")
            model, scaler = load_lstm(model_path, torch, nn)
        else:
            mats = build_training_tensors(sequences, scaler)
            model = train_lstm(mats, args.hidden_dim, args.num_layers, args.epochs,
                                args.batch_size, args.lr, args.random_state)
            save_lstm(model, scaler, args.hidden_dim, args.num_layers, model_path, torch)
    else:
        print("\n[INFO] --delta-t-mode iid : pas d'entraînement, delta_t sera tiré par "
              "bootstrap direct sur la distribution empirique réelle à chaque pas.")

    stock = load_ctgan_stock(args.ctgan_model_path, args.stock_size, args.random_state)

    n_synth = args.n_synthetic_sequences or len(sequences)
    print(f"\n[INFO] Génération de {n_synth} séquences synthétiques (delta-t-mode={args.delta_t_mode}, "
          f"delta-t-noise-std={args.delta_t_noise_std})...")
    synth_df = generate_synthetic_sequences(args.delta_t_mode, model, scaler, stock, sequences,
                                             n_synth, args.random_state, torch, nn,
                                             delta_t_noise_std=args.delta_t_noise_std)

    synth_path = os.path.join(args.output_dir, "synthetic_sequences.csv")
    synth_df.to_csv(synth_path, index=False)
    print(f"[INFO] Séquences synthétiques exportées -> {synth_path} "
          f"({len(synth_df)} flows, {synth_df['seq_id'].nunique()} séquences)")

    validate(real_used, synth_df)

    print("\nTerminé.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOP] Interrompu par l'utilisateur (Ctrl-C).")
        sys.exit(0)