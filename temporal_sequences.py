#!/usr/bin/env python3
"""
temporal_sequences.py
======================
Étapes 1-2 du pipeline dépendances temporelles : tri chronologique des flows
BENIGN de CICIDS2017 et construction des séquences (par paire d'hôtes,
par 5-tuple, ou par fenêtres temporelles fixes), avec calcul des
inter-arrival times (delta_t).

Sortie utilisée ensuite par le fit Hawkes (delta_t globaux/par groupe) et
par l'entraînement LSTM/GRU (séquences de features + delta_t).

Usage:
    python3 temporal_sequences.py --data-dir data --group-by host_pair \
        --min-seq-len 5 --output-dir out_sequences | tee temporal_sequences.log

CORRECTIFS/CONVENTIONS reprises du reste du pipeline :
    - lecture CSV en encoding="latin1" (cf. bug encodage documenté dans
      synthetic_generation.py / copula_generation.py / gan_train_generate.py).
    - normalize_cols() pour gérer BOM/espaces/tabs dans les en-têtes,
      identique aux autres scripts.
    - comparaison Label en BENIGN via strip + upper (même correctif que
      stats_features.py).
"""
from __future__ import annotations

import os
import argparse
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DEFAULT_FEATURES = [
    "Flow Duration",
    "Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "Packet Length Std",
]

# Noms candidats pour les colonnes d'identification (CICIDS2017 a des
# variantes de nommage selon les fichiers/versions du dataset).
CANDIDATE_COLS = {
    "timestamp": ["Timestamp"],
    "src_ip": ["Source IP", "Src IP", "SourceIP"],
    "dst_ip": ["Destination IP", "Dst IP", "DestinationIP"],
    "src_port": ["Source Port", "Src Port"],
    "dst_port": ["Destination Port", "Dst Port"],
    "protocol": ["Protocol"],
    "label": ["Label"],
}


# ----------------------------------------------------------------------
# UTILITAIRES (repris des autres scripts du pipeline)
# ----------------------------------------------------------------------
def normalize_cols(cols: list) -> list:
    return [" ".join(str(c).strip().lstrip("\ufeff").replace("\t", " ").split()) for c in cols]


def resolve_col(df_cols: List[str], candidates: List[str], role: str) -> Optional[str]:
    """Trouve la colonne réelle correspondant à un rôle (timestamp, src_ip, ...)."""
    for cand in candidates:
        if cand in df_cols:
            return cand
    return None


def find_csv_files(data_dir: str) -> List[str]:
    files = sorted(f for f in os.listdir(data_dir) if f.lower().endswith(".csv"))
    if not files:
        raise ValueError(f"Aucun fichier .csv trouvé dans {data_dir}")
    return files


# ----------------------------------------------------------------------
# CHARGEMENT
# ----------------------------------------------------------------------
def load_benign_with_ids(data_dir: str, features: List[str]) -> pd.DataFrame:
    """
    Charge tous les CSV, filtre BENIGN, conserve les colonnes d'identification
    (timestamp, IPs, ports, protocole) + les features continues.
    """
    csv_files = find_csv_files(data_dir)
    print(f"[INFO] {len(csv_files)} fichiers CSV trouvés dans '{data_dir}', chargement...")

    frames = []
    resolved = None  # noms de colonnes résolus, déterminés sur le 1er fichier valide

    for fname in csv_files:
        fpath = os.path.join(data_dir, fname)
        try:
            header_raw = pd.read_csv(fpath, encoding="latin1", nrows=0).columns.tolist()
            header_norm = normalize_cols(header_raw)
            norm_to_raw = dict(zip(header_norm, header_raw))

            if resolved is None:
                resolved = {}
                for role, candidates in CANDIDATE_COLS.items():
                    col = resolve_col(header_norm, candidates, role)
                    if col is None and role != "label":
                        print(f"[WARN] Colonne pour '{role}' introuvable (candidats testés : {candidates})")
                    resolved[role] = col
                missing_required = [r for r in ("timestamp", "src_ip", "dst_ip", "label") if resolved.get(r) is None]
                if missing_required:
                    raise ValueError(
                        f"Colonnes indispensables manquantes : {missing_required}. "
                        f"Colonnes disponibles (normalisées) : {header_norm}"
                    )
                print(f"[INFO] Colonnes résolues : { {k: v for k, v in resolved.items() if v} }")

            wanted_norm = set(features) | {v for v in resolved.values() if v}
            missing = wanted_norm - set(header_norm)
            if missing:
                print(f"[WARN] {fname} : colonnes manquantes {missing} -> fichier ignoré")
                continue

            raw_cols_to_load = [norm_to_raw[w] for w in wanted_norm]
            df_tmp = pd.read_csv(fpath, encoding="latin1", low_memory=False, usecols=raw_cols_to_load)
            df_tmp.columns = normalize_cols(df_tmp.columns.tolist())
            df_tmp["__source_file"] = fname
            frames.append(df_tmp)
            print(f"[OK]   {fname} : {len(df_tmp)} lignes")
        except Exception as e:
            print(f"[ERREUR] {fname} : {e} -> ignoré")

    if not frames:
        raise ValueError("Aucun fichier n'a pu être chargé correctement.")
    df = pd.concat(frames, ignore_index=True)

    label_col = resolved["label"]
    label_series = df[label_col].astype(str).str.strip()
    is_benign = label_series.str.upper() == "BENIGN"
    df = df.loc[is_benign].drop(columns=[label_col]).reset_index(drop=True)
    print(f"\n[INFO] {len(df)} flows BENIGN conservés (sur le total chargé)")

    # VALIDATION : IPs manquantes/vides -> auraient sinon créé de faux groupes
    # "nan__nan" ou "x__x" lors du groupement par paire (astype(str) transforme
    # NaN en la chaîne littérale "nan", et pandas min/max(axis=1, skipna=True)
    # collapse une paire avec une IP manquante sur l'IP restante).
    src_col, dst_col = resolved["src_ip"], resolved["dst_ip"]
    n_before = len(df)
    bad_ip_mask = (
        df[src_col].isna() | df[dst_col].isna()
        | (df[src_col].astype(str).str.strip() == "")
        | (df[dst_col].astype(str).str.strip() == "")
    )
    if bad_ip_mask.any():
        print(f"[WARN] {bad_ip_mask.sum()} lignes avec IP source/destination manquante ou vide -> supprimées")
        df = df.loc[~bad_ip_mask].reset_index(drop=True)
    else:
        print(f"[INFO] Aucune IP manquante/vide sur les {n_before} flows BENIGN")

    return df, resolved


# ----------------------------------------------------------------------
# ÉTAPE 1 : TRI TEMPOREL
# ----------------------------------------------------------------------
def parse_timestamps_robust(series: pd.Series) -> pd.Series:
    """
    Parse une colonne Timestamp CICIDS2017 en essayant plusieurs formats connus
    successivement (le dataset mélange des formats selon les fichiers, ex.
    '03/07/2017 08:55:58' vs '4/7/2017 8:54' -> l'inférence automatique de
    pd.to_datetime peut échouer sur des colonnes non-homogènes selon la version
    de pandas). Les lignes non résolues après les formats explicites passent
    par un dernier essai en parsing flexible (dateutil, plus lent mais tolérant).
    """
    candidate_formats = [
        "%d/%m/%Y %H:%M:%S",  # ex. 03/07/2017 08:55:58 (Monday)
        "%d/%m/%Y %H:%M",     # ex. 4/7/2017 8:54 (Tuesday et autres)
    ]
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    remaining = pd.Series(True, index=series.index)

    for fmt in candidate_formats:
        if not remaining.any():
            break
        parsed = pd.to_datetime(series[remaining], format=fmt, errors="coerce")
        result.loc[remaining] = parsed
        remaining = result.isna()

    if remaining.any():
        # dernier recours, plus lent : parsing flexible sur ce qui reste seulement
        parsed = pd.to_datetime(series[remaining], dayfirst=True, errors="coerce")
        result.loc[remaining] = parsed

    return result


def parse_and_sort(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    df["_ts"] = parse_timestamps_robust(df[ts_col])
    n_nat = df["_ts"].isna().sum()
    if n_nat:
        print(f"[WARN] {n_nat} timestamps non parsés (NaT) -> lignes supprimées")
        df = df.dropna(subset=["_ts"])
    df = df.sort_values("_ts").reset_index(drop=True)
    print(f"[INFO] Tri temporel effectué : {df['_ts'].min()} -> {df['_ts'].max()}")
    return df


# ----------------------------------------------------------------------
# ÉTAPE 2 : CONSTRUCTION DES SÉQUENCES
# ----------------------------------------------------------------------
def build_host_pair_sequences(df: pd.DataFrame, resolved: dict, features: List[str],
                               min_seq_len: int) -> pd.DataFrame:
    """Groupe par paire (src_ip, dst_ip) non-ordonnée -> plus de flows par groupe
    que le 5-tuple strict (regroupe toute la conversation, tous ports/protocoles
    confondus)."""
    src_col, dst_col = resolved["src_ip"], resolved["dst_ip"]
    # paire non-ordonnée : (min(ip1,ip2), max(ip1,ip2)) pour regrouper les deux sens
    pair_a = df[[src_col, dst_col]].min(axis=1)
    pair_b = df[[src_col, dst_col]].max(axis=1)
    df = df.copy()
    df["_group_id"] = pair_a.astype(str) + "__" + pair_b.astype(str)
    df["_group_label"] = pair_a.astype(str) + " <-> " + pair_b.astype(str)
    return _finalize_sequences(df, features, min_seq_len, group_label="host_pair", keep_label=True)


def build_five_tuple_sequences(df: pd.DataFrame, resolved: dict, features: List[str],
                                min_seq_len: int) -> pd.DataFrame:
    cols = [resolved["src_ip"], resolved["dst_ip"]]
    for role in ("src_port", "dst_port", "protocol"):
        if resolved.get(role):
            cols.append(resolved[role])
    df = df.copy()
    df["_group_id"] = df[cols].astype(str).agg("__".join, axis=1)
    df["_group_label"] = df["_group_id"]
    return _finalize_sequences(df, features, min_seq_len, group_label="five_tuple", keep_label=True)


def build_time_window_sequences(df: pd.DataFrame, features: List[str],
                                 window_seconds: int, min_seq_len: int) -> pd.DataFrame:
    df = df.copy()
    t0 = df["_ts"].min()
    window_idx = ((df["_ts"] - t0).dt.total_seconds() // window_seconds).astype(int)
    df["_group_id"] = "win_" + window_idx.astype(str)
    return _finalize_sequences(df, features, min_seq_len, group_label="time_window", keep_label=False)


def _finalize_sequences(df: pd.DataFrame, features: List[str], min_seq_len: int,
                         group_label: str, keep_label: bool = False) -> pd.DataFrame:
    df = df.sort_values(["_group_id", "_ts"]).reset_index(drop=True)

    group_sizes = df.groupby("_group_id").size()
    valid_groups = group_sizes[group_sizes >= min_seq_len].index
    n_total_groups = len(group_sizes)
    df = df[df["_group_id"].isin(valid_groups)].reset_index(drop=True)

    print(f"[INFO] Groupement '{group_label}' : {n_total_groups} groupes trouvés, "
          f"{len(valid_groups)} conservés (>= {min_seq_len} flows), "
          f"{df['_group_id'].nunique()} groupes / {len(df)} flows dans la sortie finale")
    if len(valid_groups) > 0:
        print(f"[INFO] Taille des groupes valides : min={group_sizes[valid_groups].min()}, "
              f"median={int(group_sizes[valid_groups].median())}, max={group_sizes[valid_groups].max()}")

        # DIAGNOSTIC : identité réelle des 10 plus gros groupes, pour repérer
        # tout de suite un artefact (IP par défaut, broadcast, etc.) plutôt
        # qu'une conversation légitime très active.
        top10 = group_sizes[valid_groups].sort_values(ascending=False).head(10)
        print(f"[DIAG] Top 10 des plus gros groupes ({group_label}) :")
        if keep_label:
            label_by_group = df.drop_duplicates("_group_id").set_index("_group_id")["_group_label"]
            for gid, size in top10.items():
                print(f"       {size:>8} flows -> {label_by_group.get(gid, gid)}")
        else:
            for gid, size in top10.items():
                print(f"       {size:>8} flows -> {gid}")

    # delta_t = inter-arrival time au sein du groupe (NaN pour le 1er flow de chaque groupe)
    df["delta_t_seconds"] = df.groupby("_group_id")["_ts"].diff().dt.total_seconds()
    df["position_in_seq"] = df.groupby("_group_id").cumcount()

    # renommage des groupes en identifiants séquentiels courts (seq_0000, seq_0001, ...)
    unique_groups = df["_group_id"].unique()
    remap = {g: f"seq_{i:05d}" for i, g in enumerate(unique_groups)}
    df["seq_id"] = df["_group_id"].map(remap)

    out_cols = ["seq_id", "position_in_seq", "_ts", "delta_t_seconds"] + features
    if keep_label:
        out_cols.insert(1, "_group_label")
    result = df[out_cols].rename(columns={"_ts": "timestamp"})
    if keep_label:
        result = result.rename(columns={"_group_label": "group_identity"})
    return result


# ----------------------------------------------------------------------
# INTER-ARRIVAL TIMES GLOBAUX (pour le futur fit Hawkes sur le process
# d'arrivée global des flows BENIGN, indépendamment du groupement)
# ----------------------------------------------------------------------
def compute_global_interarrival(df: pd.DataFrame) -> pd.DataFrame:
    ts_sorted = df["_ts"].sort_values().reset_index(drop=True)
    delta = ts_sorted.diff().dt.total_seconds()
    out = pd.DataFrame({"timestamp": ts_sorted, "delta_t_seconds": delta})
    return out


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Tri temporel + construction de séquences de flows BENIGN")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    p.add_argument("--group-by", choices=["host_pair", "five_tuple", "time_window"], default="host_pair")
    p.add_argument("--window-seconds", type=int, default=60,
                   help="Taille des fenêtres temporelles fixes (si --group-by time_window)")
    p.add_argument("--min-seq-len", type=int, default=5,
                   help="Longueur minimale d'une séquence pour être conservée")
    p.add_argument("--output-dir", default="out_sequences")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    features = normalize_cols(args.features)

    df, resolved = load_benign_with_ids(args.data_dir, features)
    df = parse_and_sort(df, resolved["timestamp"])

    # inter-arrival global (utile pour Hawkes, indépendant du groupement choisi)
    global_iat = compute_global_interarrival(df)
    global_iat_path = os.path.join(args.output_dir, "global_interarrival.csv")
    global_iat.to_csv(global_iat_path, index=False)
    print(f"\n[INFO] Inter-arrival times globaux exportés -> {global_iat_path} "
          f"({len(global_iat)} lignes, delta_t median="
          f"{global_iat['delta_t_seconds'].median():.4f}s)")

    # DIAGNOSTIC : ampleur précise de la résolution "minute" du Timestamp
    # CICIDS2017 (impacte le choix Hawkes continu vs comptage discret par bin).
    valid_dt = global_iat["delta_t_seconds"].dropna()
    pct_zero = (valid_dt == 0).mean() * 100
    pct_below_60s = (valid_dt < 60).mean() * 100
    all_seconds_zero = (df["_ts"].dt.second == 0).all()
    print(f"[DIAG] Résolution temporelle : {pct_zero:.1f}% des delta_t consécutifs = 0s, "
          f"{pct_below_60s:.1f}% < 60s")
    print(f"[DIAG] Composante 'secondes' toujours à 0 sur l'ensemble des timestamps : {all_seconds_zero} "
          f"(True = résolution native à la minute confirmée sur tout le dataset)")

    print(f"\n=== Construction des séquences (group-by={args.group_by}) ===")
    if args.group_by == "host_pair":
        seq_df = build_host_pair_sequences(df, resolved, features, args.min_seq_len)
    elif args.group_by == "five_tuple":
        seq_df = build_five_tuple_sequences(df, resolved, features, args.min_seq_len)
    else:
        seq_df = build_time_window_sequences(df, features, args.window_seconds, args.min_seq_len)

    seq_path = os.path.join(args.output_dir, f"sequences_{args.group_by}.csv")
    seq_df.to_csv(seq_path, index=False)
    print(f"\n[INFO] Séquences exportées -> {seq_path} ({len(seq_df)} flows, "
          f"{seq_df['seq_id'].nunique()} séquences)")

    print("\nTerminé.")


if __name__ == "__main__":
    main()