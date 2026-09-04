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

CANDIDATE_COLS = {
    "timestamp": ["Timestamp"],
    "src_ip": ["Source IP", "Src IP", "SourceIP"],
    "dst_ip": ["Destination IP", "Dst IP", "DestinationIP"],
    "src_port": ["Source Port", "Src Port"],
    "dst_port": ["Destination Port", "Dst Port"],
    "protocol": ["Protocol"],
    "label": ["Label"],
}


# Cleans column names: strips BOM/whitespace and collapses tabs/multiple spaces.
def normalize_cols(cols: list) -> list:
    return [" ".join(str(c).strip().lstrip("\ufeff").replace("\t", " ").split()) for c in cols]


# Finds which candidate column name is actually present for a given role.
def resolve_col(df_cols: List[str], candidates: List[str], role: str) -> Optional[str]:
    for cand in candidates:
        if cand in df_cols:
            return cand
    return None


# Lists CSV files in a directory, raises if none found.
def find_csv_files(data_dir: str) -> List[str]:
    files = sorted(f for f in os.listdir(data_dir) if f.lower().endswith(".csv"))
    if not files:
        raise ValueError(f"Aucun fichier .csv trouvé dans {data_dir}")
    return files


# Loads all CSVs, resolves the id/feature columns, keeps only BENIGN flows,
# and drops rows with missing/empty source or destination IP.
def load_benign_with_ids(data_dir: str, features: List[str]) -> pd.DataFrame:
    csv_files = find_csv_files(data_dir)
    print(f"[INFO] {len(csv_files)} fichiers CSV trouvés dans '{data_dir}', chargement...")

    frames = []
    resolved = None

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


# Parses timestamps by trying known exact formats first, then a slower flexible
# parser as a fallback for whatever doesn't match.
def parse_timestamps_robust(series: pd.Series) -> pd.Series:
    candidate_formats = [
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
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
        parsed = pd.to_datetime(series[remaining], dayfirst=True, errors="coerce")
        result.loc[remaining] = parsed

    return result


# Parses the timestamp column, drops unparseable rows, and sorts chronologically.
def parse_and_sort(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    df["_ts"] = parse_timestamps_robust(df[ts_col])
    n_nat = df["_ts"].isna().sum()
    if n_nat:
        print(f"[WARN] {n_nat} timestamps non parsés (NaT) -> lignes supprimées")
        df = df.dropna(subset=["_ts"])
    df = df.sort_values("_ts").reset_index(drop=True)
    print(f"[INFO] Tri temporel effectué : {df['_ts'].min()} -> {df['_ts'].max()}")
    return df


# Groups flows by unordered (src_ip, dst_ip) pair, i.e. one sequence per conversation
# regardless of port/protocol/direction.
def build_host_pair_sequences(df: pd.DataFrame, resolved: dict, features: List[str],
                               min_seq_len: int) -> pd.DataFrame:
    src_col, dst_col = resolved["src_ip"], resolved["dst_ip"]
    pair_a = df[[src_col, dst_col]].min(axis=1)
    pair_b = df[[src_col, dst_col]].max(axis=1)
    df = df.copy()
    df["_group_id"] = pair_a.astype(str) + "__" + pair_b.astype(str)
    df["_group_label"] = pair_a.astype(str) + " <-> " + pair_b.astype(str)
    return _finalize_sequences(df, features, min_seq_len, group_label="host_pair", keep_label=True)


# Groups flows by strict 5-tuple (src/dst IP, src/dst port, protocol).
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


# Groups flows into fixed-size time windows, independent of source/destination.
def build_time_window_sequences(df: pd.DataFrame, features: List[str],
                                 window_seconds: int, min_seq_len: int) -> pd.DataFrame:
    df = df.copy()
    t0 = df["_ts"].min()
    window_idx = ((df["_ts"] - t0).dt.total_seconds() // window_seconds).astype(int)
    df["_group_id"] = "win_" + window_idx.astype(str)
    return _finalize_sequences(df, features, min_seq_len, group_label="time_window", keep_label=False)


# Shared logic for all group-by modes: filters groups below min_seq_len, computes
# delta_t and position_in_seq within each group, and assigns short seq_ids.
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

        top10 = group_sizes[valid_groups].sort_values(ascending=False).head(10)
        print(f"[DIAG] Top 10 des plus gros groupes ({group_label}) :")
        if keep_label:
            label_by_group = df.drop_duplicates("_group_id").set_index("_group_id")["_group_label"]
            for gid, size in top10.items():
                print(f"       {size:>8} flows -> {label_by_group.get(gid, gid)}")
        else:
            for gid, size in top10.items():
                print(f"       {size:>8} flows -> {gid}")

    df["delta_t_seconds"] = df.groupby("_group_id")["_ts"].diff().dt.total_seconds()
    df["position_in_seq"] = df.groupby("_group_id").cumcount()

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


# Computes inter-arrival times across all flows globally, ignoring any grouping.
def compute_global_interarrival(df: pd.DataFrame) -> pd.DataFrame:
    ts_sorted = df["_ts"].sort_values().reset_index(drop=True)
    delta = ts_sorted.diff().dt.total_seconds()
    out = pd.DataFrame({"timestamp": ts_sorted, "delta_t_seconds": delta})
    return out


# Entry point: load benign flows, sort by time, group into sequences with the
# chosen strategy, compute global inter-arrival times, and save CSVs.
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

    global_iat = compute_global_interarrival(df)
    global_iat_path = os.path.join(args.output_dir, "global_interarrival.csv")
    global_iat.to_csv(global_iat_path, index=False)
    print(f"\n[INFO] Inter-arrival times globaux exportés -> {global_iat_path} "
          f"({len(global_iat)} lignes, delta_t median="
          f"{global_iat['delta_t_seconds'].median():.4f}s)")

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