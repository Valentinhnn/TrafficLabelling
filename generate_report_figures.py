from __future__ import annotations

import os
import sys
import argparse
import warnings

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


# Parses CLI arguments for this script.
def parse_args():
    p = argparse.ArgumentParser(description="Figures + résumé méthodologique (étape 8)")
    p.add_argument("--real-sequences-csv", required=True)
    p.add_argument("--synth-sequences-csv", required=True)
    p.add_argument("--output-dir", default="out_sequences_v2/figures")
    p.add_argument("--delta-t-mode", choices=["lstm", "iid"], default="lstm",
                   help="Doit correspondre au mode utilisé pour générer --synth-sequences-csv "
                        "(juste pour que le résumé Markdown mentionne la bonne méthode)")
    return p.parse_args()


# Average lag-1 correlation of `col` within each sequence.
def lag1_corr_within_sequences(df: pd.DataFrame, col: str) -> float:
    vals = []
    for _, g in df.groupby("seq_id", sort=False):
        v = g[col].dropna().values
        if len(v) > 2 and np.std(v) > 1e-9:
            c = np.corrcoef(v[:-1], v[1:])[0, 1]
            if not np.isnan(c):
                vals.append(c)
    return float(np.mean(vals)) if vals else float("nan")


# Imports matplotlib in headless mode; returns None if it isn't installed.
def try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("[WARN] matplotlib introuvable -> les figures PNG seront sautées "
              "(pip install matplotlib), mais le résumé Markdown sera quand même généré.")
        return None


# Plots real vs synthetic delta_t distributions (log1p histograms) and saves as PNG.
def plot_delta_t(plt, real_dt, synth_dt, output_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, np.log1p(max(real_dt.max(), synth_dt.max(), 1)), 60)
    ax.hist(np.log1p(real_dt), bins=bins, alpha=0.55, label="Réel", density=True, color="#2b6cb0")
    ax.hist(np.log1p(synth_dt), bins=bins, alpha=0.55, label="Synthétique", density=True, color="#e07a5f")
    ax.set_xlabel("log(1 + delta_t_seconds)")
    ax.set_ylabel("Densité")
    ax.set_title("Distribution du delta_t entre flows consécutifs")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(output_dir, "fig_delta_t_distribution.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Figure sauvegardée -> {out_path}")


# Plots real vs synthetic distributions for each feature in a 3x3 grid and saves as PNG.
def plot_features(plt, real_used, synth, output_dir):
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    axes = axes.flatten()
    for i, col in enumerate(FEATURES):
        ax = axes[i]
        r = np.log1p(np.clip(real_used[col].dropna().values, 0, None))
        s = np.log1p(np.clip(synth[col].dropna().values, 0, None))
        bins = np.linspace(0, max(r.max(), s.max(), 1), 50)
        ax.hist(r, bins=bins, alpha=0.55, density=True, color="#2b6cb0", label="Réel")
        ax.hist(s, bins=bins, alpha=0.55, density=True, color="#e07a5f", label="Synthétique")
        ax.set_title(col, fontsize=10)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=8)
    for j in range(len(FEATURES), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Distributions des features (log1p) — réel vs synthétique (tirage i.i.d. CTGAN)")
    fig.tight_layout()
    out_path = os.path.join(output_dir, "fig_features_distributions.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Figure sauvegardée -> {out_path}")


# Plots a real-vs-synthetic bar chart of lag-1 autocorrelations and saves as PNG.
def plot_autocorr(plt, corr_rows, output_dir):
    labels = [r[0] for r in corr_rows]
    real_vals = [r[1] for r in corr_rows]
    synth_vals = [r[2] for r in corr_rows]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, real_vals, width, label="Réel", color="#2b6cb0")
    ax.bar(x + width / 2, synth_vals, width, label="Synthétique", color="#e07a5f")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Autocorrélation lag-1")
    ax.set_title("Corrélations flow→flow au sein des séquences — réel vs synthétique")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(output_dir, "fig_lag1_autocorrelation.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Figure sauvegardée -> {out_path}")


# Writes a Markdown draft summarizing the methodology and results for the report.
def write_summary(output_dir, delta_t_mode, real_used, synth, dt_stats, feature_stats, corr_rows):
    lines = []
    lines.append("# Étape 8 — Résumé méthodologique (brouillon)\n")
    lines.append("## Objectif\n")
    lines.append(
        "Générer des séquences de flows BENIGN synthétiques respectant les corrélations "
        "flow→flow, les inter-arrival times (delta_t) et les patterns burst/idle observés "
        "dans les vraies données CICIDS2017, en s'appuyant sur le CTGAN déjà entraîné et "
        "validé en partie 1.\n"
    )
    lines.append("## Constat qui a guidé l'architecture\n")
    lines.append(
        "Un premier LSTM multivarié (features + delta_t générés ensemble, avec matching sur "
        "le stock CTGAN à chaque pas) a été testé. Diagnostic : l'autocorrélation lag-1 réelle "
        "des 7 features au sein d'une séquence host_pair est quasi nulle (0.02 à 0.05), alors "
        "que le LSTM en inventait une artificiellement (jusqu'à 0.44) — un artefact classique "
        "d'un état caché qui évolue en douceur, pas un signal réel. Générer des features "
        "corrélées aurait donc été **moins** fidèle aux données que de ne pas en générer.\n"
    )
    lines.append("## Architecture retenue\n")
    lines.append(
        "- **Features (7 colonnes)** : tirées **i.i.d.** (indépendamment) du stock de flows "
        "CTGAN déjà entraîné et validé en partie 1 (fidélité marginale identique à celle "
        "documentée pour CTGAN seul).\n"
        f"- **delta_t** : mode `{delta_t_mode}` — "
        + (
            "LSTM mono-variable entraîné en rollout autorégressif, réservé au timing "
            "puisque c'est la seule dimension où une structure temporelle réelle "
            "(bursts/idle) est plausible."
            if delta_t_mode == "lstm" else
            "tirage par bootstrap direct dans la distribution empirique réelle du delta_t, "
            "à chaque pas indépendamment — utilisé après diagnostic montrant que "
            "l'autocorrélation réelle du delta_t est elle aussi trop faible pour justifier "
            "un modèle temporel dédié."
        )
        + "\n"
    )
    lines.append("## Résultats quantitatifs\n")
    lines.append("### Delta_t\n")
    lines.append(
        f"- Réel : mean={dt_stats['real_mean']:.3f}s | Synthétique : mean={dt_stats['synth_mean']:.3f}s\n"
        f"- KS = {dt_stats['ks_stat']:.4f} (p={dt_stats['ks_p']:.4g}) | "
        f"Wasserstein = {dt_stats['wasserstein']:.4f}\n"
        f"- Autocorrélation lag-1 : réel={dt_stats['real_corr']:.4f} | "
        f"synth={dt_stats['synth_corr']:.4f}\n"
    )
    lines.append("### Fidélité marginale des features (réel vs synthétique)\n")
    lines.append("| Feature | KS | p-value | Wasserstein |\n|---|---|---|---|\n")
    for row in feature_stats:
        lines.append(f"| {row['col']} | {row['ks_stat']:.4f} | {row['ks_p']:.4g} | {row['wasserstein']:.4g} |\n")
    lines.append("\n### Corrélations flow→flow (lag-1) — doit rester proche de 0 des deux côtés\n")
    lines.append("| Variable | Réel | Synthétique |\n|---|---|---|\n")
    for label, r, s in corr_rows:
        lines.append(f"| {label} | {r:.4f} | {s:.4f} |\n")
    lines.append("\n## Limites à documenter\n")
    lines.append(
        "- Les corrélations flow→flow sur les 7 features sont volontairement non reproduites "
        "(tirage i.i.d.) car quasi absentes dans les vraies données — ce n'est pas une "
        "limite de la méthode mais une caractéristique du dataset.\n"
        "- CTGAN n'est pas utilisé activement au-delà de la constitution du stock de features "
        "; il reste néanmoins la référence de fidélité marginale établie en partie 1.\n"
        "- [À compléter après lecture des figures] : préciser si le KS sur le delta_t reste "
        "significatif malgré une Wasserstein raisonnable (interprétation à nuancer comme pour "
        "le ΔF1 IDS de la partie 1).\n"
    )
    lines.append("\n## Figures associées\n")
    lines.append(
        "- `fig_delta_t_distribution.png`\n- `fig_features_distributions.png`\n"
        "- `fig_lag1_autocorrelation.png`\n"
    )

    out_path = os.path.join(output_dir, "step8_summary_draft.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"[INFO] Brouillon de résumé méthodologique -> {out_path}")


# Entry point: load real/synth CSVs, compute stats, plot figures, and write the summary.
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    real_used = pd.read_csv(args.real_sequences_csv)
    synth = pd.read_csv(args.synth_sequences_csv)

    missing_r = [c for c in ["seq_id", "delta_t_seconds"] + FEATURES if c not in real_used.columns]
    missing_s = [c for c in ["seq_id", "delta_t_seconds"] + FEATURES if c not in synth.columns]
    if missing_r or missing_s:
        print(f"[ERREUR] Colonnes manquantes -- réel: {missing_r} | synth: {missing_s}", file=sys.stderr)
        sys.exit(1)

    real_dt = real_used["delta_t_seconds"].dropna().clip(lower=0).values
    synth_dt = synth["delta_t_seconds"].dropna().clip(lower=0).values

    ks_stat, ks_p = ks_2samp(real_dt, synth_dt)
    w = wasserstein_distance(real_dt, synth_dt)
    dt_stats = {
        "real_mean": float(real_dt.mean()), "synth_mean": float(synth_dt.mean()),
        "ks_stat": float(ks_stat), "ks_p": float(ks_p), "wasserstein": float(w),
        "real_corr": lag1_corr_within_sequences(real_used, "delta_t_seconds"),
        "synth_corr": lag1_corr_within_sequences(synth, "delta_t_seconds"),
    }

    feature_stats = []
    for col in FEATURES:
        r = real_used[col].dropna().values
        s = synth[col].dropna().values
        ks_stat_c, ks_p_c = ks_2samp(r, s)
        w_c = wasserstein_distance(r, s)
        feature_stats.append({"col": col, "ks_stat": float(ks_stat_c),
                               "ks_p": float(ks_p_c), "wasserstein": float(w_c)})
        print(f"{col:22s} : KS={ks_stat_c:.4f} (p={ks_p_c:.4g}) | Wasserstein={w_c:.4g}")

    corr_rows = [("delta_t_seconds", dt_stats["real_corr"], dt_stats["synth_corr"])]
    for col in FEATURES:
        r_corr = lag1_corr_within_sequences(real_used, col)
        s_corr = lag1_corr_within_sequences(synth, col)
        corr_rows.append((col, r_corr, s_corr))
        print(f"{col:22s} autocorr : réel={r_corr:.4f} | synth={s_corr:.4f}")

    plt = try_import_matplotlib()
    if plt is not None:
        plot_delta_t(plt, real_dt, synth_dt, args.output_dir)
        plot_features(plt, real_used, synth, args.output_dir)
        plot_autocorr(plt, corr_rows, args.output_dir)

    write_summary(args.output_dir, args.delta_t_mode, real_used, synth, dt_stats, feature_stats, corr_rows)

    print("\nTerminé.")


if __name__ == "__main__":
    main()