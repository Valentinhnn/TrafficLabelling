#!/usr/bin/env python3
"""
hawkes_discrete.py
===================
Étape 3 (première méthode) du pipeline dépendances temporelles : modélise
le processus d'arrivée des flows BENIGN comme un processus de comptage
auto-excitant en temps discret (approximation d'un Hawkes, adaptée à la
résolution "minute" réelle du Timestamp CICIDS2017 — cf. diagnostic
temporal_sequences.py : 98.8% des delta_t consécutifs = 0s).

Principe :
    - Découpe le temps en bins fixes (--bin-seconds, défaut 60s).
    - Compte N_t = nombre de flows arrivés dans le bin t.
    - Fit une régression Binomiale Négative de N_t sur (a) les comptages
      retardés N_{t-1}, ..., N_{t-p} (mémoire courte, quelques minutes) et
      (b) des dummies "heure de la journée" (0-23, saisonnalité horaire).
      NÉCESSAIRE : un premier essai sans (b) a montré que la variance des
      comptages est dominée par le cycle heures ouvrées/nuit (rafales de
      centaines de flows/min en journée, silence la nuit), pas par une
      auto-excitation court-terme -> un AR(p) seul convergeait vers un régime
      "idle" permanent en génération, incapable de re-déclencher un burst.
    - Négative Binomiale plutôt que Poisson : les comptages de flows sont
      surdispersés (variance >> moyenne) à cause des bursts, un Poisson
      sous-estimerait systématiquement la variance des séquences générées.
    - Génération : simulation récursive (chaque N_t simulé sert de lag pour
      prédire N_{t+1}), avec l'heure de chaque bin futur connue à l'avance
      (calendrier déterministe, pas une inconnue à simuler).

Usage:
    python3 hawkes_discrete.py --interarrival-csv out_sequences/global_interarrival.csv \
        --bin-seconds 60 --n-lags 5 --n-synthetic-bins 5000 \
        --output-dir out_hawkes | tee hawkes_discrete.log
"""
from __future__ import annotations

import os
import argparse
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import ks_2samp, nbinom

warnings.filterwarnings("ignore")

DEFAULT_RANDOM_STATE = 42


# ----------------------------------------------------------------------
# CHARGEMENT + BINNING
# ----------------------------------------------------------------------
def load_and_bin(interarrival_csv: str, bin_seconds: int) -> pd.DataFrame:
    """
    Charge global_interarrival.csv (colonne 'timestamp'), découpe en bins
    fixes de bin_seconds, et retourne une série de comptages N_t (un compte
    par bin, y compris les bins à 0 flow -> important pour ne pas biaiser
    l'estimation vers le haut en ignorant les périodes idle), avec le
    timestamp de début de chaque bin (nécessaire pour la saisonnalité horaire).
    """
    df = pd.read_csv(interarrival_csv, usecols=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    t0 = df["timestamp"].min()
    bin_idx = ((df["timestamp"] - t0).dt.total_seconds() // bin_seconds).astype(int)

    n_bins_total = bin_idx.max() + 1
    counts = bin_idx.value_counts().reindex(range(n_bins_total), fill_value=0).sort_index()
    counts.index.name = "bin"
    counts.name = "n_flows"

    bin_starts = t0 + pd.to_timedelta(np.arange(n_bins_total) * bin_seconds, unit="s")

    print(f"[INFO] {len(df)} flows -> {n_bins_total} bins de {bin_seconds}s "
          f"({t0} -> {df['timestamp'].max()})")
    print(f"[INFO] Comptages par bin : mean={counts.mean():.3f}, var={counts.var():.3f}, "
          f"ratio var/mean={counts.var() / max(counts.mean(), 1e-9):.2f} "
          f"(>>1 = surdispersion, justifie Negative Binomial vs Poisson)")
    print(f"[INFO] {(counts == 0).mean() * 100:.1f}% de bins vides (idle)")

    out = counts.reset_index()
    out["bin_start"] = bin_starts
    out["hour"] = out["bin_start"].dt.hour
    return out, t0


# ----------------------------------------------------------------------
# FEATURES AUTORÉGRESSIVES (lags)
# ----------------------------------------------------------------------
HOUR_COLS = [f"hour_{h}" for h in range(1, 24)]  # hour_0 = baseline (référence)


def build_lag_features(binned: pd.DataFrame, n_lags: int) -> pd.DataFrame:
    df = binned[["n_flows", "hour"]].copy()
    for lag in range(1, n_lags + 1):
        # log1p plutôt que la valeur brute : un noyau Hawkes/AR en covariable
        # linéaire brute donne un effet multiplicatif exp(coef * N) qui explose
        # sans plafond naturel quand N est grand (burst) -> instabilité en
        # rollout récursif (chaque pas amplifie le suivant). log1p sature
        # l'effet des grosses valeurs, standard en modélisation de comptages
        # auto-régressive (INGARCH/GARMA).
        df[f"lag_{lag}"] = np.log1p(df["n_flows"].shift(lag))
    df = df.dropna().reset_index(drop=True)

    hour_dummies = pd.get_dummies(df["hour"], prefix="hour", drop_first=False).astype(float)
    for col in HOUR_COLS:
        if col not in hour_dummies.columns:
            hour_dummies[col] = 0.0
    hour_dummies = hour_dummies[HOUR_COLS]  # hour_0 exclu = baseline

    return pd.concat([df, hour_dummies], axis=1)


def hour_dummy_row(hour: int) -> list:
    """Vecteur de dummies pour une heure donnée, même encodage que build_lag_features."""
    return [1.0 if hour == h else 0.0 for h in range(1, 24)]


# ----------------------------------------------------------------------
# FIT NEGATIVE BINOMIAL AUTORÉGRESSIF + SAISONNALITÉ HORAIRE
# ----------------------------------------------------------------------
def fit_negbin_ar(lag_df: pd.DataFrame, n_lags: int):
    lag_cols = [f"lag_{i}" for i in range(1, n_lags + 1)]
    feature_cols = lag_cols + HOUR_COLS
    X = sm.add_constant(lag_df[feature_cols])
    y = lag_df["n_flows"]

    model = sm.NegativeBinomial(y, X)
    result = model.fit(disp=False, maxiter=300)

    print("\n[FIT] Régression Binomiale Négative auto-régressive + saisonnalité horaire :")
    print(f"      (lags {lag_cols}, dummies heure 1-23, heure 0 = référence/baseline)")
    print(result.summary().tables[1])

    return result


# ----------------------------------------------------------------------
# GÉNÉRATION (simulation récursive)
# ----------------------------------------------------------------------
def simulate_counts(result, n_lags: int, bin_hours: np.ndarray, init_lags: np.ndarray,
                     random_state: int = DEFAULT_RANDOM_STATE,
                     reference_counts: np.ndarray = None, max_count_cap: float = None) -> np.ndarray:
    """
    Simule len(bin_hours) comptages successifs, en réinjectant chaque valeur
    générée comme lag pour le pas suivant (rollout autorégressif), avec
    l'heure de chaque bin (bin_hours) comme covariable de saisonnalité connue
    à l'avance. init_lags = les n_lags dernières valeurs réelles observées.

    reference_counts : série complète des comptages réels -> sert à calculer
    un plafond par défaut robuste basé sur le percentile 99.5 plutôt que le
    max brut (un seul outlier extrême rendrait le plafond arbitrairement
    élevé et inefficace).
    max_count_cap : plafond appliqué (a) au prédicteur linéaire avant exp()
    et (b) aux valeurs réinjectées en lag, pour éviter une explosion
    numérique/statistique. Un processus NB2 fortement surdispersé (alpha~1)
    a une variance qui croît en carré avec mu -> risque réel d'auto-
    amplification sur quelques pas de la simulation récursive, même avec
    des lags en log1p. Défaut : 4x le percentile 99.5 des comptages réels.
    """
    rng = np.random.default_rng(random_state)
    params = result.params
    coefs = params.drop("alpha") if "alpha" in params.index else params
    alpha_disp = params["alpha"] if "alpha" in params.index else 1.0
    feature_order = list(coefs.index)  # ['const', 'lag_1', ..., 'hour_1', ...]

    if max_count_cap is None:
        if reference_counts is not None and len(reference_counts) > 0:
            p995 = np.percentile(reference_counts, 99.5)
        else:
            p995 = init_lags.max()
        max_count_cap = 4.0 * max(p995, 1)
        print(f"[INFO] Plafond de sécurité auto : {max_count_cap:.0f} "
              f"(4x le percentile 99.5 des comptages réels = {p995:.0f})")
    # borne sur le prédicteur linéaire (avant exp) : garantit mu <= max_count_cap
    # même si les coefficients divergent (séparation) -> plus robuste que
    # borner mu après coup.
    linpred_cap = np.log(max_count_cap)

    history = list(init_lags[-n_lags:])
    generated = []
    n_capped = 0

    for hour in bin_hours:
        lags_raw = history[-n_lags:][::-1]  # lag_1 = plus récent, valeur brute
        lags = [np.log1p(v) for v in lags_raw]  # même transformation qu'à l'entraînement
        hour_dummies = hour_dummy_row(int(hour))
        row = {"const": 1.0}
        row.update({f"lag_{i+1}": lags[i] for i in range(n_lags)})
        row.update({HOUR_COLS[i]: hour_dummies[i] for i in range(len(HOUR_COLS))})
        x_row = np.array([row[c] for c in feature_order])

        linpred = np.dot(coefs.values, x_row)
        linpred = np.clip(linpred, -20.0, linpred_cap)  # -20 -> mu~2e-9, borne basse large
        mu = np.exp(linpred)
        mu = max(mu, 1e-6)

        # paramétrisation NB2 (statsmodels) -> conversion vers (n, p) de scipy
        var = mu + alpha_disp * mu ** 2
        if var > mu * (1 + 1e-9):  # marge pour éviter var-mu ~ 0 par erreur d'arrondi
            n = mu ** 2 / (var - mu)
            p = mu / var
        else:
            # quasi-Poisson (alpha*mu^2 négligeable devant mu) : fallback n grand
            n = 1e6
            p = n / (n + mu)

        sample = nbinom.rvs(n, p, random_state=rng)
        if sample > max_count_cap:
            n_capped += 1
            sample = int(max_count_cap)
        generated.append(sample)
        history.append(sample)

    if n_capped > 0:
        print(f"[DIAG] {n_capped}/{len(bin_hours)} bins ({100*n_capped/len(bin_hours):.2f}%) "
              f"tronqués au plafond de sécurité -> à interpréter comme limite de la méthode, "
              f"pas comme un artefact silencieux")

    return np.array(generated)


# ----------------------------------------------------------------------
# VALIDATION
# ----------------------------------------------------------------------
def validate(real_counts: np.ndarray, synth_counts: np.ndarray):
    print("\n=== Validation ===")
    print(f"Réel   : mean={real_counts.mean():.3f}, var={real_counts.var():.3f}, "
          f"%zero={np.mean(real_counts == 0) * 100:.1f}%, max={real_counts.max()}")
    print(f"Synth  : mean={synth_counts.mean():.3f}, var={synth_counts.var():.3f}, "
          f"%zero={np.mean(synth_counts == 0) * 100:.1f}%, max={synth_counts.max()}")

    ks_stat, ks_p = ks_2samp(real_counts, synth_counts)
    print(f"KS (distribution des comptages) : stat={ks_stat:.4f}, p={ks_p:.4e}")

    def acf(x, lag):
        x = x - x.mean()
        if len(x) <= lag:
            return np.nan
        num = np.sum(x[:-lag] * x[lag:])
        den = np.sum(x ** 2)
        return num / den if den > 0 else np.nan

    for lag in (1, 2):
        acf_real = acf(real_counts.astype(float), lag)
        acf_synth = acf(synth_counts.astype(float), lag)
        print(f"Autocorrélation lag-{lag} : réel={acf_real:.4f}, synth={acf_synth:.4f}")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Modèle de comptage auto-excitant (Hawkes discrétisé) sur les arrivées de flows")
    p.add_argument("--interarrival-csv", required=True,
                   help="Fichier produit par temporal_sequences.py (colonne 'timestamp')")
    p.add_argument("--bin-seconds", type=int, default=60,
                   help="Taille des bins temporels (défaut 60s, aligné sur la résolution native du dataset)")
    p.add_argument("--n-lags", type=int, default=5,
                   help="Nombre de lags auto-régressifs (mémoire du processus, en nombre de bins)")
    p.add_argument("--n-synthetic-bins", type=int, default=None,
                   help="Nombre de bins à générer (défaut : même longueur que la série réelle)")
    p.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    p.add_argument("--max-count-cap", type=float, default=None,
                   help="Plafond de sécurité sur le comptage simulé/prédicteur linéaire "
                        "(défaut : 5x le max observé dans les données réelles)")
    p.add_argument("--output-dir", default="out_hawkes")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    binned, t0 = load_and_bin(args.interarrival_csv, args.bin_seconds)
    counts = binned["n_flows"]

    lag_df = build_lag_features(binned, args.n_lags)
    result = fit_negbin_ar(lag_df, args.n_lags)

    n_synth = args.n_synthetic_bins or len(counts)
    init_lags = counts.values[:args.n_lags] if len(counts) >= args.n_lags else counts.values

    if n_synth == len(counts):
        # même longueur que le réel -> réutilise directement le calendrier réel
        bin_hours = binned["hour"].values
    else:
        # longueur différente -> reconstruit un calendrier continu à partir de t0
        future_starts = t0 + pd.to_timedelta(np.arange(n_synth) * args.bin_seconds, unit="s")
        bin_hours = future_starts.hour.values

    synth_counts = simulate_counts(result, args.n_lags, bin_hours, init_lags, args.random_state,
                                    reference_counts=counts.values, max_count_cap=args.max_count_cap)

    validate(counts.values, synth_counts)

    out_path = os.path.join(args.output_dir, "synthetic_counts.csv")
    pd.DataFrame({"bin": range(n_synth), "n_flows_synthetic": synth_counts}).to_csv(out_path, index=False)
    print(f"\n[INFO] Comptages synthétiques exportés -> {out_path}")

    real_path = os.path.join(args.output_dir, "real_counts.csv")
    binned.to_csv(real_path, index=False)
    print(f"[INFO] Comptages réels (référence) exportés -> {real_path}")

    print("\nTerminé.")


if __name__ == "__main__":
    main()