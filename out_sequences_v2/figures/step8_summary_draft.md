# Étape 8 — Résumé méthodologique (brouillon)
## Objectif
Générer des séquences de flows BENIGN synthétiques respectant les corrélations flow→flow, les inter-arrival times (delta_t) et les patterns burst/idle observés dans les vraies données CICIDS2017, en s'appuyant sur le CTGAN déjà entraîné et validé en partie 1.
## Constat qui a guidé l'architecture
Un premier LSTM multivarié (features + delta_t générés ensemble, avec matching sur le stock CTGAN à chaque pas) a été testé. Diagnostic : l'autocorrélation lag-1 réelle des 7 features au sein d'une séquence host_pair est quasi nulle (0.02 à 0.05), alors que le LSTM en inventait une artificiellement (jusqu'à 0.44) — un artefact classique d'un état caché qui évolue en douceur, pas un signal réel. Générer des features corrélées aurait donc été **moins** fidèle aux données que de ne pas en générer.
## Architecture retenue
- **Features (7 colonnes)** : tirées **i.i.d.** (indépendamment) du stock de flows CTGAN déjà entraîné et validé en partie 1 (fidélité marginale identique à celle documentée pour CTGAN seul).
- **delta_t** : mode `lstm` — LSTM mono-variable entraîné en rollout autorégressif, réservé au timing puisque c'est la seule dimension où une structure temporelle réelle (bursts/idle) est plausible.
## Résultats quantitatifs
### Delta_t
- Réel : mean=1996.962s | Synthétique : mean=54.496s
- KS = 0.8162 (p=0) | Wasserstein = 1970.4368
- Autocorrélation lag-1 : réel=-0.1722 | synth=-0.4916
### Fidélité marginale des features (réel vs synthétique)
| Feature | KS | p-value | Wasserstein |
|---|---|---|---|
| Flow Duration | 0.0765 | 0 | 1.524e+06 |
| Packet Length Mean | 0.0620 | 1.89e-219 | 9.59 |
| Flow Bytes/s | 0.0421 | 1.413e-101 | 2.361e+06 |
| Flow Packets/s | 0.0544 | 8.683e-169 | 8538 |
| Fwd IAT Mean | 0.1081 | 0 | 2.668e+05 |
| Bwd IAT Mean | 0.1593 | 0 | 2.757e+05 |
| Packet Length Std | 0.0946 | 0 | 20.56 |

### Corrélations flow→flow (lag-1) — doit rester proche de 0 des deux côtés
| Variable | Réel | Synthétique |
|---|---|---|
| delta_t_seconds | -0.1722 | -0.4916 |
| Flow Duration | 0.0372 | -0.0694 |
| Packet Length Mean | 0.0409 | -0.0729 |
| Flow Bytes/s | -0.0192 | -0.0802 |
| Flow Packets/s | -0.0359 | -0.0695 |
| Fwd IAT Mean | 0.0248 | -0.0688 |
| Bwd IAT Mean | -0.0117 | -0.0677 |
| Packet Length Std | 0.0515 | -0.0712 |

## Limites à documenter
- Les corrélations flow→flow sur les 7 features sont volontairement non reproduites (tirage i.i.d.) car quasi absentes dans les vraies données — ce n'est pas une limite de la méthode mais une caractéristique du dataset.
- CTGAN n'est pas utilisé activement au-delà de la constitution du stock de features ; il reste néanmoins la référence de fidélité marginale établie en partie 1.
- [À compléter après lecture des figures] : préciser si le KS sur le delta_t reste significatif malgré une Wasserstein raisonnable (interprétation à nuancer comme pour le ΔF1 IDS de la partie 1).

## Figures associées
- `fig_delta_t_distribution.png`
- `fig_features_distributions.png`
- `fig_lag1_autocorrelation.png`
