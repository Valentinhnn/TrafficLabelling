# Rapport d'analyse statistique — CICIDS2017

## Pourquoi la loi normale est une approximation insuffisante

Les features de flux réseau sont fortement asymétriques, bornées à zéro, et présentent une queue lourde.

## Pourquoi KDE et GMM sont plus adaptés

Le KDE est non paramétrique; le GMM est génératif et permet d'échantillonner du trafic synthétique réaliste.

## Tableau comparatif (résumé)


| feature                |   n_benign |   n_attack |   mu_log_benign |   sigma_log_benign |   mu_log_attack |   sigma_log_attack |   k_gmm_benign |   k_gmm_attack | gmm_weights_benign                  | gmm_means_log_benign                  | gmm_weights_attack                  | gmm_means_log_attack                  |
|:-----------------------|-----------:|-----------:|----------------:|-------------------:|----------------:|-------------------:|---------------:|---------------:|:------------------------------------|:--------------------------------------|:------------------------------------|:--------------------------------------|
| Flow Duration          |    2272982 |     557646 |          9.6958 |             5.1085 |         10.8776 |             6.6401 |              5 |              5 | [0.357, 0.344, 0.108, 0.128, 0.063] | [10.99, 4.934, 18.286, 15.503, 1.337] | [0.297, 0.266, 0.143, 0.192, 0.102] | [18.339, 3.93, 9.6, 15.114, 1.544]    |
| Total Fwd Packets      |    2273097 |     557646 |          1.3979 |             0.8056 |          1.4227 |             0.6    |              5 |              5 | [0.149, 0.438, 0.032, 0.24, 0.141]  | [1.654, 1.099, 3.686, 0.693, 2.701]   | [0.137, 0.296, 0.014, 0.425, 0.128] | [1.386, 0.693, 2.798, 2.005, 1.099]   |
| Total Backward Packets |    2273097 |     557646 |          1.1676 |             0.9329 |          1.1121 |             0.8164 |              5 |              5 | [0.272, 0.099, 0.154, 0.33, 0.144]  | [0.693, 3.095, 0.0, 1.099, 2.065]     | [0.376, 0.315, 0.209, 0.048, 0.052] | [1.939, 0.693, 0.0, 2.078, 1.609]     |
| Packet Length Mean     |    2273097 |     557646 |          3.5437 |             1.8266 |          3.6687 |             2.8949 |              5 |              5 | [0.141, 0.167, 0.196, 0.128, 0.368] | [3.719, 1.796, 5.658, 0.0, 4.245]     | [0.436, 0.3, 0.133, 0.108, 0.023]   | [6.86, 1.288, 0.0, 1.987, 4.754]      |
| Flow IAT Mean          |    2272982 |     557646 |          8.4901 |             4.3584 |          9.622  |             5.6216 |              5 |              5 | [0.337, 0.34, 0.158, 0.1, 0.065]    | [9.903, 4.29, 14.82, 12.523, 1.284]   | [0.283, 0.272, 0.224, 0.102, 0.118] | [15.842, 3.935, 13.393, 1.509, 8.002] |


## Limites identifiées

- Le GMM peut sous-estimer les queues extrêmement lourdes.
- Le log1p facilite le fit mais complique l'interprétation directe des paramètres.
