# inspect_gmm.py
import joblib
import numpy as np

m = joblib.load('gmm_mv_benign.joblib')
gmm = m.get('gmm') if isinstance(m, dict) else m
features = m.get('features', ['Flow Duration','Packet Length Mean'])
print("Features:", features)
print("n_components:", gmm.n_components)
print("Weights:", np.round(gmm.weights_,4))
print("Means (all components):")
for i, mu in enumerate(gmm.means_):
    print(f"  comp {i}: mean = {np.round(mu,3)}")
print("Covariances diag (all components):")
for i, cov in enumerate(gmm.covariances_):
    diag = np.round(np.diag(cov),3)
    print(f"  comp {i}: diag = {diag}")
