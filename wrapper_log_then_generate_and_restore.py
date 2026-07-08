# wrapper_log_then_generate_and_restore.py
# Utilitaire pour tester log-transform sans modifier les CSV originaux.
import os
import shutil
import subprocess
import tempfile
import json
import numpy as np
import pandas as pd

SRC_DIR = "data"
SRC_FILE = "Friday-WorkingHours-Morning.pcap_ISCX.csv"  # adapte si nécessaire
FEATURES_TO_LOG = ["Flow Duration", "Packet Length Mean"]
SYNTH_OUT = "flux_synthetiques_benign.csv"
RESTORED_OUT = "flux_synthetiques_benign_restored.csv"
PY = "python"

def prepare_tmp_dir():
    tmp = tempfile.mkdtemp(prefix="tmp_data_for_gen_")
    for f in os.listdir(SRC_DIR):
        if f.lower().endswith(".csv"):
            shutil.copy(os.path.join(SRC_DIR, f), tmp)
    src_path = os.path.join(tmp, SRC_FILE)
    df = pd.read_csv(src_path, low_memory=False)
    offsets = {}
    for col in FEATURES_TO_LOG:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)
            df[col] = np.log1p(df[col])
            offsets[col] = 0.0
    df.to_csv(src_path, index=False)
    with open(os.path.join(tmp, "transform_meta.json"), "w") as f:
        json.dump({"offsets": offsets}, f)
    return tmp

def run_generation(tmp_dir):
    cmd = [PY, "synthetic_generation.py", "--data-dir", tmp_dir,
           "--force-refit", "--n-synthetic", "5000", "--k-max", "12", "--log-transform", "--random-state", "42"]
    print("Lancement:", " ".join(cmd))
    subprocess.check_call(cmd)

def restore_synth(tmp_dir):
    synth_tmp = os.path.join(tmp_dir, SYNTH_OUT)
    synth_local = os.path.join(os.getcwd(), SYNTH_OUT)
    if os.path.exists(synth_tmp):
        synth = pd.read_csv(synth_tmp)
    elif os.path.exists(synth_local):
        synth = pd.read_csv(synth_local)
    else:
        raise FileNotFoundError("Fichier synthétique introuvable après génération.")
    meta = json.load(open(os.path.join(tmp_dir, "transform_meta.json")))
    offsets = meta.get("offsets", {})
    SAFE_EXP_MAX = 700.0
    for col in FEATURES_TO_LOG:
        if col in synth.columns:
            synth[col] = np.minimum(synth[col], SAFE_EXP_MAX)
            with np.errstate(over='ignore'):
                synth[col] = np.expm1(synth[col]) - offsets.get(col, 0.0)
            synth[col] = synth[col].clip(lower=0.0)
    synth.to_csv(RESTORED_OUT, index=False)
    print("Synthétique restauré sauvegardé ->", RESTORED_OUT)

def cleanup(tmp_dir):
    try:
        shutil.rmtree(tmp_dir)
    except Exception:
        pass

if __name__ == "__main__":
    tmp = prepare_tmp_dir()
    try:
        run_generation(tmp)
        restore_synth(tmp)
    finally:
        cleanup(tmp)
