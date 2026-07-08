"""
build_tableau_excel.py
=======================
Transforme tableau_comparatif.csv (généré par stats_features.py) en un
fichier Excel propre et lisible :

- Feuille "Résumé" : une ligne par feature, colonnes numériques simples.
- Feuille "Composantes GMM" : format long (une ligne par composante du
  mélange, par feature, par classe) -> filtrable/triable dans Excel.

Usage :
    python build_tableau_excel.py
(lit tableau_comparatif.csv dans le dossier courant, écrit tableau_comparatif.xlsx)
"""

import ast
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

INPUT_CSV = "tableau_comparatif.csv"
OUTPUT_XLSX = "tableau_comparatif.xlsx"

HEADER_FILL = PatternFill("solid", fgColor="305496")
HEADER_FONT = Font(bold=True, color="FFFFFF", name="Calibri")
BODY_FONT = Font(name="Calibri")
CENTER = Alignment(horizontal="center", vertical="center")


def parse_list(cell) -> list:
    """Les colonnes gmm_* sont stockées comme '[0.1, 0.2, ...]' (texte) dans le CSV."""
    if isinstance(cell, list):
        return cell
    if pd.isna(cell):
        return []
    return ast.literal_eval(cell)


def style_header(ws, n_cols: int, row: int = 1):
    for col in range(1, n_cols + 1):
        cell = ws.cell(row=row, column=col)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = CENTER


def autosize_columns(ws, df: pd.DataFrame):
    for i, col in enumerate(df.columns, start=1):
        max_len = max(df[col].astype(str).map(len).max(), len(str(col))) + 3
        ws.column_dimensions[get_column_letter(i)].width = min(max_len, 40)


def write_df(ws, df: pd.DataFrame, number_formats: dict | None = None):
    number_formats = number_formats or {}
    ws.append(list(df.columns))
    for _, row in df.iterrows():
        ws.append(list(row))
    style_header(ws, len(df.columns))
    autosize_columns(ws, df)
    ws.freeze_panes = "A2"

    for col_idx, col_name in enumerate(df.columns, start=1):
        fmt = number_formats.get(col_name)
        if fmt:
            for row_idx in range(2, len(df) + 2):
                ws.cell(row=row_idx, column=col_idx).number_format = fmt
        for row_idx in range(2, len(df) + 2):
            ws.cell(row=row_idx, column=col_idx).font = BODY_FONT


def main():
    raw = pd.read_csv(INPUT_CSV)

    # --- Feuille "Résumé" : on retire les colonnes listes ---
    list_cols = [c for c in raw.columns if c.startswith("gmm_")]
    resume = raw.drop(columns=list_cols).copy()
    resume = resume.rename(columns={
        "feature": "Feature",
        "n_benign": "N (BENIGN)",
        "n_attack": "N (Attaque)",
        "mu_log_benign": "μ log (BENIGN)",
        "sigma_log_benign": "σ log (BENIGN)",
        "mu_log_attack": "μ log (Attaque)",
        "sigma_log_attack": "σ log (Attaque)",
        "k_gmm_benign": "k GMM (BENIGN)",
        "k_gmm_attack": "k GMM (Attaque)",
    })

    # --- Feuille "Composantes GMM" : format long ---
    long_rows = []
    for _, row in raw.iterrows():
        feature = row["feature"]
        for classe, w_col, m_col in [
            ("BENIGN", "gmm_weights_benign", "gmm_means_log_benign"),
            ("Attaque", "gmm_weights_attack", "gmm_means_log_attack"),
        ]:
            weights = parse_list(row[w_col])
            means = parse_list(row[m_col])
            # trie les composantes par poids décroissant (plus lisible)
            order = sorted(range(len(weights)), key=lambda i: -weights[i])
            for rank, i in enumerate(order, start=1):
                long_rows.append({
                    "Feature": feature,
                    "Classe": classe,
                    "Composante": rank,
                    "Poids": round(weights[i], 4),
                    "Moyenne (log1p)": round(means[i], 4),
                })

    composantes = pd.DataFrame(long_rows)

    # --- Écriture Excel ---
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Résumé"
    write_df(ws1, resume, number_formats={
        "μ log (BENIGN)": "0.000", "σ log (BENIGN)": "0.000",
        "μ log (Attaque)": "0.000", "σ log (Attaque)": "0.000",
        "N (BENIGN)": "#,##0", "N (Attaque)": "#,##0",
    })

    ws2 = wb.create_sheet("Composantes GMM")
    write_df(ws2, composantes, number_formats={
        "Poids": "0.000", "Moyenne (log1p)": "0.000",
    })

    wb.save(OUTPUT_XLSX)
    print(f"Fichier généré : {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
