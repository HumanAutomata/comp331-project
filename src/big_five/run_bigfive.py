"""
Run both team scripts on a second dataset: the Big Five personality test (IPIP-FFM).

Data: a 3,000-respondent sample of the Open Psychometrics IPIP-FFM responses, distributed as BFPT.csv
with R Journal article RJ-2025-021 (github.com/rjournal/rjournal.github.io). 50 Likert questions
(1-5), 10 per trait (EXT, EST, AGR, CSN, OPN). In this data 0 means "question skipped", so 0 -> NaN.
Rows that are entirely empty (6) or entirely skipped (3) carry no answers and are dropped.

The team's scripts in src/student_performance are reused unchanged; only the loader changes.

Usage (from this folder): python run_bigfive.py BFPT.csv masking
                         python run_bigfive.py BFPT.csv predict
"""
import sys
import types

import numpy as np
import pandas as pd

sys.modules.setdefault("ucimlrepo", types.SimpleNamespace(fetch_ucirepo=None))
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "student_performance"))  # random_masking.py, predict_g3.py
sys.path.insert(0, str(HERE))


def load_bigfive(path):
    df = pd.read_csv(path, sep=";").dropna(how="all")
    df = df.replace(0, np.nan).dropna(how="all").astype(float)
    print("Big Five matrix:", df.shape, "| real missing (skipped) cells:", int(df.isna().sum().sum()))
    return df


if __name__ == "__main__":
    path, mode = sys.argv[1], sys.argv[2]
    if mode == "masking":
        import random_masking as rm
        rm.compare_imputers(load_bigfive(path))
    else:
        import predict_target as yp
        yp.load_data = load_bigfive  # same experiment, different data
        sys.argv = [sys.argv[0], path]
        yp.main()
