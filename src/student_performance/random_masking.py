"""
Random cell masking on the UCI Student Performance dataset

Hide a small fraction of random cells across all columns, impute them with
several methods, and score the imputed values against the true ones.

Errors are divided by each column's standard deviation, so columns on different
scales (1-5 ratings, 0-20 grades, log absences) can be pooled.

A score of 1.0 means no better than predicting the column mean.

Intermediate dataframes and the final results are saved as CSV/text files
under OUTPUT_DIR
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from ucimlrepo import fetch_ucirepo

CF_NEIGHBOURS = 5  # similar items/students used to predict a hidden cell (paper's k=5)
MASK_FRAC = 0.03  # fraction of observed cells hidden (the paper used 2-3%)
MASK_REPEATS = 30  # number of random masks to average over
MIN_OVERLAP = 10  # min co-rated rows to trust a column/student correlation
NOMINAL = ["Mjob", "Fjob", "reason", "guardian"]  # dropped for masking

OUTPUT_DIR = "output"
RAW_CSV_PATH = os.path.join(OUTPUT_DIR, "01_students_raw.csv")
NUMERIC_CSV_PATH = os.path.join(OUTPUT_DIR, "02_students_numeric.csv")
RESULTS_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "results_summary.csv")
RESULTS_PAIRED_CSV = os.path.join(OUTPUT_DIR, "results_paired_diff.csv")
RESULTS_PER_COLUMN_CSV = os.path.join(OUTPUT_DIR, "results_per_column.csv")
RESULTS_REPORT_TXT = os.path.join(OUTPUT_DIR, "results_report.txt")


###############
# FILE OUTPUT #
###############


def save_csv(df, path, index=False):
    """Save a dataframe to OUTPUT_DIR, creating the folder if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=index)
    print(f"saved {path}")


def save_text(text, path):
    """Save plain text (the printed report) to OUTPUT_DIR."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text + "\n")
    print(f"saved {path}")


####################
# DATA PREPARATION #
####################


def load_data():
    """One dataframe with the 30 features plus G1, G2 and G3."""
    dataset = fetch_ucirepo(id=320)
    students, grades = dataset.data.features, dataset.data.targets
    students_df = pd.concat([students, grades], axis=1)
    print("columns:", students_df.shape[1], "| students:", students_df.shape[0])
    save_csv(students_df, RAW_CSV_PATH)
    return students_df


def encode_numeric(students_df):
    """Numeric dataframe: binary -> 0/1, ordinal kept, absences log1p, nominal columns dropped"""
    # drop the nominal cols
    numeric_df = students_df.drop(
        columns=[column for column in students_df.columns if column in NOMINAL]
    )
    # convert non-numerical values into nums
    for column in numeric_df.columns:
        if not pd.api.types.is_numeric_dtype(numeric_df[column]):
            unique_values = sorted(numeric_df[column].dropna().unique())
            assert len(unique_values) == 2, f"{column} is not binary: {unique_values}"
            numeric_df[column] = numeric_df[column].map(
                {unique_values[0]: 0.0, unique_values[1]: 1.0}
            )

    # scale the absences since most students are 0 but outliers have over 90
    if "absences" in numeric_df.columns:
        numeric_df["absences"] = np.log1p(numeric_df["absences"])

    numeric_df = numeric_df.astype(float)
    save_csv(numeric_df, NUMERIC_CSV_PATH)
    return numeric_df


############
# Imputers #
############
# Each imputer takes a matrix with NaNs (the hidden cells) and returns it filled in.


def column_mean_impute(students_masked):
    """Impute the values based on the col means. This is the baseline."""
    return np.where(
        np.isnan(students_masked), np.nanmean(students_masked, axis=0), students_masked
    )


def standardize_columns(matrix):
    """Z-score each column using observed values only (ignoring NaN).

    Returns (column_mean, column_std, z) so callers can convert a predicted
    z-score back to the column's own units.
    """
    column_mean = np.nanmean(matrix, axis=0)
    column_std = np.nanstd(matrix, axis=0)
    column_std[column_std == 0] = 1.0
    z = (matrix - column_mean) / column_std
    return column_mean, column_std, z


def predict_z_score(known_z, corr_with_target, k):
    """Predict one hidden value's z-score from correlated neighbours.

    known_z:           z-scores of the candidate neighbours, NaN where unknown
                        (n_candidates,)
    corr_with_target:  correlation of every candidate with the target
                        (n_candidates,)
    Used for both item-based CF (candidates = this student's other items) and
    patient-based CF (candidates = other students' value for this item).
    Returns 0.0 (an average candidate) if none can be used.
    """
    # candidates with a known value and a defined correlation to the target
    usable = np.flatnonzero(~np.isnan(known_z) & ~np.isnan(corr_with_target))
    if usable.size == 0:
        return 0.0

    usable_corrs = corr_with_target[usable]
    strongest = np.argsort(-np.abs(usable_corrs))[:k]  # k strongest, sign ignored
    weights = usable_corrs[strongest]
    neighbours_z = known_z[usable[strongest]]

    weight_sum = max(np.abs(weights).sum(), 1.0)
    return (weights * neighbours_z).sum() / weight_sum


def item_cf_impute(students_masked, k=CF_NEIGHBOURS, min_overlap=MIN_OVERLAP):
    """Item-based CF with Pearson correlation (the paper's CF-SYM-PCC).
    Done in z-score space so columns on different scales can be mixed.

    Column-column correlation uses pairwise-complete rows with at least
    `min_overlap` shared observations. A hidden cell is predicted from the k
    columns most correlated with it that the student has observed:
        predicted_z = sum(corr * z) / max(sum(|corr|), 1)
        prediction  = column_mean + column_std * predicted_z
    The max(..., 1) shrinks predictions when the correlations are weak, so a
    lone neighbour with r = 0.1 moves the prediction by 0.1 * z, not 1.0 * z.
    Falls back to the column mean when no neighbour is available.
    """
    # convert to z-score
    column_mean, column_std, z_students = standardize_columns(students_masked)

    # pearson correlation between items (columns)
    col_corr = pd.DataFrame(z_students).corr(min_periods=min_overlap).to_numpy().copy()
    np.fill_diagonal(col_corr, np.nan)  # an item is never its own neighbour

    # mean (fallback)
    students_imputed = np.where(np.isnan(students_masked), column_mean, students_masked)

    hidden_students, hidden_cols = np.where(np.isnan(students_masked))
    for student_idx, target_col in zip(hidden_students, hidden_cols):
        predicted_z = predict_z_score(z_students[student_idx], col_corr[target_col], k)
        students_imputed[student_idx, target_col] = (
            column_mean[target_col] + column_std[target_col] * predicted_z
        )
    return students_imputed


def user_cf_impute(students_masked, k=CF_NEIGHBOURS, min_overlap=MIN_OVERLAP):
    """Patient-based CF with Pearson correlation (the paper's CF-PAT-PCC).

    Mirrors item_cf_impute, but correlates students instead of columns: items
    are standardized first (same z_students item_cf_impute would compute),
    then students are correlated across the items they share. A hidden cell
    (student p, item i) is predicted from the k students most correlated
    with p, using their own z-score on item i:
        predicted_z = sum(corr * z) / max(sum(|corr|), 1)
        prediction  = item_mean + item_std * predicted_z
    Falls back to the item mean when no neighbour is available.
    """
    # same per-item standardizing as item_cf_impute
    item_mean, item_std, z_students = standardize_columns(students_masked)

    # pearson correlation between students (rows), via the transposed matrix
    student_corr = (
        pd.DataFrame(z_students.T).corr(min_periods=min_overlap).to_numpy().copy()
    )
    np.fill_diagonal(student_corr, np.nan)  # a student is never their own neighbour

    # mean (fallback)
    students_imputed = np.where(np.isnan(students_masked), item_mean, students_masked)

    hidden_students, hidden_cols = np.where(np.isnan(students_masked))
    for student_idx, target_col in zip(hidden_students, hidden_cols):
        predicted_z = predict_z_score(
            z_students[:, target_col], student_corr[student_idx], k
        )
        students_imputed[student_idx, target_col] = (
            item_mean[target_col] + item_std[target_col] * predicted_z
        )
    return students_imputed


def standardized(imputer):
    """Standardize columns (using observed values only) around an imputer."""

    def wrapper(students_masked):
        column_mean, column_std, z_students = standardize_columns(students_masked)
        return imputer(z_students) * column_std + column_mean

    return wrapper


# could add more imputers if we want to compare
IMPUTERS = {
    "column mean": column_mean_impute,
    "item-CF (PCC)": item_cf_impute,
    "user-CF (PCC)": user_cf_impute,
    "MICE": standardized(
        lambda z_students: IterativeImputer(max_iter=10, random_state=0).fit_transform(
            z_students
        )
    ),
}


##############
# Evaluation #
##############


def hide_random_cells(students_matrix, mask_fraction, random_generator):
    """Hide a random fraction of the observed cells.

    Returns hidden_mask (True where a cell was hidden) and students_masked
    (a copy of the data with those cells set to NaN).
    """
    is_observed = ~np.isnan(students_matrix)
    picked_at_random = random_generator.random(students_matrix.shape) < mask_fraction
    hidden_mask = is_observed & picked_at_random
    students_masked = np.where(hidden_mask, np.nan, students_matrix)
    return hidden_mask, students_masked


def impute_and_clip(imputer, students_masked):
    """Run an imputer and clip each column to the range of its observed values."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # MICE often warns that it did not fully converge
        students_imputed = imputer(students_masked)
    col_min = np.nanmin(students_masked, axis=0)
    col_max = np.nanmax(students_masked, axis=0)
    return np.clip(students_imputed, col_min, col_max)


def scaled_errors_on_hidden_cells(
    students_imputed, students_matrix, hidden_rows, hidden_cols, column_std
):
    """Imputation errors on the hidden cells only, in units of each column's std."""
    errors = (
        students_imputed[hidden_rows, hidden_cols]
        - students_matrix[hidden_rows, hidden_cols]
    )
    return errors / column_std[hidden_cols]


def collect_scores(students_matrix, imputers, mask_fraction, num_repeats, seed):
    """Hide cells `num_repeats` times and score every imputer on each mask."""

    # calc the std, used for scaling errors
    n_columns = students_matrix.shape[1]
    column_std = np.nanstd(students_matrix, axis=0)
    column_std[column_std == 0] = 1.0

    # containers for the results
    rmse_by_repeat = {name: [] for name in imputers}
    mae_by_repeat = {name: [] for name in imputers}
    squared_error_by_column = {name: np.zeros(n_columns) for name in imputers}
    hidden_count_by_column = np.zeros(n_columns)

    for repeat in range(num_repeats):

        # reproducible rng
        random_generator = np.random.default_rng(seed + repeat)

        # hide the cells
        hidden_mask, students_masked = hide_random_cells(
            students_matrix, mask_fraction, random_generator
        )
        hidden_rows, hidden_cols = np.where(hidden_mask)
        hidden_count_by_column += np.bincount(hidden_cols, minlength=n_columns)

        for method_name, imputer in imputers.items():
            # impute
            students_imputed = impute_and_clip(imputer, students_masked)
            # calc errors
            scaled_errors = scaled_errors_on_hidden_cells(
                students_imputed, students_matrix, hidden_rows, hidden_cols, column_std
            )
            rmse_by_repeat[method_name].append(np.sqrt(np.mean(scaled_errors**2)))
            mae_by_repeat[method_name].append(np.mean(np.abs(scaled_errors)))
            squared_error_by_column[method_name] += np.bincount(
                hidden_cols, weights=scaled_errors**2, minlength=n_columns
            )

    return (
        rmse_by_repeat,
        mae_by_repeat,
        squared_error_by_column,
        hidden_count_by_column,
    )


def summarize_scores(rmse_by_repeat, mae_by_repeat):
    """Mean and std over the repeats, one row per method."""
    return pd.DataFrame(
        {
            "nRMSE mean": {name: np.mean(v) for name, v in rmse_by_repeat.items()},
            "nRMSE std": {name: np.std(v) for name, v in rmse_by_repeat.items()},
            "nMAE mean": {name: np.mean(v) for name, v in mae_by_repeat.items()},
            "nMAE std": {name: np.std(v) for name, v in mae_by_repeat.items()},
        }
    )


def paired_comparison(rmse_by_repeat, reference):
    """Difference in nRMSE vs the reference method, on the same mask each repeat."""
    reference_rmse = np.array(rmse_by_repeat[reference])
    return pd.DataFrame(
        {
            name: {
                "mean diff": np.mean(np.array(rmse_values) - reference_rmse),
                "std diff": np.std(np.array(rmse_values) - reference_rmse),
                "wins": int(np.sum(np.array(rmse_values) < reference_rmse)),
            }
            for name, rmse_values in rmse_by_repeat.items()
            if name != reference
        }
    ).T


def per_column_scores(squared_error_by_column, hidden_count_by_column, column_names):
    """nRMSE of each method for each column, pooled over all repeats."""
    per_column = pd.DataFrame(
        {
            name: np.sqrt(squared_errors / np.maximum(hidden_count_by_column, 1))
            for name, squared_errors in squared_error_by_column.items()
        },
        index=column_names,
    )
    per_column["item-CF minus MICE"] = per_column["item-CF (PCC)"] - per_column["MICE"]
    per_column["user-CF minus MICE"] = per_column["user-CF (PCC)"] - per_column["MICE"]
    return per_column


def format_report(
    n_students, n_columns, mask_fraction, num_repeats, reference, summary, paired_diff, per_column
):
    """Build the printed/saved report as one string."""
    sections = [
        f"=== Random masking: {n_students} students x {n_columns} cols, "
        f"{mask_fraction:.0%} of cells hidden, {num_repeats} repeats "
        f"(errors in column-std units) ===",
        summary.round(3).to_string(),
        "",
        f"=== Paired nRMSE difference vs {reference} "
        f"(negative = better than {reference}) ===",
        paired_diff.round(4).to_string(),
        "",
        "=== Per-column nRMSE (1.0 = no better than predicting the mean) ===",
        per_column.sort_values("item-CF minus MICE").round(3).to_string(),
    ]
    return "\n".join(sections)


def compare_imputers(
    students_df,
    mask_fraction=MASK_FRAC,
    num_repeats=MASK_REPEATS,
    seed=0,
    imputers=IMPUTERS,
    reference="MICE",
):
    """Impute the missing cells, summarize the results, and save everything."""
    students_matrix = students_df.to_numpy(dtype=float)
    n_students, n_columns = students_matrix.shape

    rmse_by_repeat, mae_by_repeat, squared_error_by_column, hidden_count_by_column = (
        collect_scores(students_matrix, imputers, mask_fraction, num_repeats, seed)
    )

    summary = summarize_scores(rmse_by_repeat, mae_by_repeat)
    paired_diff = paired_comparison(rmse_by_repeat, reference)
    per_column = per_column_scores(
        squared_error_by_column, hidden_count_by_column, list(students_df.columns)
    )

    report = format_report(
        n_students, n_columns, mask_fraction, num_repeats, reference,
        summary, paired_diff, per_column,
    )
    print(report)
    save_text(report, RESULTS_REPORT_TXT)
    save_csv(summary.round(3), RESULTS_SUMMARY_CSV, index=True)
    save_csv(paired_diff.round(4), RESULTS_PAIRED_CSV, index=True)
    save_csv(
        per_column.sort_values("item-CF minus MICE").round(3),
        RESULTS_PER_COLUMN_CSV,
        index=True,
    )
    return summary, paired_diff, per_column


def main():
    """driver code"""
    students_df = load_data()
    # all numeric/binary columns, with G3 included as another column
    compare_imputers(encode_numeric(students_df))


if __name__ == "__main__":
    main()
