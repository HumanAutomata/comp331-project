"""
Random cell masking on the Iris dataset

Hide a small fraction of random cells across all columns, impute them with
several methods, and score the imputed values against the true ones.

Errors are divided by each column's standard deviation, so columns on
different scales (centimetres for the measurements, 0/1 for the species
flags) can be pooled.

A score of 1.0 means no better than predicting the column mean.

Intermediate dataframes and the final results are saved as CSV/text files
under OUTPUT_DIR, so you can inspect the data at each step outside Python.
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

CF_NEIGHBOURS = 5  # similar items/samples used to predict a hidden cell (paper's k=5)
MASK_FRAC = 0.03  # fraction of observed cells hidden (the paper used 2-3%)
MASK_REPEATS = 30  # number of random masks to average over
# Iris only has 7 columns after encoding, so two flowers can share at most 7
# observed items -- the student script's min_overlap=10 could never be met.
# Use a threshold scaled to this dataset's own (much smaller) number of columns.
MIN_OVERLAP = 5

OUTPUT_DIR = "output"
RAW_CSV_PATH = os.path.join(OUTPUT_DIR, "01_flowers_raw.csv")
NUMERIC_CSV_PATH = os.path.join(OUTPUT_DIR, "02_flowers_numeric.csv")
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
    """One dataframe with the 4 flower measurements plus the species label."""
    iris = load_iris(as_frame=True)
    flowers_df = iris.data.copy()
    flowers_df["species"] = iris.target_names[iris.target]
    print("columns:", flowers_df.shape[1], "| flowers:", flowers_df.shape[0])
    save_csv(flowers_df, RAW_CSV_PATH)
    return flowers_df


def encode_numeric(flowers_df):
    """Numeric dataframe: the 4 measurements kept as-is, species one-hot
    encoded into 3 binary flags (so every column can be correlated with the
    others, the same way the student script's binary yes/no columns are)."""
    numeric_df = flowers_df.drop(columns=["species"])
    for species_name in sorted(flowers_df["species"].unique()):
        numeric_df[f"species_{species_name}"] = (
            flowers_df["species"] == species_name
        ).astype(float)

    numeric_df = numeric_df.astype(float)
    save_csv(numeric_df, NUMERIC_CSV_PATH)
    return numeric_df


############
# Imputers #
############
# Each imputer takes a matrix with NaNs (the hidden cells) and returns it filled in.


def column_mean_impute(samples_masked):
    """Impute the values based on the col means. This is the baseline."""
    return np.where(
        np.isnan(samples_masked), np.nanmean(samples_masked, axis=0), samples_masked
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
    Used for both item-based CF (candidates = this sample's other items) and
    sample-based CF (candidates = other samples' value for this item).
    Returns 0.0 (an average candidate) if none can be used.
    """
    usable = np.flatnonzero(~np.isnan(known_z) & ~np.isnan(corr_with_target))
    if usable.size == 0:
        return 0.0

    usable_corrs = corr_with_target[usable]
    strongest = np.argsort(-np.abs(usable_corrs))[:k]  # k strongest, sign ignored
    weights = usable_corrs[strongest]
    neighbours_z = known_z[usable[strongest]]

    weight_sum = max(np.abs(weights).sum(), 1.0)
    return (weights * neighbours_z).sum() / weight_sum


def item_cf_impute(samples_masked, k=CF_NEIGHBOURS, min_overlap=MIN_OVERLAP):
    """Item-based CF with Pearson correlation (the paper's CF-SYM-PCC).
    Done in z-score space so columns on different scales can be mixed.

    Column-column correlation uses pairwise-complete rows with at least
    `min_overlap` shared observations. A hidden cell is predicted from the k
    columns most correlated with it that the sample has observed:
        predicted_z = sum(corr * z) / max(sum(|corr|), 1)
        prediction  = column_mean + column_std * predicted_z
    Falls back to the column mean when no neighbour is available.
    """
    column_mean, column_std, z_samples = standardize_columns(samples_masked)

    col_corr = pd.DataFrame(z_samples).corr(min_periods=min_overlap).to_numpy().copy()
    np.fill_diagonal(col_corr, np.nan)  # an item is never its own neighbour

    samples_imputed = np.where(np.isnan(samples_masked), column_mean, samples_masked)

    hidden_rows, hidden_cols = np.where(np.isnan(samples_masked))
    for row_idx, target_col in zip(hidden_rows, hidden_cols):
        predicted_z = predict_z_score(z_samples[row_idx], col_corr[target_col], k)
        samples_imputed[row_idx, target_col] = (
            column_mean[target_col] + column_std[target_col] * predicted_z
        )
    return samples_imputed


def user_cf_impute(samples_masked, k=CF_NEIGHBOURS, min_overlap=MIN_OVERLAP):
    """Sample-based CF with Pearson correlation (the paper's CF-PAT-PCC).

    Mirrors item_cf_impute, but correlates samples instead of columns: items
    are standardized first (same z_samples item_cf_impute would compute),
    then samples are correlated across the items they share. A hidden cell
    (sample p, item i) is predicted from the k samples most correlated with
    p, using their own z-score on item i:
        predicted_z = sum(corr * z) / max(sum(|corr|), 1)
        prediction  = item_mean + item_std * predicted_z
    Falls back to the item mean when no neighbour is available.
    """
    item_mean, item_std, z_samples = standardize_columns(samples_masked)

    sample_corr = (
        pd.DataFrame(z_samples.T).corr(min_periods=min_overlap).to_numpy().copy()
    )
    np.fill_diagonal(sample_corr, np.nan)  # a sample is never its own neighbour

    samples_imputed = np.where(np.isnan(samples_masked), item_mean, samples_masked)

    hidden_rows, hidden_cols = np.where(np.isnan(samples_masked))
    for row_idx, target_col in zip(hidden_rows, hidden_cols):
        predicted_z = predict_z_score(z_samples[:, target_col], sample_corr[row_idx], k)
        samples_imputed[row_idx, target_col] = (
            item_mean[target_col] + item_std[target_col] * predicted_z
        )
    return samples_imputed


def standardized(imputer):
    """Standardize columns (using observed values only) around an imputer."""

    def wrapper(samples_masked):
        column_mean, column_std, z_samples = standardize_columns(samples_masked)
        return imputer(z_samples) * column_std + column_mean

    return wrapper


IMPUTERS = {
    "column mean": column_mean_impute,
    "item-CF (PCC)": item_cf_impute,
    "user-CF (PCC)": user_cf_impute,
    "MICE": standardized(
        lambda z_samples: IterativeImputer(max_iter=10, random_state=0).fit_transform(
            z_samples
        )
    ),
}


##############
# Evaluation #
##############


def hide_random_cells(samples_matrix, mask_fraction, random_generator):
    """Hide a random fraction of the observed cells.

    Returns hidden_mask (True where a cell was hidden) and samples_masked
    (a copy of the data with those cells set to NaN).
    """
    is_observed = ~np.isnan(samples_matrix)
    picked_at_random = random_generator.random(samples_matrix.shape) < mask_fraction
    hidden_mask = is_observed & picked_at_random
    samples_masked = np.where(hidden_mask, np.nan, samples_matrix)
    return hidden_mask, samples_masked


def impute_and_clip(imputer, samples_masked):
    """Run an imputer and clip each column to the range of its observed values."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # MICE often warns that it did not fully converge
        samples_imputed = imputer(samples_masked)
    col_min = np.nanmin(samples_masked, axis=0)
    col_max = np.nanmax(samples_masked, axis=0)
    return np.clip(samples_imputed, col_min, col_max)


def scaled_errors_on_hidden_cells(
    samples_imputed, samples_matrix, hidden_rows, hidden_cols, column_std
):
    """Imputation errors on the hidden cells only, in units of each column's std."""
    errors = (
        samples_imputed[hidden_rows, hidden_cols]
        - samples_matrix[hidden_rows, hidden_cols]
    )
    return errors / column_std[hidden_cols]


def collect_scores(samples_matrix, imputers, mask_fraction, num_repeats, seed):
    """Hide cells `num_repeats` times and score every imputer on each mask."""
    n_columns = samples_matrix.shape[1]
    column_std = np.nanstd(samples_matrix, axis=0)
    column_std[column_std == 0] = 1.0

    rmse_by_repeat = {name: [] for name in imputers}
    mae_by_repeat = {name: [] for name in imputers}
    squared_error_by_column = {name: np.zeros(n_columns) for name in imputers}
    hidden_count_by_column = np.zeros(n_columns)

    for repeat in range(num_repeats):
        random_generator = np.random.default_rng(seed + repeat)

        hidden_mask, samples_masked = hide_random_cells(
            samples_matrix, mask_fraction, random_generator
        )
        hidden_rows, hidden_cols = np.where(hidden_mask)
        hidden_count_by_column += np.bincount(hidden_cols, minlength=n_columns)

        for method_name, imputer in imputers.items():
            samples_imputed = impute_and_clip(imputer, samples_masked)
            scaled_errors = scaled_errors_on_hidden_cells(
                samples_imputed, samples_matrix, hidden_rows, hidden_cols, column_std
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
    n_samples, n_columns, mask_fraction, num_repeats, reference,
    summary, paired_diff, per_column,
):
    """Build the printed/saved report as one string."""
    sections = [
        f"=== Random masking: {n_samples} flowers x {n_columns} cols, "
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
    samples_df,
    mask_fraction=MASK_FRAC,
    num_repeats=MASK_REPEATS,
    seed=0,
    imputers=IMPUTERS,
    reference="MICE",
):
    """Impute the missing cells, summarize the results, and save everything."""
    samples_matrix = samples_df.to_numpy(dtype=float)
    n_samples, n_columns = samples_matrix.shape

    rmse_by_repeat, mae_by_repeat, squared_error_by_column, hidden_count_by_column = (
        collect_scores(samples_matrix, imputers, mask_fraction, num_repeats, seed)
    )

    summary = summarize_scores(rmse_by_repeat, mae_by_repeat)
    paired_diff = paired_comparison(rmse_by_repeat, reference)
    per_column = per_column_scores(
        squared_error_by_column, hidden_count_by_column, list(samples_df.columns)
    )

    report = format_report(
        n_samples, n_columns, mask_fraction, num_repeats, reference,
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
    flowers_df = load_data()
    compare_imputers(encode_numeric(flowers_df))


if __name__ == "__main__":
    main()
