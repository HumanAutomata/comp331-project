"""
Random cell masking on the UCI Student Performance dataset

Hide a small fraction of random cells across all columns, impute them with
several methods, and score the imputed values against the true ones.

Errors are divided by each column's standard deviation, so columns on different
scales (1-5 ratings, 0-20 grades, log absences) can be pooled.

A score of 1.0 means no better than predicting the column mean.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from ucimlrepo import fetch_ucirepo

ITEM_CF_NEIGHBOURS = 5  # similar columns used to predict a hidden cell
USER_CF_NEIGHBOURS = 10  # similar students used to predict a hidden cell
MASK_FRAC = 0.03  # fraction of observed cells hidden (the paper used 2-3%)
MASK_REPEATS = 30  # number of random masks to average over
MIN_OVERLAP = 10  # min co-rated rows to trust a column-column correlation
NOMINAL = ["Mjob", "Fjob", "reason", "guardian"]  # dropped for masking

####################
# DATA PREPARATION #
####################


def load_data():
    """One dataframe with the 30 features plus G1, G2 and G3."""
    dataset = fetch_ucirepo(id=320)
    students, grades = dataset.data.features, dataset.data.targets
    students_df = pd.concat([students, grades], axis=1)
    print("columns:", students_df.shape[1], "| students:", students_df.shape[0])
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
    return numeric_df.astype(float)


############
# Imputers #
############
# Each imputer takes a matrix with NaNs (the hidden cells) and returns it filled in.


def column_mean_impute(students_masked):
    """Impute the values based on the col means This is the baseline."""
    return np.where(
        np.isnan(students_masked), np.nanmean(students_masked, axis=0), students_masked
    )


def predict_z_score(student_z, corr_with_target, k):
    """Predict one hidden cell's z-score from the student's other columns.
    student_z:        this student's z-scores, NaN where hidden (n_columns,)
    corr_with_target: correlation of every column with the target column (n_columns,)
    Returns 0.0 (an average student) if no column can be used.
    """
    # columns where the student has a value and the correlation is defined
    usable_cols = np.flatnonzero(~np.isnan(student_z) & ~np.isnan(corr_with_target))
    if usable_cols.size == 0:
        return 0.0

    usable_corrs = corr_with_target[usable_cols]
    strongest = np.argsort(-np.abs(usable_corrs))[:k]  # k strongest, sign ignored
    weights = usable_corrs[strongest]
    neighbours_z = student_z[usable_cols[strongest]]

    weight_sum = max(np.abs(weights).sum(), 1.0)
    return (weights * neighbours_z).sum() / weight_sum


def item_cf_impute(students_masked, k=ITEM_CF_NEIGHBOURS, min_overlap=MIN_OVERLAP):
    """Item-based CF with Pearson correlation.
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
    column_mean = np.nanmean(students_masked, axis=0)
    column_std = np.nanstd(students_masked, axis=0)
    column_std[column_std == 0] = 1.0
    z_students = (students_masked - column_mean) / column_std

    # pearson correlation
    col_corr = pd.DataFrame(z_students).corr(min_periods=min_overlap).to_numpy().copy()
    # fill in the diagonal
    np.fill_diagonal(col_corr, np.nan)

    # mean (fallback)
    students_imputed = np.where(np.isnan(students_masked), column_mean, students_masked)

    hidden_students, hidden_cols = np.where(np.isnan(students_masked))
    for student_idx, target_col in zip(hidden_students, hidden_cols):
        predicted_z = predict_z_score(z_students[student_idx], col_corr[target_col], k)
        students_imputed[student_idx, target_col] = (
            column_mean[target_col] + column_std[target_col] * predicted_z
        )
    return students_imputed


def standardized(imputer):
    """Standardize columns (using observed values only) around an imputer."""

    def wrapper(students_masked):
        column_mean = np.nanmean(students_masked, axis=0)
        column_std = np.nanstd(students_masked, axis=0)
        column_std[column_std == 0] = 1.0
        z_students = (students_masked - column_mean) / column_std
        return imputer(z_students) * column_std + column_mean

    return wrapper


# could add more imputers if we want to compare
IMPUTERS = {
    "column mean": column_mean_impute,
    "item-CF (PCC)": item_cf_impute,
    # user-CF is using the k nearest neighbours
    "user-CF (kNN)": standardized(
        lambda z_students: KNNImputer(
            n_neighbors=USER_CF_NEIGHBOURS, weights="distance"
        ).fit_transform(z_students)
    ),
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
        warnings.simplefilter(
            "ignore"
        )  # MICE often warns that it did not fully converge
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

        # reproducable rng
        random_generator = np.random.default_rng(seed + repeat)

        # hide the cells
        hidden_mask, students_masked = hide_random_cells(students_matrix, mask_fraction, random_generator)
        hidden_rows, hidden_cols = np.where(hidden_mask)
        hidden_count_by_column += np.bincount(hidden_cols, minlength=n_columns)

        for method_name, imputer in imputers.items():
            # impute
            students_imputed = impute_and_clip(imputer, students_masked)
            # calc errors
            scaled_errors = scaled_errors_on_hidden_cells(students_imputed, students_matrix, hidden_rows, hidden_cols, column_std)
            rmse_by_repeat[method_name].append(np.sqrt(np.mean(scaled_errors**2)))
            mae_by_repeat[method_name].append(np.mean(np.abs(scaled_errors)))
            squared_error_by_column[method_name] += np.bincount(hidden_cols, weights=scaled_errors**2, minlength=n_columns)

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
            "name mean": {name: np.mean(v) for name, v in mae_by_repeat.items()},
            "name std": {name: np.std(v) for name, v in mae_by_repeat.items()},
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
    return per_column


def compare_imputers(
    students_df,
    mask_fraction=MASK_FRAC,
    num_repeats=MASK_REPEATS,
    seed=0,
    imputers=IMPUTERS,
    reference="MICE",
):
    """Impute the missing cells and summarize the results"""
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

    print(
        f"\n=== Random masking: {n_students} students x {n_columns} cols, "
        f"{mask_fraction:.0%} of cells hidden, {num_repeats} repeats "
        f"(errors in column-std units) ==="
    )
    print(summary.round(3).to_string())

    print(
        f"\n=== Paired nRMSE difference vs {reference} "
        f"(negative = better than {reference}) ==="
    )
    print(paired_diff.round(4).to_string())

    print("\n=== Per-column nRMSE (1.0 = no better than predicting the mean) ===")
    print(per_column.sort_values("item-CF minus MICE").round(3).to_string())
    return summary, paired_diff, per_column


def main():
    """driver code"""
    students_df = load_data()
    # all numeric/binary columns, with G3 included as another column
    compare_imputers(encode_numeric(students_df))


if __name__ == "__main__":
    main()
