"""
Impute G3 (final grade) from student features

Scenario 1:
user = student (every feature including G1 & G2)
rating = G3

Scenario 2:
user = student (every feature excluding G1 & G2)
rating = G3
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

K = 5  # neighbours for both CF variants
N_SEEDS = 10  # number of random train/test splits to average over
TEST_SIZE = 0.2  # 80/20 training/test split


####################
# DATA PREPARATION #
####################


def load_data():
    """Load the data and split the user and item tables"""
    dataset = fetch_ucirepo(id=320)
    students, grades = dataset.data.features, dataset.data.targets

    # add G1 and G2 to the user matrix
    for g in ("G1", "G2"):
        students[g] = grades[g]

    # converting from int to float for later mean calculations
    # and converting into an array
    g3 = grades["G3"].astype(float).to_numpy()

    return students, g3


# convert all the cols to numerical values
def encode(df):
    """One-hot encode categoricals; returns a float matrix and column names."""
    enc = pd.get_dummies(df, dtype=float)
    return enc.to_numpy(), enc.columns.tolist()


def minmax_fit_transform(features_train, features_test):
    """Scale to [0, 1]."""
    col_min, col_max = features_train.min(0), features_train.max(0)
    col_range = np.where(col_max > col_min, col_max - col_min, 1.0)
    scaled_train = (features_train - col_min) / col_range
    scaled_test = (features_test - col_min) / col_range
    return scaled_train, scaled_test


###########################
# Collaborative filtering #
###########################


def user_cf(students_train, grades_train, students_test, k=K):
    """Predict G3 as the similarity-weighted mean G3 of the k most similar students."""
    # calculate the cosine similarity
    cos_sim = cosine_similarity(students_test, students_train)
    # find the most similar students
    ranked_students = np.argsort(-cos_sim, axis=1)
    # keep only the top k neighours
    neighbours_idx = ranked_students[:, :k]

    # below we're basically calculating the expected value of X
    # sum(grade * weight) / sum(weights)

    # turn the cos_sim into weights
    weights = np.take_along_axis(cos_sim, neighbours_idx, axis=1)
    # calculate the sum of the weights for each student
    weight_sum = weights.sum(axis=1)
    # all weights should be positive
    assert (weight_sum > 0).all()

    # replace the neighbours index with their grades
    neighbour_grades = grades_train[neighbours_idx]
    # multiply the grades by their weight and then sum them
    weighted_grade_sum = (weights * neighbour_grades).sum(axis=1)

    predicted_grades = weighted_grade_sum / weight_sum

    return predicted_grades


def item_cf(students_train, grades_train, students_test, k=K):
    """
    Analogue of the paper's CF-SYM-PCC: find the k features most correlated
    with G3 (Pearson, across students), then predict G3 for each test student
    from their own values on those features, weighted by correlation.
    """
    # calculate the mean and standard deviation
    feature_mean = students_train.mean(axis=0)
    feature_std = students_train.std(axis=0)

    # calculate the z-scores
    z_students_train = (students_train - feature_mean) / feature_std
    z_students_test = (students_test - feature_mean) / feature_std
    z_grades_train = (grades_train - grades_train.mean()) / grades_train.std()

    # ^ makes sure that we're comparing on the same scale

    # pearson: linear similarity
    corr_with_g3 = (z_students_train * z_grades_train[:, None]).mean(axis=0)

    top_features_idx = np.argsort(-np.abs(corr_with_g3))[:k]
    top_corrs = corr_with_g3[top_features_idx]

    # get the predicted g3, as a z-score
    predicted_z = (z_students_test[:, top_features_idx] * top_corrs).sum(axis=1) / np.abs(top_corrs).sum()
    # return the actual predicted grade
    return grades_train.mean() + grades_train.std() * predicted_z


##############
# Evaluation #
##############


def rmse(p, t):
    return float(np.sqrt(np.mean((p - t) ** 2)))


def mae(p, t):
    return float(np.mean(np.abs(p - t)))


def run_scenario(scenario_name, students, grades, drop_cols=()):
    """Predict and display results"""
    students_used = students.drop(columns=list(drop_cols))
    students_matrix, _ = encode(students_used) # non-numeric values are now floats

    # need for LOCF
    g2_grades = students["G2"].to_numpy(float)

    results = []

    for split_seed in range(N_SEEDS):

        # split the training and testing sets
        train_idx, test_idx = train_test_split(
            np.arange(len(students_used)), test_size=TEST_SIZE, random_state=split_seed
        )
        students_train, students_test = minmax_fit_transform(
            students_matrix[train_idx], students_matrix[test_idx]
        )
        grades_train, grades_test = grades[train_idx], grades[test_idx]

        # predict (we could test more methods)
        predictions_by_method = {
            "user-CF (cosine)": user_cf(students_train, grades_train, students_test),
            "item-CF (PCC)": item_cf(students_train, grades_train, students_test),
            "global mean": np.full_like(grades_test, grades_train.mean()),
        }

        # we can only d LOCF for the first scenario
        if "G2" in students.columns and "G2" not in drop_cols:
            predictions_by_method["copy G2"] = g2_grades[test_idx]


        for method_name, predictions in predictions_by_method.items():
            predictions = np.clip(predictions, 0, 20)  # forces prediction into [0,20]
            results.append({
                "method": method_name,
                "RMSE": rmse(predictions, grades_test),
                "MAE": mae(predictions, grades_test),
            })

    # display the results
    summary = pd.DataFrame(results).groupby("method").agg(["mean", "std"]).round(3)
    summary = summary.sort_values(("MAE", "mean"))
    print(f"\n=== {scenario_name} (n={len(students_used)}, {N_SEEDS} random splits) ===")
    print(summary.to_string())


def main():
    """ driver code"""
    students, g3 = load_data()
    run_scenario("All features incl. G1, G2", students, g3)
    run_scenario("Without G1 and G2", students, g3, drop_cols=("G1", "G2"))


if __name__ == "__main__":
    main()
