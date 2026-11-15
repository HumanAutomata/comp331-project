
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# ==========================================
# STEP 1: LOAD ASSISTMENTS DATASET
# ==========================================


project_root = Path(__file__).resolve().parents[2]

data = pd.read_csv(
    project_root / "data" / "assistments_matrix.csv",
    index_col=0
)

matrix = data.to_numpy(dtype=float)

MASK_FRACTION = 0.03
SEEDS = [42, 123, 202, 456, 789]
K = 10

results = []

print("Dataset shape:", matrix.shape, flush=True)
print("Observed values:", np.isfinite(matrix).sum(), flush=True)


# ==========================================
# STEP 2: EVALUATION FUNCTIONS
# ==========================================

def calculate_scores(actual, predicted):
    predicted = np.clip(np.asarray(predicted), 0, 1)

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    return float(mae), float(rmse)


# ==========================================
# STEP 3: COLLABORATIVE FILTERING
# ==========================================

def cf_predict(train, positions, means, mode):

    frame = pd.DataFrame(train)

    if mode == "User CF":
        similarities = frame.T.corr(
            min_periods=2
        ).to_numpy()

    else:
        similarities = frame.corr(
            min_periods=2
        ).to_numpy()

    # FIX: Make NumPy array writable.
    similarities = similarities.copy()

    # Exclude self-similarity.
    np.fill_diagonal(similarities, np.nan)

    predictions = []
    used_neighbors = 0

    for student, skill in positions:

        if mode == "User CF":
            values = train[:, skill]
            weights = similarities[student]

        else:
            values = train[student, :]
            weights = similarities[skill]

        valid = (
            np.isfinite(values)
            & np.isfinite(weights)
            & (weights > 0)
        )

        indices = np.flatnonzero(valid)

        if len(indices) == 0:
            predictions.append(means[skill])
            continue

        # Select the top K neighbors.
        indices = indices[
            np.argsort(weights[indices])[-K:]
        ]

        prediction = np.average(
            values[indices],
            weights=weights[indices]
        )

        predictions.append(prediction)
        used_neighbors += 1

    return np.array(predictions), used_neighbors


# ==========================================
# STEP 4: RUN RANDOM MASKING EXPERIMENTS
# ==========================================

for seed in SEEDS:

    print(f"\nRunning seed {seed}...", flush=True)

    rng = np.random.default_rng(seed)

    observed = np.argwhere(np.isfinite(matrix))

    # Protect at least one known value
    # for every student and every skill.

    protected = set()

    for student in range(matrix.shape[0]):

        skills = np.flatnonzero(
            np.isfinite(matrix[student])
        )

        if len(skills) > 0:
            protected.add(
                (student, int(rng.choice(skills)))
            )

    for skill in range(matrix.shape[1]):

        students = np.flatnonzero(
            np.isfinite(matrix[:, skill])
        )

        if len(students) > 0:
            protected.add(
                (int(rng.choice(students)), skill)
            )

    candidates = np.array(
        [
            (int(s), int(k))
            for s, k in observed
            if (int(s), int(k)) not in protected
        ],
        dtype=int
    )

    count = round(
        len(observed) * MASK_FRACTION
    )

    selected = rng.choice(
        len(candidates),
        size=count,
        replace=False
    )

    positions = candidates[selected]

    # Store actual values before hiding.
    actual = matrix[
        positions[:, 0],
        positions[:, 1]
    ]

    # Create training matrix.
    train = matrix.copy()

    train[
        positions[:, 0],
        positions[:, 1]
    ] = np.nan


    # ======================================
    # METHOD 1: SKILL MEAN
    # ======================================

    skill_means = np.nanmean(
        train,
        axis=0
    )

    predictions = {
        "Skill Mean": skill_means[
            positions[:, 1]
        ]
    }


    # ======================================
    # METHOD 2: USER-BASED CF
    # ======================================

    user_pred, user_used = cf_predict(
        train,
        positions,
        skill_means,
        "User CF"
    )

    predictions["User CF"] = user_pred


    # ======================================
    # METHOD 3: ITEM-BASED CF
    # ======================================

    item_pred, item_used = cf_predict(
        train,
        positions,
        skill_means,
        "Item CF"
    )

    predictions["Item CF"] = item_pred


    # ======================================
    # METHOD 4: MICE IMPUTATION
    # ======================================

    mice = IterativeImputer(
        max_iter=5,
        n_nearest_features=15,
        initial_strategy="mean",
        keep_empty_features=True,
        random_state=seed
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        completed = mice.fit_transform(
            train
        )

    predictions["MICE"] = completed[
        positions[:, 0],
        positions[:, 1]
    ]


    # ======================================
    # STEP 5: CALCULATE MAE AND RMSE
    # ======================================

    for method, predicted in predictions.items():

        mae, rmse = calculate_scores(
            actual,
            predicted
        )

        results.append({
            "Masking": int(MASK_FRACTION * 100),
            "Seed": seed,
            "Method": method,
            "MAE": mae,
            "RMSE": rmse,
            "Hidden": count
        })

        print(
            f"{method}: "
            f"MAE={mae:.4f}, "
            f"RMSE={rmse:.4f}",
            flush=True
        )

    print(
        f"CF neighbor coverage: "
        f"User {user_used}/{count}, "
        f"Item {item_used}/{count}",
        flush=True
    )


# ==========================================
# STEP 6: SAVE ALL EXPERIMENT RESULTS
# ==========================================


results_df = pd.DataFrame(results)

results_df.to_csv(
    project_root / "results" / "assistments_final_all_results.csv",
    index=False
)


# ==========================================
# STEP 7: CALCULATE AVERAGE RESULTS
# ==========================================

summary = (
    results_df
    .groupby("Method")[["MAE", "RMSE"]]
    .agg(["mean", "std"])
    .round(4)
)

summary.columns = [
    "MAE_Mean",
    "MAE_Std",
    "RMSE_Mean",
    "RMSE_Std"
]

summary = summary.sort_values(
    "RMSE_Mean"
)


summary.to_csv(
    project_root / "results" / "assistments_final_summary.csv"
)


# ==========================================
# STEP 8: DISPLAY FINAL RESULTS
# ==========================================

print("\n===================================")
print("FINAL AVERAGE RESULTS")
print("===================================")

print(summary.to_string())

print("\nExperiments completed successfully!")

print("\nFiles saved:")
print("assistments_final_all_results.csv")
print("assistments_final_summary.csv")
