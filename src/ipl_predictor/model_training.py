# src/ipl_predictor/model_training.py

import os

# import pandas as pd  # Keep pandas import if needed by loaded data object
import joblib
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import randint
from typing import List  # Keep for type hints

# Local application imports (Now loads full feature set from features.csv)
from .data_processing import get_prepared_data, MODEL_FEATURE_COLUMNS

# --- Configuration Constants ---
MODEL_DIR: str = "models"
# --- NEW FILENAME for model trained with player form features and tuned ---
MODEL_FILENAME: str = (
    "ipl_winner_pipeline_rf_formfeat_balanced_tuned_v1.joblib"  # Reflects form features
)
MODEL_PATH: str = os.path.join(MODEL_DIR, MODEL_FILENAME)

RANDOM_STATE: int = 42
N_SPLITS: int = 5  # CV splits for RandomizedSearch
N_ITER_SEARCH: int = 50  # Number of tuning iterations

# Hyperparameter Distribution for RandomForestClassifier
PARAM_DISTRIBUTIONS = {
    "classifier__n_estimators": randint(100, 600),  # Slightly wider range
    "classifier__max_depth": [None] + list(randint(5, 40).rvs(7)),  # Wider depth range
    "classifier__min_samples_split": randint(2, 15),
    "classifier__min_samples_leaf": randint(1, 15),
    "classifier__max_features": ["sqrt", "log2", None],
    # Add other parameters like criterion if desired
    # 'classifier__criterion': ['gini', 'entropy']
}
# --- End Configuration ---


def train_model() -> str:
    """
    Loads pre-calculated full feature set, defines pipeline, tunes RF model using
    RandomizedSearchCV, saves the best model found.
    """
    start_time = datetime.now()
    print(
        "--- Starting Model Training/Tuning Pipeline (RF Balanced + All Features) ---"
    )
    print(f"Timestamp: {start_time.isoformat()}")

    # 1. Load Pre-calculated Full Feature Set
    print("Step 1: Loading pre-calculated features...")
    X, y = get_prepared_data()  # Loads 17 features + target from features.csv

    # 2. Identify Feature Types based on known columns
    print("Step 2: Identifying feature types...")
    # Ensure these lists exactly match MODEL_FEATURE_COLUMNS from data_processing
    categorical_features: List[str] = [
        "team1",
        "team2",
        "toss_winner",
        "toss_decision",
        "venue",
        "city",
    ]
    numerical_features: List[str] = [
        "team1_win_pct",
        "team2_win_pct",
        "team1_h2h_win_pct",
        "team1_prev_score",
        "team1_prev_wkts",
        "team2_prev_score",
        "team2_prev_wkts",
        "team1_avg_recent_bat_sr",
        "team1_avg_recent_bowl_econ",  # Player Form features
        "team2_avg_recent_bat_sr",
        "team2_avg_recent_bowl_econ",
    ]
    # Verify consistency
    if set(MODEL_FEATURE_COLUMNS) != set(categorical_features + numerical_features):
        raise ValueError(
            "Mismatch between features defined in data_processing and model_training!"
        )
    if not all(col in X.columns for col in MODEL_FEATURE_COLUMNS):
        missing = [col for col in MODEL_FEATURE_COLUMNS if col not in X.columns]
        raise ValueError(
            f"Loaded data X is missing required feature columns: {missing}"
        )

    print(f"Using {len(categorical_features)} categorical features.")
    print(f"Using {len(numerical_features)} numerical features.")

    # 3. Define Preprocessing Steps (OHE + Scaler - applied to correct columns)
    print("Step 3: Defining preprocessing steps...")
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
            (
                "num",
                StandardScaler(),
                numerical_features,
            ),  # Applies scaler to all 11 numerical
        ],
        remainder="drop",  # Explicitly drop any columns not specified (should be none)
    )
    print("Preprocessor defined.")

    # 4. Define the BASE Model
    print("Step 4: Defining the base model (Random Forest with balanced weights)...")
    base_model = RandomForestClassifier(
        random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
    )
    print(f"Using base model: {type(base_model).__name__}")

    # 5. Create the Base Pipeline
    print("Step 5: Creating the base Scikit-learn pipeline...")
    pipeline: Pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", base_model)]
    )
    print("Base pipeline created successfully.")

    # --- Step 6 & 7: Hyperparameter Tuning using RandomizedSearchCV ---
    print(
        f"\nStep 6/7: Fitting RandomizedSearchCV ({N_ITER_SEARCH} iterations, {N_SPLITS}-Fold)..."
    )
    cv_strategy = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER_SEARCH,
        cv=cv_strategy,
        scoring="f1_macro",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
        refit=True,  # Refit best model on all data
    )
    try:
        search.fit(X, y)  # Fit on the full feature data
        print("\nRandomizedSearchCV fitting complete.")
        print(f"Best CV F1 Macro score found: {search.best_score_:.4f}")
        print("Best parameters found:")
        # Format best parameters for readability
        best_params_formatted = {
            k.replace("classifier__", ""): v for k, v in search.best_params_.items()
        }
        print(best_params_formatted)
        best_pipeline = search.best_estimator_
    except Exception as e:
        print(f"ERROR: Failed during RandomizedSearchCV fitting: {e}")
        raise

    # --- Step 8: Save the BEST Tuned Pipeline ---
    print("\nStep 8: Saving the best tuned pipeline...")
    try:
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        joblib.dump(best_pipeline, MODEL_PATH)  # Saves to the updated filename
        print(f"Best tuned pipeline successfully saved to: {MODEL_PATH}")
    except Exception as e:
        print(f"ERROR: Failed to save the best pipeline: {e}")
        raise

    end_time = datetime.now()
    print("\n--- Model Training/Tuning Pipeline Finished ---")
    print(f"Timestamp: {end_time.isoformat()}")
    return MODEL_PATH


# --- Testing Block ---
if __name__ == "__main__":
    print("\n[INFO] Running model_training.py (Tuning with ALL Features) directly...")
    try:
        saved_model_path = train_model()
        print(
            f"\n[SUCCESS] Model tuning script finished. Best model saved at: {saved_model_path}"
        )
    except Exception as e:
        print(f"\n[FAILURE] Model tuning script failed: {e}")
