# src/ipl_predictor/model_training.py

import os

# import pandas as pd
import joblib
from datetime import datetime  # For potential logging/versioning
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier  # Keep RF
from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
)  # <-- Import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import randint  # For sampling integer hyperparameters

# Local application imports
from .data_processing import get_prepared_data

# --- Configuration Constants ---
MODEL_DIR: str = "models"
# --- NEW FILENAME for the TUNED model ---
# Reflects RF model, features, balancing, and tuning
MODEL_FILENAME: str = (
    "ipl_winner_pipeline_rf_advfeat_balanced_tuned_v1.joblib"  # Added 'tuned'
)
MODEL_PATH: str = os.path.join(MODEL_DIR, MODEL_FILENAME)

RANDOM_STATE: int = 42
N_SPLITS: int = 5  # Reduce CV splits for faster tuning (adjust if needed)
N_ITER_SEARCH: int = 50  # Number of parameter settings to sample (adjust based on time)

# --- Hyperparameter Distribution for RandomizedSearchCV ---
# Define parameter ranges to search for RandomForestClassifier
# Use 'classifier__' prefix to target parameters within the pipeline
PARAM_DISTRIBUTIONS = {
    "classifier__n_estimators": randint(100, 500),  # Sample between 100 and 500 trees
    "classifier__max_depth": [None]
    + list(randint(5, 30).rvs(5)),  # None + 5 random depths between 5-30
    "classifier__min_samples_split": randint(2, 11),  # Sample between 2 and 10
    "classifier__min_samples_leaf": randint(1, 11),  # Sample between 1 and 10
    "classifier__max_features": [
        "sqrt",
        "log2",
        None,
    ],  # Different options for feature consideration
    # 'classifier__class_weight': ['balanced', 'balanced_subsample'] # Already set to balanced
}
# --- End Configuration ---


def train_model() -> str:
    """
    Orchestrates model tuning: loads data, defines pipeline, uses RandomizedSearchCV
    to find optimal hyperparameters based on F1 Macro, trains final best model, saves it.

    Returns:
        The file path where the best trained pipeline was saved.
    """
    start_time = datetime.now()
    print("--- Starting Hyperparameter Tuning Pipeline (RF Balanced + AdvFeats) ---")
    print(f"Timestamp: {start_time.isoformat()}")

    # 1. Load Prepared Data
    print("Step 1: Loading prepared data...")
    X, y = get_prepared_data()

    # 2. Identify Feature Types
    print("Step 2: Identifying feature types...")
    categorical_features: list[str] = [
        "team1",
        "team2",
        "toss_winner",
        "toss_decision",
        "venue",
        "city",
    ]
    numerical_features: list[str] = [
        "team1_win_pct",
        "team2_win_pct",
        "team1_h2h_win_pct",
        "team1_prev_score",
        "team1_prev_wkts",
        "team2_prev_score",
        "team2_prev_wkts",
    ]
    print(f"Categorical: {categorical_features}")
    print(f"Numerical: {numerical_features}")

    # 3. Define Preprocessing Steps (OHE + Scaler)
    print("Step 3: Defining preprocessing steps...")
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
            ("num", StandardScaler(), numerical_features),
        ],
        remainder="passthrough",
    )
    print("Preprocessor defined.")

    # 4. Define the BASE Model (within the pipeline)
    print("Step 4: Defining the base model (Random Forest with balanced weights)...")
    # We define the base model here, tuning will explore variations of its params
    base_model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,  # Use cores within each RF fit if possible
    )
    print(f"Using base model: {type(base_model).__name__}")

    # 5. Create the Base Pipeline
    print("Step 5: Creating the base Scikit-learn pipeline...")
    pipeline: Pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", base_model),  # Tuner will modify params of 'classifier'
        ]
    )
    print("Base pipeline created successfully.")

    # --- Step 6: Hyperparameter Tuning using RandomizedSearchCV (NEW) ---
    print(
        f"\nStep 6: Performing Randomized Search CV ({N_ITER_SEARCH} iterations, {N_SPLITS}-Fold)..."
    )
    cv_strategy = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )

    # Setup RandomizedSearchCV
    # It will search over the PARAM_DISTRIBUTIONS for the pipeline's classifier step
    search = RandomizedSearchCV(
        estimator=pipeline,  # Our full pipeline is the estimator
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER_SEARCH,  # Number of parameter combinations to try
        cv=cv_strategy,  # Cross-validation strategy
        scoring="f1_macro",  # Optimize for F1 Macro score
        n_jobs=-1,  # Use all available CPU cores for the search
        random_state=RANDOM_STATE,
        verbose=2,  # Print progress during search
        refit=True,  # Automatically refit the best model found on the whole data
    )

    # --- Step 7: Fit the Search (Replaces CV Eval and Final Training) ---
    print("\nStep 7: Fitting RandomizedSearchCV (this may take a while)...")
    try:
        search.fit(X, y)  # Fit the search object on the entire dataset X, y
        print("RandomizedSearchCV fitting complete.")
        print(f"Best F1 Macro score found: {search.best_score_:.4f}")
        print("Best parameters found:")
        for param, value in search.best_params_.items():
            print(f"  {param}: {value}")

        # The best model (pipeline with best params, refit on all data) is stored in search.best_estimator_
        best_pipeline = search.best_estimator_

    except Exception as e:
        print(f"ERROR: Failed during RandomizedSearchCV fitting: {e}")
        raise

    # --- Step 8: Save the BEST Trained Pipeline ---
    print("\nStep 8: Saving the best tuned pipeline...")
    try:
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        # Save the best estimator found by the search
        joblib.dump(best_pipeline, MODEL_PATH)
        print(f"Best tuned pipeline successfully saved to: {MODEL_PATH}")
    except Exception as e:
        print(f"ERROR: Failed to save the best pipeline: {e}")
        raise

    end_time = datetime.now()
    print("\n--- Hyperparameter Tuning Pipeline Finished ---")
    print(f"Timestamp: {end_time.isoformat()}")
    print(f"Total duration: {end_time - start_time}")
    return MODEL_PATH


# --- Testing Block ---
if __name__ == "__main__":
    print("\n[INFO] Running model_training.py (Tuning) directly...")
    try:
        saved_model_path = train_model()
        print(
            f"\n[SUCCESS] Model tuning script finished. Best model saved at: {saved_model_path}"
        )
    except Exception as e:
        print(f"\n[FAILURE] Model tuning script failed: {e}")
