# src/ipl_predictor/model_training.py

# Standard library imports
import os

# import sys # Removed sys.path modification

# Third-party imports
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# from sklearn.metrics import (
#     accuracy_score,
#     f1_score,
# )
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Local application imports
from .data_processing import get_prepared_data

# --- Configuration Constants ---
MODEL_DIR: str = "models"
# Filename reflects features/model type used for FINAL training
MODEL_FILENAME: str = "ipl_winner_pipeline_gb_advfeat_balanced_v1.joblib"
MODEL_PATH: str = os.path.join(MODEL_DIR, MODEL_FILENAME)
RANDOM_STATE: int = 42
N_SPLITS: int = 10  # Number of folds for cross-validation
# --- End Configuration ---


def train_model() -> str:
    """
    Orchestrates the model training process with engineered features: loads data,
    defines preprocessing (OHE for cats, Scaler for nums), defines GB model,
    EVALUATES using Stratified K-Fold Cross-Validation (Accuracy, F1 Macro),
    then trains final model on ALL data and saves it.

    Returns:
        The file path where the final trained pipeline was saved.
    """
    print(
        f"--- Starting Model Training Pipeline (GB + AdvFeats + {N_SPLITS}-Fold CV Eval) ---"
    )  # Updated title slightly

    # 1. Load Prepared Data (with engineered features)
    print("Step 1: Loading prepared data (with advanced features)...")
    X, y = get_prepared_data()

    # 2. Identify Feature Types
    print("Step 2: Identifying feature types...")
    categorical_features: list[str] = [  # Using list hint directly
        "team1",
        "team2",
        "toss_winner",
        "toss_decision",
        "venue",
        "city",
    ]
    numerical_features: list[str] = [  # Using list hint directly
        "team1_win_pct",
        "team2_win_pct",
        "team1_h2h_win_pct",
        "team1_prev_score",
        "team1_prev_wkts",
        "team2_prev_score",
        "team2_prev_wkts",
    ]
    # Check if all expected columns exist
    expected_cols = categorical_features + numerical_features
    if not all(col in X.columns for col in expected_cols):
        missing = [col for col in expected_cols if col not in X.columns]
        raise ValueError(f"DataFrame X is missing expected feature columns: {missing}")

    print(f"Using categorical features: {categorical_features}")
    print(f"Using numerical features: {numerical_features}")

    # 3. Define Preprocessing Steps
    print("Step 3: Defining preprocessing steps (OHE + Scaler)...")
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    numerical_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]  # Scale numerical features
    )
    # Use ColumnTransformer to apply different transformers to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numerical_transformer, numerical_features),
        ],
        remainder="passthrough",
    )
    print("Preprocessor defined.")

    # --- Step 4: Define the Model (Corrected) ---
    print("Step 4: Defining the model (Random Forest with balanced class weights)...")
    model = RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
    )
    print(f"Using model: {type(model).__name__}")

    # --- Step 5: Create the Full Pipeline (Corrected) ---
    print("Step 5: Creating the full Scikit-learn pipeline...")
    # Pipeline type hint optional here, can be inferred
    pipeline: Pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),  # Use the preprocessor defined in Step 3
            ("classifier", model),  # Use the model defined in Step 4
        ]
    )
    print("Pipeline created successfully.")

    # --- Step 6: Evaluate Pipeline using Cross-Validation ---
    print(
        f"\nStep 6: Evaluating pipeline using {N_SPLITS}-Fold Stratified Cross-Validation..."
    )
    cv_strategy = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )

    # Calculate cross-validated Accuracy
    try:
        print("Calculating CV Accuracy scores...")
        # Use scoring='accuracy' string identifier
        accuracy_scores = cross_val_score(
            pipeline, X, y, cv=cv_strategy, scoring="accuracy", n_jobs=-1
        )
        print(f"CV Accuracy Scores per fold: {accuracy_scores}")
        print(
            f"CV Accuracy Mean: {accuracy_scores.mean():.4f} (+/- {accuracy_scores.std() * 2:.4f})"
        )
    except Exception as e:
        print(f"ERROR during CV Accuracy calculation: {e}")

    # Calculate cross-validated Macro F1 Score
    try:
        print("\nCalculating CV F1 Macro scores...")
        # Use scoring='f1_macro' string identifier
        f1_macro_scores = cross_val_score(
            pipeline, X, y, cv=cv_strategy, scoring="f1_macro", n_jobs=-1
        )
        print(f"CV F1 Macro Scores per fold: {f1_macro_scores}")
        print(
            f"CV F1 Macro Mean: {f1_macro_scores.mean():.4f} (+/- {f1_macro_scores.std() * 2:.4f})"
        )
    except Exception as e:
        print(f"ERROR during CV F1 Macro calculation: {e}")

    # --- Step 7: Train Final Model on ALL Data ---
    print("\nStep 7: Training the final pipeline on ALL data...")
    try:
        pipeline.fit(X, y)  # Fit on the whole dataset X, y
        print("Final pipeline training complete.")
    except Exception as e:
        print(f"ERROR: Failed during final pipeline training: {e}")
        raise

    # --- Step 8: Save the FINAL Trained Pipeline ---
    print("\nStep 8: Saving the final trained pipeline...")
    try:
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        joblib.dump(pipeline, MODEL_PATH)  # Saves the final pipeline
        print(f"Final pipeline successfully saved to: {MODEL_PATH}")
    except Exception as e:
        print(f"ERROR: Failed to save the final pipeline: {e}")
        raise

    print("--- Model Training Pipeline Finished ---")
    return MODEL_PATH


# --- Testing Block ---
if __name__ == "__main__":
    print("\n[INFO] Running model_training.py directly...")
    try:
        saved_model_path = train_model()
        # Corrected f-string formatting for multi-line
        print(
            f"\n[SUCCESS] Model training script finished. "
            f"Final model saved at: {saved_model_path}"
        )
    except Exception as e:
        print(f"\n[FAILURE] Model training script failed: {e}")
