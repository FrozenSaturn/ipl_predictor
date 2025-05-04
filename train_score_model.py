# train_score_model.py

import os
import pandas as pd
import joblib
import sys
from datetime import datetime

# import numpy as np  # For MAE calculation with cross_val_score

# Scikit-learn imports
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, cross_val_score  # Use KFold for regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# from sklearn.metrics import mean_absolute_error, r2_score  # Regression metrics

# XGBoost Regressor
from xgboost import XGBRegressor

# Local application imports (Loads full feature set from features_v3_with_scores.csv)
# Ensure data_processing points to the correct CSV and defines features
try:
    from src.ipl_predictor.data_processing import (
        # get_prepared_data,
        MODEL_FEATURE_COLUMNS,
    )

    # Define the target column for score prediction
    SCORE_TARGET_COLUMN: str = "first_innings_score"
except ImportError as e:
    print(
        f"ERROR: Could not import from data_processing. Ensure it exists and defines variables: {e}"
    )
    sys.exit(1)
except Exception as e:
    print(f"ERROR during import: {e}")
    sys.exit(1)


# --- Configuration Constants ---
MODEL_DIR: str = "models"
# --- NEW FILENAME for the score prediction model ---
MODEL_FILENAME: str = (
    "ipl_score_pipeline_xgb_v1.joblib"  # Reflects score prediction + XGB
)
MODEL_PATH: str = os.path.join(MODEL_DIR, MODEL_FILENAME)

RANDOM_STATE: int = 42
N_SPLITS: int = 5  # Number of folds for cross-validation
# --- End Configuration ---


def train_score_model() -> str:
    """
    Loads full feature set, defines regression pipeline with XGBRegressor,
    EVALUATES using K-Fold Cross-Validation (MAE, R2),
    trains final model on ALL data, and saves it.
    """
    start_time = datetime.now()
    print(
        f"--- Starting Score Prediction Model Training (XGBoost + All Features + {N_SPLITS}-Fold CV Eval) ---"
    )
    print(f"Timestamp: {start_time.isoformat()}")

    # 1. Load Pre-calculated Features and Target Score
    print("Step 1: Loading pre-calculated features and score target...")
    try:
        # get_prepared_data loads features defined in MODEL_FEATURE_COLUMNS
        # We need to load the full CSV again to get the score target
        feature_file_path = os.path.join(
            os.getcwd(), "data", "processed", "features_v3_with_scores.csv"
        )
        df_full = pd.read_csv(feature_file_path)

        # Verify columns
        if SCORE_TARGET_COLUMN not in df_full.columns:
            raise ValueError(
                f"Score target column '{SCORE_TARGET_COLUMN}' not found in {feature_file_path}"
            )
        if not all(col in df_full.columns for col in MODEL_FEATURE_COLUMNS):
            missing = [
                col for col in MODEL_FEATURE_COLUMNS if col not in df_full.columns
            ]
            raise ValueError(f"Feature file missing required columns: {missing}")

        # Drop rows where target score might be NaN (should have been filled by build_features)
        initial_rows = len(df_full)
        df_full.dropna(subset=[SCORE_TARGET_COLUMN], inplace=True)
        if len(df_full) < initial_rows:
            print(
                f"Warning: Dropped {initial_rows - len(df_full)} rows with missing target score."
            )

        X = df_full[MODEL_FEATURE_COLUMNS]
        y = df_full[SCORE_TARGET_COLUMN]  # Target is the first innings score

        print(f"Loaded data shape: X={X.shape}, y={y.shape}")

    except FileNotFoundError:
        print(
            f"ERROR: Feature file not found at '{feature_file_path}'. Run build_features.py."
        )
        raise
    except Exception as e:
        print(f"ERROR loading data: {e}")
        raise

    # 2. Identify Feature Types (Same as winner model)
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
        col for col in MODEL_FEATURE_COLUMNS if col not in categorical_features
    ]
    print(
        f"Using {len(categorical_features)} categorical and {len(numerical_features)} numerical features."
    )

    # 3. Define Preprocessing Steps (OHE + Scaler - Same as winner model)
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
        remainder="drop",
    )
    print("Preprocessor defined.")

    # 4. Define the XGBoost Regressor Model
    print("Step 4: Defining the model (XGBoost Regressor)...")
    # Use objective suitable for regression
    model = XGBRegressor(
        objective="reg:squarederror",  # Standard objective for regression
        # eval_metric='rmse',         # Common evaluation metric for XGBoost reg
        n_estimators=100,  # Start with 100 trees
        random_state=RANDOM_STATE,
        n_jobs=-1,  # Use all CPU cores
        # Consider tuning learning_rate, max_depth, subsample etc. later
    )
    print(f"Using model: {type(model).__name__}")

    # 5. Create the Full Pipeline
    print("Step 5: Creating the full Scikit-learn pipeline...")
    pipeline: Pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", model),  # Changed step name to 'regressor'
        ]
    )
    print("Pipeline created successfully.")

    # --- Step 6: Evaluate Pipeline using Cross-Validation ---
    print(
        f"\nStep 6: Evaluating pipeline using {N_SPLITS}-Fold KFold Cross-Validation..."
    )
    # Use KFold for regression (stratification isn't needed/meaningful)
    cv_strategy = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # Calculate cross-validated Negative Mean Absolute Error (MAE)
    # Scikit-learn uses 'neg_mean_absolute_error' - higher is better (closer to 0)
    try:
        print("Calculating CV Negative MAE scores...")
        neg_mae_scores = cross_val_score(
            pipeline, X, y, cv=cv_strategy, scoring="neg_mean_absolute_error", n_jobs=-1
        )
        mae_scores = -neg_mae_scores  # Convert back to positive MAE (lower is better)
        print(f"CV MAE Scores per fold: {mae_scores}")
        print(f"CV MAE Mean: {mae_scores.mean():.2f} (+/- {mae_scores.std() * 2:.2f})")
    except Exception as e:
        print(f"ERROR during CV MAE calculation: {e}")

    # Calculate cross-validated R-squared (R2) score
    # R2 = 1 is perfect prediction, 0 means baseline model, negative is worse than baseline
    try:
        print("\nCalculating CV R2 scores...")
        r2_scores = cross_val_score(
            pipeline, X, y, cv=cv_strategy, scoring="r2", n_jobs=-1
        )
        print(f"CV R2 Scores per fold: {r2_scores}")
        print(f"CV R2 Mean: {r2_scores.mean():.4f} (+/- {r2_scores.std() * 2:.4f})")
    except Exception as e:
        print(f"ERROR during CV R2 calculation: {e}")

    # --- Step 7: Train Final Model on ALL Data ---
    print("\nStep 7: Training the final pipeline on ALL data...")
    try:
        pipeline.fit(X, y)  # Fit on the whole dataset X, y
        print("Final score prediction pipeline training complete.")
    except Exception as e:
        print(f"ERROR: Failed during final pipeline training: {e}")
        raise

    # --- Step 8: Save the FINAL Trained Pipeline ---
    print("\nStep 8: Saving the final trained score prediction pipeline...")
    try:
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        joblib.dump(pipeline, MODEL_PATH)  # Saves the score prediction pipeline
        print(f"Final score pipeline successfully saved to: {MODEL_PATH}")
    except Exception as e:
        print(f"ERROR: Failed to save the score pipeline: {e}")
        raise

    end_time = datetime.now()
    print("\n--- Score Prediction Model Training Finished ---")
    print(f"Timestamp: {end_time.isoformat()}")
    # print(f"Total duration: {end_time - start_time}")
    return MODEL_PATH


# --- Testing Block ---
if __name__ == "__main__":
    print("\n[INFO] Running train_score_model.py directly...")
    try:
        saved_model_path = train_score_model()
        print(
            f"\n[SUCCESS] Score model training script finished. Model saved at: {saved_model_path}"
        )
    except Exception as e:
        print(f"\n[FAILURE] Score model training script failed: {e}")
