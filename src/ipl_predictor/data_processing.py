# src/ipl_predictor/data_processing.py

import pandas as pd
import os
from typing import Tuple, List  # Keep List for type hints

# --- Configuration ---
# Assumes this script is run relative to the project root OR
# that the CWD is the project root when imported.
PROCESSED_DATA_DIR: str = os.path.join(os.getcwd(), "data", "processed")
# --- Point to the NEW feature file ---
FEATURE_FILE: str = (
    "features_v2_with_rolling.csv"  # <-- Use the file with rolling features
)
FEATURE_FILE_PATH: str = os.path.join(PROCESSED_DATA_DIR, FEATURE_FILE)

# --- Define ALL features present in the new CSV (used by the model) ---
# MUST match the columns saved by the latest build_features.py run
MODEL_FEATURE_COLUMNS: List[str] = [
    # Categorical (6)
    "team1",
    "team2",
    "toss_winner",
    "toss_decision",
    "venue",
    "city",
    # Historical Win % / Prev Match (7)
    "team1_win_pct",
    "team2_win_pct",
    "team1_h2h_win_pct",
    "team1_prev_score",
    "team1_prev_wkts",
    "team2_prev_score",
    "team2_prev_wkts",
    # Aggregated Player Form (4) - NEW
    "team1_avg_recent_bat_sr",
    "team1_avg_recent_bowl_econ",
    "team2_avg_recent_bat_sr",
    "team2_avg_recent_bowl_econ",
]  # Total 17 features
TARGET_COLUMN: str = "winner"
# --- End Configuration ---


# --- THIS IS THE FUNCTION THAT WAS MISSING ---
def get_prepared_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads the pre-calculated feature set (including rolling player form)
    and target variable from the processed CSV file.
    """
    print(f"--- Loading Pre-calculated Features from {FEATURE_FILE_PATH} ---")
    try:
        df = pd.read_csv(FEATURE_FILE_PATH)
        print(f"Loaded {len(df)} rows from feature file: {FEATURE_FILE}")

        # Verify necessary columns exist
        if TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"Target column '{TARGET_COLUMN}' not found in {FEATURE_FILE_PATH}."
            )
        if not all(col in df.columns for col in MODEL_FEATURE_COLUMNS):
            missing = [col for col in MODEL_FEATURE_COLUMNS if col not in df.columns]
            raise ValueError(
                f"Feature file {FEATURE_FILE_PATH} is missing required model columns: {missing}"
            )

        # Separate features (X) and target (y)
        X = df[MODEL_FEATURE_COLUMNS]
        y = df[TARGET_COLUMN]

        # Final check for unexpected NaNs
        if X.isnull().sum().sum() > 0:
            print("Warning: Found NaN values in final feature columns loaded from CSV.")
            print(X.isnull().sum())
            # Consider adding fillna logic here if necessary

        print(
            f"Data loading complete. Features shape: {X.shape}, Target shape: {y.shape}"
        )
        return X, y

    except FileNotFoundError:
        print(f"ERROR: Processed feature file not found at '{FEATURE_FILE_PATH}'.")
        print(
            "Please ensure the 'build_features.py' script generating this file ran successfully."
        )
        raise
    except Exception as e:
        print(
            f"ERROR: Failed to load or process feature file '{FEATURE_FILE_PATH}': {e}"
        )
        raise


# --- END OF FUNCTION DEFINITION ---

# --- Testing Block ---
if __name__ == "__main__":
    # This allows testing the loading process directly
    print("\n[INFO] Running data_processing.py directly for testing...")
    try:
        X_data, y_data = get_prepared_data()
        print("\n[SUCCESS] Feature loading test finished.")
        print("\nSample Features (X head):")
        print(X_data.head())
    except Exception as e:
        print(f"\n[FAILURE] Feature loading test failed: {e}")
