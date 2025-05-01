import pandas as pd
import os
from typing import Tuple, List

# --- Configuration Constants ---
# Assuming 'data/raw' exists at the project root based on Day 0 setup
RAW_DATA_DIR: str = "data/raw"
MATCH_INFO_FILE: str = "Match_Info.csv"

# Define columns needed for initial winner prediction + the target
# Keep this minimal initially; we can add more later if needed.
RELEVANT_COLUMNS: List[str] = [
    "team1",
    "team2",
    "toss_winner",
    "toss_decision",
    "venue",
    "city",
    "winner",  # Our target variable
]

TARGET_COLUMN: str = "winner"
# --- End Configuration ---


def load_match_data(file_path: str, columns: List[str]) -> pd.DataFrame:
    """
    Loads match data from the specified CSV file, selecting specific columns.

    Args:
        file_path: Path to the CSV file.
        columns: List of column names to load.

    Returns:
        A pandas DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: For other potential loading errors.
    """
    print(f"Attempting to load data from: {file_path}")
    try:
        df = pd.read_csv(file_path, usecols=columns)
        print(f"Successfully loaded {len(df)}" f"rows and {len(df.columns)} columns.")
        # Basic validation: Check if all requested columns are present
        if not all(col in df.columns for col in columns):
            missing = [col for col in columns if col not in df.columns]
            raise ValueError(f"Missing expected columns in CSV: {missing}")
        return df
    except FileNotFoundError:
        print(f"ERROR: File not found at '{file_path}'.")
        print("Please ensure the dataset exists in the 'data/raw/' directory.")
        raise
    except ValueError as ve:
        print(f"ERROR: Column mismatch or value error during loading: {ve}")
        raise
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during data loading: {e}")
        raise


def clean_data(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Performs basic cleaning on the match DataFrame.

    - Drops rows where the target variable is missing.
    - Optional: Placeholder
    for future cleaning steps (e.g., handling missing city).

    Args:
        df: The input DataFrame.
        target_col: The name of the target variable column.

    Returns:
        The cleaned pandas DataFrame.
    """
    print("Starting data cleaning...")
    initial_rows = len(df)
    print(f"Initial row count: {initial_rows}")

    # --- Crucial Cleaning Step ---
    df.dropna(subset=[target_col], inplace=True)
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        print(
            f"Dropped {rows_dropped} rows due to missing"
            f" values in target column ('{target_col}')."
        )
    else:
        print(f"No rows dropped due to missing target ('{target_col}').")

    # --- Placeholder for Future Cleaning ---
    # More complex logic might infer city from venue.
    if "city" in df.columns and df["city"].isnull().any():
        missing_city_count = df["city"].isnull().sum()
        print(
            f"Found {missing_city_count} missing"
            f" values in 'city'. Filling with 'Unknown'."
        )
        df["city"] = df["city"].fillna("Unknown")

    # Add any other simple consistency checks or cleaning here if needed.

    print(f"Finished data cleaning. Final row count: {len(df)}")
    return df


def get_prepared_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Orchestrates the loading and cleaning of match data, returning features (X)
    and target (y).

    Returns:
        A tuple containing:
            - X: DataFrame of features.
            - y: Series of the target variable.
    """
    print("--- Starting Data Preparation Pipeline ---")
    file_path = os.path.join(RAW_DATA_DIR, MATCH_INFO_FILE)

    # Load
    df_raw = load_match_data(file_path=file_path, columns=RELEVANT_COLUMNS)

    # Clean
    df_cleaned = clean_data(df=df_raw, target_col=TARGET_COLUMN)

    # Separate features and target
    X = df_cleaned.drop(columns=[TARGET_COLUMN])
    y = df_cleaned[TARGET_COLUMN]

    print(
        f"Data preparation complete. Features shape:"
        f" {X.shape}, Target shape: {y.shape}"
    )
    print("--- End Data Preparation Pipeline ---")
    return X, y


# --- Testing Block ---
if __name__ == "__main__":
    # It's a good practice for testing the module's functions.
    print("\n[INFO] Running data_processing.py directly for testing...")
    try:
        X_data, y_data = get_prepared_data()
        print("\n[SUCCESS] Data loading and preparation test finished.")
        print("\nSample Features (X head):")
        print(X_data.head())
        print("\nSample Target (y head):")
        print(y_data.head())
        print("\nTarget Value Counts:")
        print(y_data.value_counts())  # Useful to check for class imbalance
    except Exception as e:
        print(f"\n[FAILURE] Data loading" f" and preparation test failed: {e}")
        # raise e
