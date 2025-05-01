# Standard library imports
import os
import sys
from typing import List

from ipl_predictor.data_processing import get_prepared_data

# Third-party imports
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# --- Ensure src is in path for sibling imports ---
# Resolve the absolute path to the src directory
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
# Now we can reliably import from data_processing

# --- Configuration Constants ---
MODEL_DIR: str = "models"
MODEL_FILENAME: str = "ipl_winner_pipeline_v1.joblib"
MODEL_PATH: str = os.path.join(MODEL_DIR, MODEL_FILENAME)

TEST_SIZE: float = 0.2  # 20% of data for testing
RANDOM_STATE: int = 42  # Ensures reproducibility for split and model training

# --- End Configuration ---


def train_model() -> str:
    """
    Orchestrates the model training process: loads data, defines preprocessing,
    trains a model pipeline, evaluates it, and saves the pipeline.

    Returns:
        The file path where the trained pipeline was saved.
    """
    print("--- Starting Model Training Pipeline ---")

    # 1. Load Prepared Data
    print("Step 1: Loading prepared data...")
    try:
        X, y = get_prepared_data()
    except Exception as e:
        print(f"ERROR: Failed to load prepared data: {e}")
        raise

    # 2. Identify Feature Types (Crucial for Preprocessing)
    print("Step 2: Identifying feature types...")
    # If numerical features were included, they'd need separate handling.
    categorical_features: List[str] = X.columns.tolist()
    if not categorical_features:
        raise ValueError("No categorical features" " identified. Check data loading.")
    print(f"Using categorical features: {categorical_features}")

    # 3. Split Data into Training and Testing Sets
    print("Step 3: Splitting data into training and testing sets...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        print(f"Training set shape: X={X_train.shape}, y={y_train.shape}")
        print(f"Testing set shape: X={X_test.shape}, y={y_test.shape}")
    except Exception as e:
        print(f"ERROR: Failed during data splitting: {e}")
        raise

    # 4. Define Preprocessing Steps
    print("Step 4: Defining preprocessing steps...")
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            )
        ],
        remainder="passthrough",
    )
    print("Preprocessor defined (OneHotEncoder for specified categoricals).")

    # 5. Define the Model
    print("Step 5: Defining the model...")

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

    print(f"Using model: {type(model).__name__}")

    # 6. Create the Full Pipeline
    print("Step 6: Creating the full Scikit-learn pipeline...")

    pipeline: Pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", model)]
    )
    print("Pipeline created successfully.")
    print(pipeline)  # Print the pipeline structure

    # 7. Train the Pipeline
    print("Step 7: Training the pipeline...")
    try:
        pipeline.fit(X_train, y_train)
        print("Pipeline training complete.")
    except Exception as e:
        print(f"ERROR: Failed during pipeline training: {e}")
        raise

    # 8. Evaluate the Pipeline (Basic Accuracy)
    print("Step 8: Evaluating the pipeline on the test set...")
    try:
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Set Accuracy: {accuracy:.4f}")
    except Exception as e:
        print(f"ERROR: Failed during pipeline evaluation: {e}")
        raise

    # 9. Save the Trained Pipeline
    print("Step 9: Saving the trained pipeline...")
    try:
        if not os.path.exists(MODEL_DIR):
            print(f"Creating models directory: {MODEL_DIR}")
            os.makedirs(MODEL_DIR)
        joblib.dump(pipeline, MODEL_PATH)
        print(f"Pipeline successfully saved to: {MODEL_PATH}")
    except Exception as e:
        print(f"ERROR: Failed to save the pipeline: {e}")
        raise

    print("--- Model Training Pipeline Finished ---")
    return MODEL_PATH


# --- Testing Block ---
if __name__ == "__main__":
    print("\n[INFO] Running model_training.py directly...")
    try:
        saved_model_path = train_model()
        print(
            f"\n[SUCCESS] Model training script"
            f" finished. Model saved at: {saved_model_path}"
        )
    except Exception as e:
        print(f"\n[FAILURE] Model training script failed: {e}")
