import joblib
import pandas as pd
import os
import sys
from typing import Dict, List, Any, Optional

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# --- Configuration Constants ---
MODEL_DIR: str = "models"
MODEL_FILENAME: str = (
    "ipl_winner_pipeline_v1.joblib"  # Must match the saved model filename
)
MODEL_PATH: str = os.path.join(MODEL_DIR, MODEL_FILENAME)

# --- Global variable for the loaded pipeline (Lazy Loading) ---
_pipeline: Optional[Any] = None
# --- End Configuration ---


def load_pipeline() -> Any:
    """
    Loads the trained pipeline from disk using lazy loading.

    Returns:
        The loaded scikit-learn pipeline object.

    Raises:
        FileNotFoundError: If the model file does not exist.
        Exception: For other potential loading errors.
    """
    global _pipeline
    if _pipeline is None:
        print(f"Attempting to load model pipeline from: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            error_msg = (
                f"ERROR: Model pipeline not found at '{MODEL_PATH}'. "
                "Please ensure the model has been trained and saved correctly."
            )
            print(error_msg)
            raise FileNotFoundError(error_msg)
        try:
            _pipeline = joblib.load(MODEL_PATH)
            print("Model pipeline loaded successfully into memory.")
        except Exception as e:
            print(f"ERROR: Failed to load model" f" pipeline from {MODEL_PATH}: {e}")
            raise
    # else: # Optional: uncomment for debugging multiple calls
    # print("Model pipeline already in memory.")
    return _pipeline


def predict_winner(input_data: Dict[str, Any]) -> List[str]:
    """
    Makes a match winner prediction using the loaded pipeline.

    Args:
        input_data: A dictionary containing features for a single match.
                    Keys must match the feature
                    names the pipeline was trained on:
                    ['team1', 'team2', 'toss_winner',
                    'toss_decision', 'venue', 'city']

    Returns:
        A list containing the predicted winner string(s)
        (typically one prediction).

    Raises:
        ValueError: If input data is missing
        required keys or is badly formatted.
        RuntimeError: If prediction fails for other reasons.
    """
    pipeline = load_pipeline()  # Ensures pipeline is loaded

    # --- Input Validation and Formatting ---
    # Convert input dict to DataFrame - This is CRUCIAL!
    try:
        input_df = pd.DataFrame([input_data])

        expected_cols = [
            "team1",
            "team2",
            "toss_winner",
            "toss_decision",
            "venue",
            "city",
        ]
        if not all(col in input_df.columns for col in expected_cols):
            missing = [col for col in expected_cols if col not in input_df.columns]
            raise ValueError(f"Input data is missing" f" required features: {missing}")
        input_df = input_df[expected_cols]

        print(f"\nInput DataFrame for prediction:\n{input_df}")

    except Exception as e:
        print(f"ERROR: Failed to process input data dictionary: {e}")
        raise ValueError("Invalid input data format or missing keys.") from e

    # --- Make Prediction ---
    try:
        print("Calling pipeline.predict()...")
        prediction = pipeline.predict(input_df)
        # predict() usually returns a NumPy array
        print(f"Raw prediction output: {prediction}")
        return (
            prediction.tolist()
        )  # Convert to standard Python list for easier handling
    except Exception as e:
        print(f"ERROR: Failed during prediction step: {e}")
        # Consider logging input_df here for debugging difficult cases
        # print(f"DataFrame passed to predict:\n{input_df.to_string()}")
        raise RuntimeError("Prediction failed.") from e


# --- Testing Block ---
if __name__ == "__main__":
    print("\n[INFO] Running predictor.py directly for testing...")

    # Define example input data - Use realistic values based on your dataset
    # Ensure keys match EXACTLY the expected feature names
    test_match = {
        "team1": "Kolkata Knight Riders",  # Example
        "team2": "Mumbai Indians",  # Example
        "toss_winner": "Mumbai Indians",  # Example
        "toss_decision": "field",  # Example: must be 'field' or 'bat'
        "venue": "Eden Gardens",  # Example
        "city": "Kolkata",
    }
    print(f"\nTest Input Data: {test_match}")

    try:
        predicted_winner = predict_winner(input_data=test_match)
        print("\n[SUCCESS] Prediction test finished.")
        print(f"Predicted Winner: {predicted_winner[0]}")

    except FileNotFoundError:
        print("\n[FAILURE] Prediction test failed: Model file not found.")
        print(
            "Please ensure you have run the " "model_training.py script successfully."
        )
    except Exception as e:
        print(f"\n[FAILURE] Prediction test failed: {e}")
        # raise e # Re-raise if needed
