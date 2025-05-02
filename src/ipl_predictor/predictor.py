import joblib
import pandas as pd
import os
import sys
from typing import Dict, List, Any, Optional
import requests  # <-- Add this import for HTTP requests
import json  # <-- Add this import for JSON handling

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# --- Configuration Constants ---
MODEL_DIR: str = "models"
MODEL_FILENAME: str = "ipl_winner_pipeline_gb_v1.joblib"
MODEL_PATH: str = os.path.join(MODEL_DIR, MODEL_FILENAME)

OLLAMA_API_URL: str = "http://localhost:11434/api/generate"
OLLAMA_MODEL: str = "smollm"

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


# --- LLM Interaction Function ---
def get_llm_explanation(prompt: str) -> Optional[str]:
    """
    Sends a prompt to the local Ollama API and returns the generated explanation.

    Args:
        prompt: The text prompt to send to the LLM.

    Returns:
        The explanation text from the LLM, or None if an error occurs.
    """
    print(f"\nSending prompt to Ollama ({OLLAMA_MODEL}):\n'''{prompt}'''")
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,  # Request a single complete response
            # Optional parameters can be added here, e.g., "temperature", "top_p"
            # "options": {
            #     "temperature": 0.7
            # }
        }
        # Set a reasonable timeout (e.g., 60 seconds) as LLM inference can take time
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        response.raise_for_status()

        # Parse the JSON response
        response_data = response.json()

        # Extract the generated text (structure depends slightly on Ollama version)
        explanation = response_data.get("response", "").strip()

        if explanation:
            print("Ollama response received successfully.")
            return explanation
        else:
            print("WARNING: Ollama response was empty.")
            # print(f"Full Ollama response: {response_data}") # Uncomment for debugging
            return None

    except requests.exceptions.RequestException as e:
        print(
            f"ERROR: Could not connect to Ollama API at {OLLAMA_API_URL}."
            f" Is Ollama running?"
        )
        print(f"Error details: {e}")
        return None
    except json.JSONDecodeError as e:
        print("ERROR: Could not decode JSON response from Ollama.")
        print(f"Response text: {response.text}")
        print(f"Error details: {e}")
        return None
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during Ollama interaction: {e}")
        return None


# --- LLM Interaction Function ---


def predict_winner(
    input_data: Dict[str, Any],
) -> Dict[str, Optional[str]]:  # <-- Changed return type hint
    """
    Makes a match winner prediction using the ML pipeline and gets an
    explanation from the LLM.

    Args:
        input_data: A dictionary containing features for a single match.
                    Keys must match the feature names the pipeline was trained on:
                    ['team1', 'team2', 'toss_winner', 'toss_decision', 'venue', 'city']

    Returns:
        A dictionary containing:
            - 'prediction': The predicted winner string.
            - 'explanation': The explanation text from the LLM, or None if failed.

    Raises:
        ValueError: If input data is missing required keys or is badly formatted.
        RuntimeError: If ML prediction fails for other reasons.
        FileNotFoundError: If the model file cannot be loaded.
    """
    pipeline = load_pipeline()  # Ensures pipeline is loaded

    # --- Input Validation and Formatting (Keep existing code) ---
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
            raise ValueError(f"Input data is missing required features: {missing}")
        input_df = input_df[expected_cols]
        print(f"\nInput DataFrame for prediction:\n{input_df}")
    except Exception as e:
        print(f"ERROR: Failed to process input data dictionary: {e}")
        raise ValueError("Invalid input data format or missing keys.") from e

    # --- Make ML Prediction (Keep existing code) ---
    ml_prediction_list: List[str]
    try:
        print("Calling pipeline.predict()...")
        ml_prediction = pipeline.predict(input_df)
        print(f"Raw prediction output: {ml_prediction}")
        ml_prediction_list = ml_prediction.tolist()
        if not ml_prediction_list:  # Basic check
            raise RuntimeError("ML model returned an empty prediction.")
    except Exception as e:
        print(f"ERROR: Failed during ML prediction step: {e}")
        raise RuntimeError("ML Prediction failed.") from e

    predicted_winner: str = ml_prediction_list[0]
    print(f"[ML Prediction] -> Winner: {predicted_winner}")

    # --- NEW: Get LLM Explanation ---
    explanation: Optional[str] = None  # Default to None
    try:
        # Construct the prompt dynamically based on input and ML prediction
        prompt = (
            f"Match Context:\n"
            f"- Team 1: {input_data['team1']}\n"
            f"- Team 2: {input_data['team2']}\n"
            f"- Venue: {input_data['venue']} ({input_data['city']})\n"
            f"- Toss: {input_data['toss_winner']} won and chose to {input_data['toss_decision']}.\n"
            f"- Predicted Winner: {predicted_winner}\n\n"
            f"Task: Provide exactly one concise (target: 1 sentence, maximum 2 sentences)"
            f" plausible reason why the predicted winner "
            f"({predicted_winner}) might be favored in this specific match context."
            f" Focus *only* on factors directly inferable "
            f"from the context provided (like potential home advantage at the venue,"
            f" or impact of the toss decision). "
            f"Do not speculate broadly on general team strength or past tournaments."
            f" Be factual based ONLY on the context. "
            f"Start the explanation directly without preamble like 'One plausible reason is...'."
        )

        # Call the LLM function
        explanation = get_llm_explanation(prompt=prompt)

    except Exception as e:
        # Catch any unexpected errors during prompt generation or the LLM call setup
        print(f"ERROR: Failed to generate prompt or call LLM function: {e}")
        explanation = (
            "Error generating explanation."  # Set explanation to indicate failure
        )

    # --- Return both results ---
    return {
        "prediction": predicted_winner,
        "explanation": explanation,  # Will be None if LLM call failed or returned empty
    }


# --- Testing Block ---
if __name__ == "__main__":
    print("\n[INFO] Running predictor.py directly for testing...")

    test_match = {
        "team1": "Kolkata Knight Riders",
        "team2": "Mumbai Indians",
        "toss_winner": "Mumbai Indians",
        "toss_decision": "field",
        "venue": "Eden Gardens",
        "city": "Kolkata",
    }
    print(f"\nTest Input Data: {test_match}")

    try:
        # Call the updated function which now returns a dict
        result = predict_winner(input_data=test_match)
        winner = result.get("prediction")
        explanation = result.get("explanation")

        print(f"\n[ML Prediction Result]: {winner}")
        print(
            f"[LLM Explanation Result]:"
            f" {explanation if explanation else 'Not Available'}"
        )

        print("\n[SUCCESS] Predictor test finished.")

    except FileNotFoundError:
        print("\n[FAILURE] Prediction test failed: Model file not found.")
        print("Please ensure you have run the model_training.py script successfully.")
    except Exception as e:
        print(f"\n[FAILURE] Prediction test failed: {e}")
