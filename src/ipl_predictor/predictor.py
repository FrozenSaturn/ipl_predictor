# src/ipl_predictor/predictor.py

import joblib
import pandas as pd
import os
import sys
from typing import Dict, Any, Optional, Union  # <-- Added Union
import requests

# import json
from datetime import date, datetime
from asgiref.sync import sync_to_async
import asyncio

# --- Django Environment Setup ---
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ipl_django_project.settings")
try:
    import django

    django.setup()
    print("Django environment setup successfully in predictor.py.")
    from predictor_api.models import Match, Team
    from django.db.models import Q

    DJANGO_LOADED = True
except Exception as e:
    print(f"ERROR: Failed to setup Django environment in predictor.py: {e}")
    # Match = None
    # Team = None
    Q = None
    DJANGO_LOADED = False
# --- End Django Setup ---


# --- Configuration Constants ---
MODEL_DIR: str = "models"
# !!! IMPORTANT: Verify/Update this filename to your best model (e.g., RF balanced) !!!
MODEL_FILENAME: str = (
    "ipl_winner_pipeline_rf_advfeat_balanced_tuned_v1.joblib"  # <-- VERIFY/UPDATE THIS!
)
MODEL_PATH: str = os.path.join(MODEL_DIR, MODEL_FILENAME)

OLLAMA_API_URL: str = "http://localhost:11434/api/generate"
OLLAMA_MODEL: str = "smollm"  # Your chosen Ollama model

DEFAULT_WIN_PCT = 0.0
DEFAULT_H2H_WIN_PCT = 0.5
DEFAULT_SCORE = 150
DEFAULT_WKTS = 5
# --- End Configuration ---


# --- Global pipeline variable and load_pipeline ---
_pipeline: Optional[Any] = None


def load_pipeline() -> Any:
    """Loads the trained pipeline specified by MODEL_PATH."""
    global _pipeline
    if _pipeline is None:
        print(f"Attempting to load model pipeline from: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            error_msg = (
                f"ERROR: Model pipeline not found at '{MODEL_PATH}'. "
                "Ensure the correct model trained with features exists."
            )
            print(error_msg)
            raise FileNotFoundError(error_msg)
        try:
            _pipeline = joblib.load(MODEL_PATH)
            print("Model pipeline loaded successfully into memory.")
        except Exception as e:
            print(f"ERROR: Failed to load model pipeline from {MODEL_PATH}: {e}")
            raise
    return _pipeline


# --- End load_pipeline ---


# --- LLM Interaction Function ---
def get_llm_explanation(prompt: str) -> Optional[str]:
    """Sends prompt to Ollama API and returns explanation."""
    print(f"\nSending prompt to Ollama ({OLLAMA_MODEL}):\n'''{prompt}'''")
    try:
        payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        response_data = response.json()
        explanation = response_data.get("response", "").strip()
        if explanation:
            print("Ollama response received successfully.")
            return explanation
        else:
            print("WARNING: Ollama response was empty.")
            return None
    except Exception as e:
        print(f"ERROR during Ollama interaction: {e}")
        return None


# --- End LLM Interaction Function ---


# --- Synchronous Helper Function for DB Logic ---
def _fetch_features_sync(
    team1_name: str, team2_name: str, prediction_date: date
) -> Optional[Dict[str, Any]]:
    """Synchronous helper to perform all DB lookups for features."""
    if not DJANGO_LOADED:
        return None

    features: Dict[str, Any] = {}
    try:
        team1_obj = Team.objects.get(name=team1_name)
        team2_obj = Team.objects.get(name=team2_name)

        # Find last matches
        last_match_t1 = (
            Match.objects.filter(
                Q(team1=team1_obj) | Q(team2=team1_obj), date__lt=prediction_date
            )
            .order_by("-date", "-match_number")
            .first()
        )
        last_match_t2 = (
            Match.objects.filter(
                Q(team1=team2_obj) | Q(team2=team2_obj), date__lt=prediction_date
            )
            .order_by("-date", "-match_number")
            .first()
        )
        last_h2h_match = (
            Match.objects.filter(
                Q(team1=team1_obj, team2=team2_obj)
                | Q(team1=team2_obj, team2=team1_obj),
                date__lt=prediction_date,
            )
            .order_by("-date", "-match_number")
            .first()
        )

        # Extract features (Team 1)
        if last_match_t1:
            if last_match_t1.team1 == team1_obj:
                features["team1_win_pct"] = last_match_t1.team1_win_pct
                features["team1_prev_score"] = last_match_t1.team1_prev_score
                features["team1_prev_wkts"] = last_match_t1.team1_prev_wkts
            else:
                features["team1_win_pct"] = last_match_t1.team2_win_pct
                features["team1_prev_score"] = last_match_t1.team2_prev_score
                features["team1_prev_wkts"] = last_match_t1.team2_prev_wkts
        # Extract features (Team 2)
        if last_match_t2:
            if last_match_t2.team1 == team2_obj:
                features["team2_win_pct"] = last_match_t2.team1_win_pct
                features["team2_prev_score"] = last_match_t2.team1_prev_score
                features["team2_prev_wkts"] = last_match_t2.team1_prev_wkts
            else:
                features["team2_win_pct"] = last_match_t2.team2_win_pct
                features["team2_prev_score"] = last_match_t2.team2_prev_score
                features["team2_prev_wkts"] = last_match_t2.team2_prev_wkts
        # Extract features (H2H)
        if last_h2h_match:
            if last_h2h_match.team1 == team1_obj:
                features["team1_h2h_win_pct"] = last_h2h_match.team1_h2h_win_pct
            else:
                features["team1_h2h_win_pct"] = (
                    (1.0 - last_h2h_match.team1_h2h_win_pct)
                    if last_h2h_match.team1_h2h_win_pct is not None
                    else None
                )

        # Apply defaults
        all_feature_keys_defaults = {
            "team1_win_pct": DEFAULT_WIN_PCT,
            "team2_win_pct": DEFAULT_WIN_PCT,
            "team1_h2h_win_pct": DEFAULT_H2H_WIN_PCT,
            "team1_prev_score": DEFAULT_SCORE,
            "team1_prev_wkts": DEFAULT_WKTS,
            "team2_prev_score": DEFAULT_SCORE,
            "team2_prev_wkts": DEFAULT_WKTS,
        }
        final_features = {}
        for key, default_value in all_feature_keys_defaults.items():
            fetched_value = features.get(key)
            final_features[key] = (
                fetched_value if fetched_value is not None else default_value
            )
        return final_features

    except Team.DoesNotExist:
        print(f"ERROR (sync): Team not found: {team1_name} or {team2_name}")
        return None
    except Exception as e:
        print(f"ERROR (sync): Failed fetching features: {e}")
        return None


# --- End Synchronous Helper ---


# --- Async Feature Fetching Function ---
async def get_features_for_match(
    input_data: Dict[str, Any]
) -> Optional[Dict[str, float]]:  # Hint reflects float return
    """Asynchronously calls the synchronous DB lookup function."""
    print("Fetching historical features from database (via sync_to_async)...")
    try:
        team1_name = input_data["team1"]
        team2_name = input_data["team2"]
        prediction_date = datetime.strptime(input_data["match_date"], "%Y-%m-%d").date()
        features = await sync_to_async(_fetch_features_sync, thread_sensitive=True)(
            team1_name, team2_name, prediction_date
        )
        if features:
            print(f"Fetched features from DB: {features}")
        else:
            print("Feature fetching failed (returned None).")
        return features
    except ValueError:
        print(f"ERROR: Invalid date format: {input_data.get('match_date')}")
        return None
    except Exception as e:
        print(f"ERROR: Prep for feature fetching: {e}")
        return None


# --- End Async Feature Fetching Function ---


# --- Predictor Function ---
# Updated return type hint
async def predict_winner(
    input_data: Dict[str, Any]
) -> Dict[str, Optional[Union[str, float]]]:
    """
    Predicts winner, confidence score, and gets LLM explanation async.
    Requires 'match_date' (YYYY-MM-DD). Uses model specified by MODEL_FILENAME.
    """
    pipeline = load_pipeline()
    return_payload: Dict[str, Optional[Union[str, float]]] = {
        "prediction": None,
        "confidence": None,
        "explanation": None,
    }  # Init payload

    if "match_date" not in input_data:
        return_payload["explanation"] = (
            "Missing 'match_date' (YYYY-MM-DD) in input_data."
        )
        return return_payload

    # Fetch Engineered Features
    engineered_features = await get_features_for_match(input_data)
    if engineered_features is None:
        return_payload["explanation"] = (
            "Failed to retrieve features needed for prediction."
        )
        return return_payload

    # Prepare Input DataFrame
    try:
        combined_data = {**input_data, **engineered_features}
        expected_cols = [  # Full list...
            "team1",
            "team2",
            "toss_winner",
            "toss_decision",
            "venue",
            "city",
            "team1_win_pct",
            "team2_win_pct",
            "team1_h2h_win_pct",
            "team1_prev_score",
            "team1_prev_wkts",
            "team2_prev_score",
            "team2_prev_wkts",
        ]
        input_df = pd.DataFrame([combined_data])
        if not all(col in input_df.columns for col in expected_cols):
            missing = [col for col in expected_cols if col not in input_df.columns]
            raise ValueError(f"Internal error: DataFrame missing columns: {missing}")
        input_df = input_df[expected_cols]
        print(
            f"\nInput DataFrame for prediction (with features):\n{input_df.iloc[0].to_dict()}"
        )
    except Exception as e:
        print(f"ERROR: Failed to process combined input data: {e}")
        return_payload["explanation"] = f"Internal error processing input data: {e}"
        return return_payload

    # Make ML Prediction & Get Probabilities
    try:
        print("Calling pipeline.predict() and pipeline.predict_proba()...")
        ml_prediction = pipeline.predict(input_df)
        ml_probabilities = pipeline.predict_proba(input_df)

        if not ml_prediction.any():
            raise RuntimeError("ML model returned empty prediction.")

        predicted_winner = ml_prediction[0]
        return_payload["prediction"] = predicted_winner  # Set prediction in payload

        # Extract confidence score for the predicted class
        try:
            # Ensure pipeline has 'classes_' attribute (fitted)
            if hasattr(pipeline, "classes_"):
                winner_index = list(pipeline.classes_).index(predicted_winner)
                confidence = ml_probabilities[0, winner_index]
                return_payload["confidence"] = round(
                    float(confidence), 4
                )  # Assign confidence
                print(
                    f"[ML Prediction] -> Winner: {predicted_winner}, Confidence: {confidence:.4f}"
                )
            else:
                print(
                    "Warning: Cannot determine confidence, pipeline has no 'classes_' attribute."
                )
                print(f"[ML Prediction] -> Winner: {predicted_winner}, Confidence: N/A")

        except ValueError:  # Winner not in classes_ list
            print(
                f"Warning: Predicted winner '{predicted_winner}' not found in classes: {getattr(pipeline, 'classes_', 'N/A')}"
            )
            print(f"[ML Prediction] -> Winner: {predicted_winner}, Confidence: N/A")
        except IndexError:  # Problem accessing probabilities array
            print(
                f"Warning: Could not access probabilities correctly. Shape: {ml_probabilities.shape}"
            )
            print(f"[ML Prediction] -> Winner: {predicted_winner}, Confidence: N/A")

    except Exception as e:
        print(f"ERROR: Failed during ML prediction/probability step: {e}")
        return_payload["explanation"] = f"ML prediction failed: {e}"
        # Reset prediction/confidence if prediction failed
        return_payload["prediction"] = None
        return_payload["confidence"] = None
        return return_payload

    # Get LLM Explanation (Only if prediction succeeded)
    if return_payload["prediction"] is not None:
        try:
            prompt = (  # Refined prompt...
                f"Match Context:\n- Team 1: {input_data['team1']}..."  # etc.
            )
            explanation = get_llm_explanation(prompt=prompt)
            return_payload["explanation"] = explanation  # Assign explanation
        except Exception as e:
            print(f"ERROR: Failed to generate prompt or call LLM function: {e}")
            return_payload["explanation"] = "Error generating explanation."
    else:
        # If prediction is None due to earlier error, explanation should reflect that
        if return_payload["explanation"] is None:
            return_payload["explanation"] = "Prediction could not be made."

    # Return final payload
    return return_payload


# --- End predict_winner ---


# --- Testing Block (__main__) ---
async def main_test():
    """Async function to test the predictor."""
    print("\n[INFO] Running predictor.py directly for testing...")
    if not DJANGO_LOADED:
        print("\n[FAILURE] Cannot run test: Django environment failed to load.")
        return

    # USE A VALID DATE FROM YOUR DATASET FOR TESTING!
    test_match = {
        "team1": "Kolkata Knight Riders",
        "team2": "Mumbai Indians",
        "toss_winner": "Mumbai Indians",
        "toss_decision": "field",
        "venue": "Eden Gardens",
        "city": "Kolkata",
        "match_date": "2024-05-10",  # <-- ADJUST DATE TO A VALID ONE
    }
    print(f"\nTest Input Data: {test_match}")

    try:
        result = await predict_winner(input_data=test_match)  # Use await
        winner = result.get("prediction")
        confidence = result.get("confidence")  # Get confidence
        explanation = result.get("explanation")

        print(f"\n[ML Prediction Result]: {winner if winner else 'N/A'}")
        print(
            f"[ML Confidence]: {confidence if confidence is not None else 'N/A'}"
        )  # Print confidence
        print(
            f"[LLM Explanation Result]: {explanation if explanation else 'Not Available'}"
        )

        if winner is None:
            print("\n[INFO] Prediction returned None or failed. Check logs for errors.")

        print("\n[SUCCESS] Predictor test finished.")

    except FileNotFoundError:
        print("\n[FAILURE] Prediction test failed: Model file not found.")
        print(f"Ensure model '{MODEL_FILENAME}' exists.")
    except Exception as e:
        print(f"\n[FAILURE] Prediction test failed during execution: {e}")
        # import traceback
        # traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main_test())  # Use asyncio.run
