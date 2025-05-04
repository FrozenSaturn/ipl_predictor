# src/ipl_predictor/predictor.py

import joblib
import pandas as pd
import os
from typing import Dict, Any, Optional, Union
import requests

# import json # requests handles json sufficiently
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from asgiref.sync import sync_to_async
import asyncio

# --- Django Environment Setup ---
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ipl_django_project.settings")
try:
    import django

    django.setup()
    print("Django environment setup successfully in predictor.py.")
    from predictor_api.models import Match, Team
    from django.db.models import Q

    DJANGO_LOADED = True
except Exception as e:
    print(f"ERROR: Failed Django setup in predictor.py: {e}")
    Match = None
    Team = None
    Q = None
    DJANGO_LOADED = False
# --- End Django Setup ---


# --- Configuration Constants ---
MODEL_DIR: str = "models"
# Winner Prediction Model Files
WINNER_PIPELINE_FILENAME: str = (
    "ipl_winner_pipeline_xgb_formfeat_v1.joblib"  # Verify this!
)
WINNER_ENCODER_FILENAME: str = "target_label_encoder_v1.joblib"  # Verify this!
WINNER_PIPELINE_PATH: str = os.path.join(MODEL_DIR, WINNER_PIPELINE_FILENAME)
WINNER_ENCODER_PATH: str = os.path.join(MODEL_DIR, WINNER_ENCODER_FILENAME)
# Score Prediction Model File (NEW)
SCORE_PIPELINE_FILENAME: str = "ipl_score_pipeline_xgb_v1.joblib"  # Verify this!
SCORE_PIPELINE_PATH: str = os.path.join(MODEL_DIR, SCORE_PIPELINE_FILENAME)

OLLAMA_API_URL: str = "http://localhost:11434/api/generate"
OLLAMA_MODEL: str = "phi3:instruct"  # Changed model
# Feature Defaults
DEFAULT_WIN_PCT = 0.0
DEFAULT_H2H_WIN_PCT = 0.5
DEFAULT_SCORE = 150
DEFAULT_WKTS = 5
DEFAULT_BATTING_SR = 130.0
DEFAULT_BOWLING_ECON = 8.5
# --- End Configuration ---

# --- Global variables for lazy loading ---
_winner_pipeline: Optional[Any] = None
_winner_label_encoder: Optional[Any] = None
_score_pipeline: Optional[Any] = None  # New global for score model


def load_winner_pipeline_and_encoder() -> tuple[Any, Any]:
    """Loads the winner pipeline AND label encoder using lazy loading."""
    global _winner_pipeline, _winner_label_encoder
    if _winner_pipeline is None:
        print(
            f"Attempting to load WINNER pipeline from: {WINNER_PIPELINE_PATH}"
        )  # Uses correct constant
        if not os.path.exists(WINNER_PIPELINE_PATH):
            raise FileNotFoundError(
                f"Winner pipeline not found: '{WINNER_PIPELINE_PATH}'."
            )
        try:
            _winner_pipeline = joblib.load(WINNER_PIPELINE_PATH)
            print("Winner pipeline loaded.")
        except Exception as e:
            print(f"ERROR loading winner pipeline: {e}")
            raise
    if _winner_label_encoder is None:
        # --- THIS IS THE LINE TO FIX ---
        print(
            f"Attempting to load label encoder from: {WINNER_ENCODER_PATH}"
        )  # Use WINNER_ENCODER_PATH
        if not os.path.exists(WINNER_ENCODER_PATH):
            raise FileNotFoundError(
                f"Label encoder not found: '{WINNER_ENCODER_PATH}'."
            )
        try:
            _winner_label_encoder = joblib.load(
                WINNER_ENCODER_PATH
            )  # Use WINNER_ENCODER_PATH
            # --- END FIX ---
            print("Label encoder loaded.")
            if hasattr(_winner_label_encoder, "classes_"):
                print(f"  Encoder classes: {_winner_label_encoder.classes_}")
        except Exception as e:
            print(f"ERROR loading label encoder: {e}")
            raise
    if _winner_pipeline is None or _winner_label_encoder is None:
        raise RuntimeError("Failed to load winner pipeline or encoder.")
    return _winner_pipeline, _winner_label_encoder


def load_score_pipeline() -> Any:  # NEW function for score model
    """Loads the score prediction pipeline using lazy loading."""
    global _score_pipeline
    if _score_pipeline is None:
        print(f"Attempting to load SCORE pipeline from: {SCORE_PIPELINE_PATH}")
        if not os.path.exists(SCORE_PIPELINE_PATH):
            raise FileNotFoundError(
                f"Score pipeline not found: '{SCORE_PIPELINE_PATH}'. Run score training."
            )
        try:
            _score_pipeline = joblib.load(SCORE_PIPELINE_PATH)
            print("Score pipeline loaded successfully.")
        except Exception as e:
            print(f"ERROR loading score pipeline from {SCORE_PIPELINE_PATH}: {e}")
            raise
    return _score_pipeline


def get_llm_explanation(prompt: str) -> Optional[str]:
    """Sends prompt to Ollama API and returns explanation."""
    # ... (LLM logic unchanged) ...
    print(f"\nSending prompt to Ollama ({OLLAMA_MODEL})...")
    try:  # ... (API call logic) ...
        payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        response_data = response.json()
        explanation = response_data.get("response", "").strip()
        if explanation:
            print("Ollama response received.")
            return explanation
        else:
            print("WARNING: Ollama response empty.")
            return None
    except Exception as e:
        print(f"ERROR Ollama interaction: {e}")
        return None


# --- Synchronous DB Feature Fetch Helper (Unchanged) ---
def _fetch_all_features_sync(
    team1_name: str, team2_name: str, prediction_date: date
) -> Optional[Dict[str, Any]]:
    """Synchronous helper to fetch all 11 pre-calculated numerical features from DB."""
    # ... (DB lookup logic unchanged, returns 11 features + defaults) ...
    if not DJANGO_LOADED:
        return None
    print(
        f"  DB Lookup for features: {team1_name} vs {team2_name} before {prediction_date}"
    )
    features: Dict[str, Any] = {}
    try:  # ... (Queries Match fields for win%, prev_score/wkts, h2h%) ...
        # ... (Adds default placeholders for avg_sr, avg_econ) ...
        # ... (Applies defaults robustly) ...
        team1_obj = Team.objects.get(name=team1_name)
        team2_obj = Team.objects.get(name=team2_name)
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
        features["team1_win_pct"] = (
            (
                last_match_t1.team1_win_pct
                if last_match_t1 and last_match_t1.team1 == team1_obj
                else last_match_t1.team2_win_pct
            )
            if last_match_t1
            else None
        )
        features["team1_prev_score"] = (
            (
                last_match_t1.team1_prev_score
                if last_match_t1 and last_match_t1.team1 == team1_obj
                else last_match_t1.team2_prev_score
            )
            if last_match_t1
            else None
        )
        features["team1_prev_wkts"] = (
            (
                last_match_t1.team1_prev_wkts
                if last_match_t1 and last_match_t1.team1 == team1_obj
                else last_match_t1.team2_prev_wkts
            )
            if last_match_t1
            else None
        )
        features["team2_win_pct"] = (
            (
                last_match_t2.team1_win_pct
                if last_match_t2 and last_match_t2.team1 == team2_obj
                else last_match_t2.team2_win_pct
            )
            if last_match_t2
            else None
        )
        features["team2_prev_score"] = (
            (
                last_match_t2.team1_prev_score
                if last_match_t2 and last_match_t2.team1 == team2_obj
                else last_match_t2.team2_prev_score
            )
            if last_match_t2
            else None
        )
        features["team2_prev_wkts"] = (
            (
                last_match_t2.team1_prev_wkts
                if last_match_t2 and last_match_t2.team1 == team2_obj
                else last_match_t2.team2_prev_wkts
            )
            if last_match_t2
            else None
        )
        h2h_pct = None
        if last_h2h_match:
            h2h_pct = (
                last_h2h_match.team1_h2h_win_pct
                if last_h2h_match.team1 == team1_obj
                else (
                    (1.0 - last_h2h_match.team1_h2h_win_pct)
                    if last_h2h_match.team1_h2h_win_pct is not None
                    else None
                )
            )
        features["team1_h2h_win_pct"] = h2h_pct
        features["team1_avg_recent_bat_sr"] = DEFAULT_BATTING_SR
        features["team1_avg_recent_bowl_econ"] = DEFAULT_BOWLING_ECON
        features["team2_avg_recent_bat_sr"] = DEFAULT_BATTING_SR
        features["team2_avg_recent_bowl_econ"] = DEFAULT_BOWLING_ECON
        all_feature_keys_defaults = {
            "team1_win_pct": DEFAULT_WIN_PCT,
            "team2_win_pct": DEFAULT_WIN_PCT,
            "team1_h2h_win_pct": DEFAULT_H2H_WIN_PCT,
            "team1_prev_score": DEFAULT_SCORE,
            "team1_prev_wkts": DEFAULT_WKTS,
            "team2_prev_score": DEFAULT_SCORE,
            "team2_prev_wkts": DEFAULT_WKTS,
            "team1_avg_recent_bat_sr": DEFAULT_BATTING_SR,
            "team1_avg_recent_bowl_econ": DEFAULT_BOWLING_ECON,
            "team2_avg_recent_bat_sr": DEFAULT_BATTING_SR,
            "team2_avg_recent_bowl_econ": DEFAULT_BOWLING_ECON,
        }
        final_features = {}
        for key, default_value in all_feature_keys_defaults.items():
            final_features[key] = (
                features.get(key, default_value)
                if features.get(key) is not None
                else default_value
            )
        return final_features
    except Exception as e:
        print(f"ERROR (sync): Failed fetching features: {e}")
        return None


# --- End Synchronous Helper ---


# --- Async Feature Fetching Function ---
async def get_features_for_match(
    input_data: Dict[str, Any]
) -> Optional[Dict[str, float]]:
    """Asynchronously calls the synchronous DB lookup helper."""
    # ... (Logic unchanged, calls _fetch_all_features_sync) ...
    print("Fetching pre-calculated features from database (async)...")
    try:
        team1_name = input_data["team1"]
        team2_name = input_data["team2"]
        prediction_date = datetime.strptime(input_data["match_date"], "%Y-%m-%d").date()
        features = await sync_to_async(_fetch_all_features_sync, thread_sensitive=True)(
            team1_name, team2_name, prediction_date
        )
        if features:
            print(f"Fetched features from DB: {features}")
        else:
            print("Feature fetching failed (returned None).")
        return features
    except Exception as e:
        print(f"ERROR: Prep for feature fetching: {e}")
        return None


# --- End Async Feature Fetching Function ---


# --- Winner Predictor Function (Unchanged from last working version) ---
# --- Predictor Function ---
async def predict_winner(
    input_data: Dict[str, Any]
) -> Dict[str, Optional[Union[str, float]]]:
    """
    Predicts winner (decoding labels), confidence, gets LLM explanation async.
    Requires 'match_date'. Uses model and encoder specified by constants.
    """
    try:
        # Load pipeline and encoder (ensure load_winner_pipeline_and_encoder is defined above)
        pipeline, label_encoder = load_winner_pipeline_and_encoder()
    except Exception as e:
        print(f"ERROR: Model/Encoder loading failed: {e}")
        return {
            "prediction": None,
            "confidence": None,
            "explanation": f"Model/Encoder loading failed: {e}",
        }

    # Initialize return payload
    return_payload: Dict[str, Optional[Union[str, float]]] = {
        "prediction": None,
        "confidence": None,
        "explanation": None,
    }

    # --- Input Validation ---
    if "match_date" not in input_data:
        return_payload["explanation"] = (
            "Missing 'match_date' (YYYY-MM-DD) in input_data."
        )
        return return_payload

    # --- Feature Fetching ---
    engineered_features = await get_features_for_match(
        input_data
    )  # Ensure get_features_for_match is defined above
    if engineered_features is None:
        return_payload["explanation"] = (
            "Failed to retrieve features needed for prediction."
        )
        return return_payload

    # --- Prepare Input DataFrame ---
    try:
        combined_data = {**input_data, **engineered_features}
        # This list MUST match the features the loaded pipeline was trained on
        expected_cols = [
            "team1",
            "team2",
            "toss_winner",
            "toss_decision",
            "venue",
            "city",  # Categorical (6)
            "team1_win_pct",
            "team2_win_pct",
            "team1_h2h_win_pct",  # Historical (3)
            "team1_prev_score",
            "team1_prev_wkts",
            "team2_prev_score",
            "team2_prev_wkts",  # Prev Match (4)
            "team1_avg_recent_bat_sr",
            "team1_avg_recent_bowl_econ",  # Team Form (4) - Currently defaults/placeholders
            "team2_avg_recent_bat_sr",
            "team2_avg_recent_bowl_econ",
        ]  # Total 17 features
        input_df = pd.DataFrame([combined_data])
        if not all(col in input_df.columns for col in expected_cols):
            missing = [col for col in expected_cols if col not in input_df.columns]
            raise ValueError(f"DataFrame missing columns: {missing}.")
        input_df = input_df[expected_cols]  # Ensure order
        print(f"\nInput DataFrame for WINNER prediction:\n{input_df.iloc[0].to_dict()}")
    except Exception as e:
        print(f"ERROR: Failed to process combined input data: {e}")
        return_payload["explanation"] = f"Internal error processing input data: {e}"
        return return_payload

    # --- Make ML Prediction & Get Probabilities ---
    predicted_winner_name: Optional[str] = None
    confidence_score: Optional[float] = None
    try:
        print("Calling pipeline.predict() and pipeline.predict_proba()...")
        ml_prediction_encoded = pipeline.predict(input_df)  # Returns encoded label
        ml_probabilities = pipeline.predict_proba(input_df)
        if not ml_prediction_encoded.size:
            raise RuntimeError("ML model returned empty prediction.")

        predicted_label_encoded = ml_prediction_encoded[0]

        # Decode prediction using the loaded label_encoder
        try:
            if label_encoder is None:
                raise ValueError("Label encoder not loaded.")
            predicted_winner_name = label_encoder.inverse_transform(
                [predicted_label_encoded]
            )[0]
            return_payload["prediction"] = predicted_winner_name  # Assign decoded name
        except Exception as decode_e:
            print(
                f"ERROR: Decoding failed. Label={predicted_label_encoded}, Classes={getattr(label_encoder, 'classes_', 'N/A')}"
            )
            raise RuntimeError("Prediction label decoding failed.") from decode_e

        # Extract confidence score using label_encoder.classes_
        try:
            if label_encoder is None:
                raise ValueError("Label encoder not loaded.")
            winner_index = list(label_encoder.classes_).index(predicted_winner_name)
            conf = ml_probabilities[0, winner_index]
            # Use Decimal for stable rounding before converting back to float
            confidence_score = float(
                Decimal(str(conf)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
            )
            return_payload["confidence"] = confidence_score
            print(
                f"[ML Prediction] -> Winner: {predicted_winner_name}, Confidence: {conf:.4f}"
            )
        except Exception as inner_e:
            print(f"Warning: Could not determine confidence: {inner_e}")

    except Exception as e:
        print(f"ERROR: ML prediction failed: {e}")
        return_payload["explanation"] = f"ML prediction failed: {e}"
        return return_payload

    # --- Get LLM Explanation ---
    if return_payload["prediction"] is not None:
        try:
            # --- Prepare Context Variables for Prompt ---
            # Ensure predicted_winner_name is used here
            t1_wp = round(
                engineered_features.get("team1_win_pct", DEFAULT_WIN_PCT) * 100, 1
            )
            t2_wp = round(
                engineered_features.get("team2_win_pct", DEFAULT_WIN_PCT) * 100, 1
            )
            h2h_wp = round(
                engineered_features.get("team1_h2h_win_pct", DEFAULT_H2H_WIN_PCT) * 100,
                1,
            )
            t1_ps = int(engineered_features.get("team1_prev_score", DEFAULT_SCORE))
            t1_pw = int(engineered_features.get("team1_prev_wkts", DEFAULT_WKTS))
            t2_ps = int(engineered_features.get("team2_prev_score", DEFAULT_SCORE))
            t2_pw = int(engineered_features.get("team2_prev_wkts", DEFAULT_WKTS))
            t1_sr = round(
                engineered_features.get("team1_avg_recent_bat_sr", DEFAULT_BATTING_SR),
                1,
            )
            t1_ec = round(
                engineered_features.get(
                    "team1_avg_recent_bowl_econ", DEFAULT_BOWLING_ECON
                ),
                2,
            )
            t2_sr = round(
                engineered_features.get("team2_avg_recent_bat_sr", DEFAULT_BATTING_SR),
                1,
            )
            t2_ec = round(
                engineered_features.get(
                    "team2_avg_recent_bowl_econ", DEFAULT_BOWLING_ECON
                ),
                2,
            )
            pred_conf_str: str = (
                f"{round(confidence_score * 100, 1)}"
                if confidence_score is not None
                else "N/A"
            )

            # --- Define the Improved Prompt ---
            prompt = f"""
Act as a concise cricket analyst providing a brief insight based *only* on the provided pre-match data.

**Match Context:**
* Match: {input_data['team1']} vs {input_data['team2']}
* Venue: {input_data['venue']}, {input_data['city']}
* Toss: {input_data['toss_winner']} won, chose to {input_data['toss_decision']}.

**Prediction Details:**
* Predicted Winner: {predicted_winner_name} (Confidence: {pred_conf_str}%)

**Relevant Pre-Match Statistics:**
* Historical Win %: {input_data['team1']} ({t1_wp}%) vs {input_data['team2']} ({t2_wp}%)
* Head-to-Head Win % ({input_data['team1']} first): {h2h_wp}%
* Previous Match Score/Wickets: {input_data['team1']} ({t1_ps}/{t1_pw}) vs {input_data['team2']} ({t2_ps}/{t2_pw})
* Team Recent Form (Avg SR / Avg Econ): {input_data['team1']} ({t1_sr} / {t1_ec}) vs {input_data['team2']} ({t2_sr} / {t2_ec})

**Your Task:**
Based *strictly* on the statistics provided above, identify ONE key statistical factor (e.g., historical win %, H2H record,
recent form indicators, toss advantage) that likely favors the predicted winner ({predicted_winner_name}). Explain its potential
significance in **one single, concise sentence.**

**Constraints:**
* **One sentence only.**
* Base reason **only** on provided statistics.
* **Do NOT** use external knowledge or hallucinate facts.
* **Do NOT** mention the prediction model or confidence.
* Start sentence directly.
"""
            prompt = "\n".join(line.strip() for line in prompt.strip().splitlines())
            # --- End Prompt Definition ---

            # Call LLM Function (ensure get_llm_explanation is defined above)
            explanation = get_llm_explanation(prompt=prompt)
            return_payload["explanation"] = explanation  # Assign explanation
        except Exception as e:
            print(f"ERROR: Failed LLM call/prompt generation: {e}")
            return_payload["explanation"] = "Error generating explanation."
    else:  # Handle case where prediction failed earlier
        if return_payload["explanation"] is None:
            return_payload["explanation"] = "Prediction could not be made."

    # Return final payload containing prediction, confidence, and explanation
    return return_payload


# --- End predict_winner ---

# --- End predict_winner ---


# --- Score Predictor Function (NEW) ---
async def predict_score(input_data: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Predicts the first innings score using the score model and DB features async.
    Requires 'match_date'. Uses model specified by SCORE_PIPELINE_FILENAME.
    """
    try:
        score_pipeline = load_score_pipeline()  # Load the score prediction pipeline
    except Exception as e:
        print(f"ERROR: Failed to load score prediction pipeline: {e}")
        return {"predicted_score": None, "error": "Score model loading failed."}

    return_payload: Dict[str, Optional[float]] = {"predicted_score": None}
    error_message: Optional[str] = None  # To hold potential error messages

    if "match_date" not in input_data:
        error_message = "Missing 'match_date' (YYYY-MM-DD) in input_data."
        return {"predicted_score": None, "error": error_message}

    # Fetch the same 11 engineered features used by the winner model
    engineered_features = await get_features_for_match(input_data)
    if engineered_features is None:
        error_message = "Failed to retrieve features needed for score prediction."
        return {"predicted_score": None, "error": error_message}

    # Prepare Input DataFrame (using the same 17 features)
    try:
        combined_data = {**input_data, **engineered_features}
        # Ensure this list matches the features the score pipeline was trained on
        expected_cols = [
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
            "team1_avg_recent_bat_sr",
            "team1_avg_recent_bowl_econ",
            "team2_avg_recent_bat_sr",
            "team2_avg_recent_bowl_econ",
        ]
        input_df = pd.DataFrame([combined_data])
        if not all(col in input_df.columns for col in expected_cols):
            missing = [col for col in expected_cols if col not in input_df.columns]
            raise ValueError(
                f"DataFrame missing columns for score prediction: {missing}."
            )
        input_df = input_df[expected_cols]  # Ensure order
        print(f"\nInput DataFrame for SCORE prediction:\n{input_df.iloc[0].to_dict()}")
    except Exception as e:
        print(f"ERROR: Failed to process combined input data for score prediction: {e}")
        error_message = f"Internal error processing input data: {e}"
        return {"predicted_score": None, "error": error_message}

    # Make Score Prediction
    try:
        print("Calling score_pipeline.predict()...")
        score_prediction = score_pipeline.predict(input_df)

        if not score_prediction.size:  # Check if prediction array is empty
            raise RuntimeError("Score prediction model returned empty result.")

        predicted_score = round(
            float(score_prediction[0]), 1
        )  # Get first element, convert, round
        return_payload["predicted_score"] = predicted_score
        print(f"[Score Prediction] -> Predicted First Innings Score: {predicted_score}")

    except Exception as e:
        print(f"ERROR: Failed during score prediction step: {e}")
        error_message = f"Score prediction failed: {e}"
        return {"predicted_score": None, "error": error_message}

    # Add error message to payload if one occurred earlier but didn't cause return
    if error_message:
        return_payload["error"] = error_message  # type: ignore

    return return_payload


# --- End predict_score ---


# --- Testing Block (__main__) ---
async def main_test():
    """Async function to test both predictors."""
    print("\n[INFO] Running predictor.py directly for testing...")
    if not DJANGO_LOADED:
        print("\n[FAILURE] Cannot run test: Django env failed.")
        return

    # USE A VALID DATE FROM YOUR DATASET FOR TESTING!
    test_match = {
        "team1": "Kolkata Knight Riders",
        "team2": "Mumbai Indians",
        "toss_winner": "Mumbai Indians",
        "toss_decision": "field",
        "venue": "Eden Gardens",
        "city": "Kolkata",
        "match_date": "2012-04-27",  # <-- ADJUST DATE
    }
    print(f"\nTest Input Data: {test_match}")

    try:
        # --- Test Winner Prediction ---
        print("\n--- Testing Winner Prediction ---")
        winner_result = await predict_winner(input_data=test_match)
        winner = winner_result.get("prediction")
        confidence = winner_result.get("confidence")
        explanation = winner_result.get("explanation")
        print(f"\n[ML Prediction Result]: {winner if winner else 'N/A'}")
        print(f"[ML Confidence]: {confidence if confidence is not None else 'N/A'}")
        print(
            f"[LLM Explanation Result]: {explanation if explanation else 'Not Available'}"
        )
        if winner is None:
            print("\n[INFO] Winner Prediction returned None. Check logs.")

        # --- Test Score Prediction ---
        print("\n--- Testing Score Prediction ---")
        score_result = await predict_score(input_data=test_match)
        predicted_score = score_result.get("predicted_score")
        score_error = score_result.get("error")  # Check for errors
        print(
            f"\n[Score Prediction Result]: {predicted_score if predicted_score is not None else 'N/A'}"
        )
        if score_error:
            print(f"[Score Prediction Error]: {score_error}")

        print("\n[SUCCESS] Predictor tests finished.")

    except FileNotFoundError:
        print("\n[FAILURE] Model/Encoder file not found. Check paths/filenames.")
    except Exception as e:
        print(f"\n[FAILURE] Prediction test failed during execution: {e}")


if __name__ == "__main__":
    asyncio.run(main_test())
