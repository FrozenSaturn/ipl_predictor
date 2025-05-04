# src/ipl_predictor/predictor.py

import joblib
import pandas as pd
import os
import sys
from typing import Dict, Any, Optional, Union
import requests
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
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
    print(f"ERROR: Failed Django setup in predictor.py: {e}")
    # Match = None
    # Team = None
    # Q = None
    DJANGO_LOADED = False
# --- End Django Setup ---


# --- Configuration Constants ---
MODEL_DIR: str = "models"
# !!! Verify these filenames match your saved artifacts !!!
PIPELINE_FILENAME: str = (
    "ipl_winner_pipeline_xgb_formfeat_v1.joblib"  # Model trained with all features
)
ENCODER_FILENAME: str = "target_label_encoder_v1.joblib"  # Saved LabelEncoder
PIPELINE_PATH: str = os.path.join(MODEL_DIR, PIPELINE_FILENAME)
ENCODER_PATH: str = os.path.join(MODEL_DIR, ENCODER_FILENAME)

OLLAMA_API_URL: str = "http://localhost:11434/api/generate"
OLLAMA_MODEL: str = "phi3:instruct"
DEFAULT_WIN_PCT = 0.0
DEFAULT_H2H_WIN_PCT = 0.5
DEFAULT_SCORE = 150
DEFAULT_WKTS = 5
DEFAULT_BATTING_SR = 130.0
DEFAULT_BOWLING_ECON = 8.5
# --- End Configuration ---

_pipeline: Optional[Any] = None
_label_encoder: Optional[Any] = None


def load_pipeline_and_encoder() -> tuple[Any, Any]:
    """Loads the trained pipeline AND label encoder using lazy loading."""
    global _pipeline, _label_encoder
    if _pipeline is None:
        print(f"Attempting to load model pipeline from: {PIPELINE_PATH}")
        if not os.path.exists(PIPELINE_PATH):
            raise FileNotFoundError(f"Model pipeline not found: '{PIPELINE_PATH}'.")
        try:
            _pipeline = joblib.load(PIPELINE_PATH)
            print("Model pipeline loaded.")
        except Exception as e:
            print(f"ERROR loading pipeline: {e}")
            raise
    if _label_encoder is None:
        print(f"Attempting to load label encoder from: {ENCODER_PATH}")
        if not os.path.exists(ENCODER_PATH):
            raise FileNotFoundError(f"Label encoder not found: '{ENCODER_PATH}'.")
        try:
            _label_encoder = joblib.load(ENCODER_PATH)
            print("Label encoder loaded.")
            if hasattr(_label_encoder, "classes_"):
                print(f"  Encoder classes: {_label_encoder.classes_}")
        except Exception as e:
            print(f"ERROR loading label encoder: {e}")
            raise
    if _pipeline is None or _label_encoder is None:
        raise RuntimeError("Failed to load pipeline or encoder.")
    return _pipeline, _label_encoder


def get_llm_explanation(prompt: str) -> Optional[str]:
    """Sends prompt to the configured Ollama API and returns the explanation text."""
    # ... (LLM logic unchanged) ...
    print(f"\nSending prompt to Ollama ({OLLAMA_MODEL})...")
    try:
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
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Ollama connection: {e}")
        return None
    except Exception as e:
        print(f"ERROR Ollama interaction: {e}")
        return None


# --- Synchronous DB Feature Fetch Helper ---
def _fetch_all_features_sync(
    team1_name: str, team2_name: str, prediction_date: date
) -> Optional[Dict[str, Any]]:
    """Synchronous helper to fetch pre-calculated features from DB Match fields and add form placeholders."""
    # ... (DB lookup logic unchanged) ...
    if not DJANGO_LOADED:
        return None
    print(
        f"  DB Lookup for features: {team1_name} vs {team2_name} before {prediction_date}"
    )
    features: Dict[str, Any] = {}
    try:
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
    # ... (Logic unchanged) ...
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


# --- Predictor Function ---
async def predict_winner(
    input_data: Dict[str, Any]
) -> Dict[str, Optional[Union[str, float]]]:
    """
    Predicts winner (decoding labels), confidence, gets LLM explanation async.
    Requires 'match_date'. Uses model and encoder specified by constants.
    """
    try:
        # Load pipeline and encoder (ensure load_pipeline_and_encoder is defined above)
        pipeline, label_encoder = load_pipeline_and_encoder()
    except Exception as e:
        print(f"FATAL: Failed to load pipeline or encoder: {e}")
        return {
            "prediction": None,
            "confidence": None,
            "explanation": "Model loading failed.",
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
            "team1_avg_recent_bowl_econ",  # Team Form (4) - Currently defaults
            "team2_avg_recent_bat_sr",
            "team2_avg_recent_bowl_econ",
        ]  # Total 17 features
        input_df = pd.DataFrame([combined_data])
        if not all(col in input_df.columns for col in expected_cols):
            missing = [col for col in expected_cols if col not in input_df.columns]
            raise ValueError(f"DataFrame missing columns: {missing}.")
        input_df = input_df[expected_cols]  # Ensure order
        print(
            f"\nInput DataFrame for prediction (ALL features):\n{input_df.iloc[0].to_dict()}"
        )
    except Exception as e:
        print(f"ERROR: Failed to process combined input data: {e}")
        return_payload["explanation"] = f"Internal error processing input data: {e}"
        return return_payload

    # --- Make ML Prediction & Get Probabilities ---
    predicted_winner_name: Optional[str] = None
    confidence_score: Optional[float] = None
    try:
        print("Calling pipeline.predict() and pipeline.predict_proba()...")
        ml_prediction_encoded = pipeline.predict(input_df)
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
            # Prepare context for prompt
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

            # Define the Improved Prompt (using predicted_winner_name)
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
Based *strictly* on the statistics provided above, identify ONE key statistical factor
(e.g., historical win %, H2H record, recent form indicators, toss advantage) that likely favors the predicted winner ({predicted_winner_name}).
Explain its potential significance in **one single, concise sentence.**

**Constraints:**
* **One sentence only.**
* Base reason **only** on provided statistics.
* **Do NOT** use external knowledge or hallucinate facts.
* **Do NOT** mention the prediction model or confidence.
* Start sentence directly.
"""
            prompt = "\n".join(line.strip() for line in prompt.strip().splitlines())
            # Call LLM Function (ensure get_llm_explanation is defined above)
            explanation = get_llm_explanation(prompt=prompt)
            return_payload["explanation"] = explanation
        except Exception as e:
            print(f"ERROR: Failed LLM call/prompt generation: {e}")
            return_payload["explanation"] = "Error generating explanation."
    else:  # Handle case where prediction failed earlier
        if return_payload["explanation"] is None:
            return_payload["explanation"] = "Prediction could not be made."

    return return_payload


# --- End predict_winner ---


# --- Testing Block (__main__) ---
async def main_test():
    # ... (Keep existing test logic) ...
    print("\n[INFO] Running predictor.py directly for testing...")
    if not DJANGO_LOADED:
        print("\n[FAILURE] Cannot run test.")
        return
    test_match = {
        "team1": "Kolkata Knight Riders",
        "team2": "Mumbai Indians",
        "toss_winner": "Mumbai Indians",
        "toss_decision": "field",
        "venue": "Eden Gardens",
        "city": "Kolkata",
        "match_date": "2012-04-27",
    }  # <-- Use valid date
    print(f"\nTest Input Data: {test_match}")
    try:
        result = await predict_winner(input_data=test_match)
        winner = result.get("prediction")
        confidence = result.get("confidence")
        explanation = result.get("explanation")
        print(f"\n[ML Prediction Result]: {winner if winner else 'N/A'}")
        print(f"[ML Confidence]: {confidence if confidence is not None else 'N/A'}")
        print(
            f"[LLM Explanation Result]: {explanation if explanation else 'Not Available'}"
        )
        if winner is None:
            print("\n[INFO] Prediction returned None. Check logs.")
        print("\n[SUCCESS] Predictor test finished.")
    except FileNotFoundError:
        print("\n[FAILURE] Model/Encoder file not found. Check paths/filenames.")
    except Exception as e:
        print(f"\n[FAILURE] Prediction test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main_test())
