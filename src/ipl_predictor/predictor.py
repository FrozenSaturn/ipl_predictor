# src/ipl_predictor/predictor.py

import joblib
import pandas as pd
import os
import sys
from typing import Dict, Any, Optional, Union
import requests

# import json
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
    from predictor_api.models import (
        Match,
        Team,
    )  # PlayerPerformance not needed for current feature fetch
    from django.db.models import Q

    DJANGO_LOADED = True
except Exception as e:
    print(f"ERROR: Failed Django setup: {e}")
    # Match = None
    # Team = None
    # Q = None
    DJANGO_LOADED = False
# --- End Django Setup ---


# --- Configuration Constants ---
MODEL_DIR: str = "models"
# --- POINT TO THE LATEST MODEL TRAINED WITH ALL FEATURES ---
MODEL_FILENAME: str = (
    "ipl_winner_pipeline_rf_formfeat_balanced_tuned_v1.joblib"  # <-- Use latest saved model
)
MODEL_PATH: str = os.path.join(MODEL_DIR, MODEL_FILENAME)

OLLAMA_API_URL: str = "http://localhost:11434/api/generate"
OLLAMA_MODEL: str = "phi3:instruct"
N_RECENT_MATCHES_FORM: int = 5  # Kept for consistency, but logic simplified below
DEFAULT_WIN_PCT = 0.0
DEFAULT_H2H_WIN_PCT = 0.5
DEFAULT_SCORE = 150
DEFAULT_WKTS = 5
DEFAULT_BATTING_SR = 130.0
DEFAULT_BOWLING_ECON = 8.5  # Keep defaults even if features not fully used below
# --- End Configuration ---

_pipeline: Optional[Any] = None


def load_pipeline() -> Any:  # --- load_pipeline (Unchanged) ---
    global _pipeline
    if _pipeline is None:
        print(f"Attempting to load model pipeline from: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model pipeline not found: '{MODEL_PATH}'.")
        try:
            _pipeline = joblib.load(MODEL_PATH)
            print("Model pipeline loaded.")
        except Exception as e:
            print(f"ERROR loading pipeline: {e}")
            raise
    return _pipeline


def get_llm_explanation(
    prompt: str,
) -> Optional[str]:  # --- get_llm_explanation (Unchanged) ---
    print(f"\nSending prompt to Ollama ({OLLAMA_MODEL})...")
    try:  # ... (LLM call logic unchanged) ...
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


# --- Synchronous DB Feature Fetch Helper (Fetches 11 Numerical Features) ---
def _fetch_all_features_sync(
    team1_name: str, team2_name: str, prediction_date: date
) -> Optional[Dict[str, Any]]:
    """Synchronous helper to fetch all pre-calculated features from Match model fields AND calc basic form."""
    if not DJANGO_LOADED:
        return None
    print(
        f"  DB Lookup for features: {team1_name} vs {team2_name} before {prediction_date}"
    )
    features: Dict[str, Any] = {}
    try:
        team1_obj = Team.objects.get(name=team1_name)
        team2_obj = Team.objects.get(name=team2_name)

        # Find relevant last matches from Match table
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

        # Extract basic historical features directly from Match fields
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

        # Placeholder/Default logic for Player Form Features (as they weren't properly saved/calculated yet)
        # In a real scenario, this would query PlayerMatchPerformance or fetch from Match fields if populated
        features["team1_avg_recent_bat_sr"] = DEFAULT_BATTING_SR
        features["team1_avg_recent_bowl_econ"] = DEFAULT_BOWLING_ECON
        features["team2_avg_recent_bat_sr"] = DEFAULT_BATTING_SR
        features["team2_avg_recent_bowl_econ"] = DEFAULT_BOWLING_ECON

        # Apply defaults for ALL required features if DB lookup returned None
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
) -> Optional[Dict[str, float]]:
    """Asynchronously calls the synchronous DB lookup helper."""
    print("Fetching historical & form features from database (async)...")  # Updated log
    try:
        team1_name = input_data["team1"]
        team2_name = input_data["team2"]
        prediction_date = datetime.strptime(input_data["match_date"], "%Y-%m-%d").date()
        # Call the helper function which now returns all 11 numerical features
        features = await sync_to_async(_fetch_all_features_sync, thread_sensitive=True)(
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
async def predict_winner(
    input_data: Dict[str, Any]
) -> Dict[str, Optional[Union[str, float]]]:
    """Predicts winner, confidence, gets LLM explanation using DB features async."""
    pipeline = load_pipeline()  # Loads the model trained with 17 features
    return_payload: Dict[str, Optional[Union[str, float]]] = {
        "prediction": None,
        "confidence": None,
        "explanation": None,
    }

    if "match_date" not in input_data:  # Keep date check
        return_payload["explanation"] = (
            "Missing 'match_date' (YYYY-MM-DD) in input_data."
        )
        return return_payload

    # Fetch all 11 engineered features
    engineered_features = await get_features_for_match(input_data)
    if engineered_features is None:
        return_payload["explanation"] = (
            "Failed to retrieve features needed for prediction."
        )
        return return_payload

    # Prepare Input DataFrame (using all 17 features)
    try:
        combined_data = {**input_data, **engineered_features}
        # This list MUST match the features the loaded pipeline expects
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
            "team1_avg_recent_bowl_econ",  # Team Form (4)
            "team2_avg_recent_bat_sr",
            "team2_avg_recent_bowl_econ",
        ]  # Total 17 features
        input_df = pd.DataFrame([combined_data])
        if not all(col in input_df.columns for col in expected_cols):
            missing = [col for col in expected_cols if col not in input_df.columns]
            raise ValueError(
                f"DataFrame missing columns: {missing}. Check feature fetcher."
            )
        input_df = input_df[expected_cols]  # Ensure columns/order match training
        print(
            f"\nInput DataFrame for prediction (ALL features):\n{input_df.iloc[0].to_dict()}"
        )
    except Exception as e:
        print(f"ERROR: Failed to process combined input data: {e}")
        return_payload["explanation"] = f"Internal error processing input data: {e}"
        return return_payload

    # Make ML Prediction & Get Probabilities (Unchanged logic)
    try:  # ... (predict, predict_proba, extract winner/confidence) ...
        print("Calling pipeline.predict() and pipeline.predict_proba()...")
        ml_prediction = pipeline.predict(input_df)
        ml_probabilities = pipeline.predict_proba(input_df)
        if not ml_prediction.any():
            raise RuntimeError("ML prediction empty.")
        predicted_winner = ml_prediction[0]
        return_payload["prediction"] = predicted_winner
        try:
            if hasattr(pipeline, "classes_"):
                winner_index = list(pipeline.classes_).index(predicted_winner)
                confidence = ml_probabilities[0, winner_index]
                return_payload["confidence"] = float(
                    Decimal(str(confidence)).quantize(
                        Decimal("0.0001"), rounding=ROUND_HALF_UP
                    )
                )
                print(
                    f"[ML Prediction] -> Winner: {predicted_winner}, Confidence: {confidence:.4f}"
                )
            else:
                print(f"[ML Prediction] -> Winner: {predicted_winner}, Confidence: N/A")
        except Exception as inner_e:
            print(f"Warning: Confidence determination failed: {inner_e}")
    except Exception as e:  # ... (Error handling) ...
        print(f"ERROR: ML prediction failed: {e}")
        return_payload["explanation"] = f"ML prediction failed: {e}"
        return_payload["prediction"] = None
        return_payload["confidence"] = None
        return return_payload

    # Get LLM Explanation (Unchanged logic, uses improved prompt)
    if return_payload["prediction"] is not None:
        try:  # ... (Generate prompt using input_data and engineered_features for context) ...
            # Format features for the prompt (now includes form features)
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
            t1_ps = engineered_features.get("team1_prev_score", DEFAULT_SCORE)
            t1_pw = engineered_features.get("team1_prev_wkts", DEFAULT_WKTS)
            t2_ps = engineered_features.get("team2_prev_score", DEFAULT_SCORE)
            t2_pw = engineered_features.get("team2_prev_wkts", DEFAULT_WKTS)
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
            confidence_val = return_payload.get("confidence")
            pred_conf: str = "N/A"  # Default to N/A string
            # Only try to round if it's actually a number (float or int)
            if isinstance(confidence_val, (float, int)):
                try:
                    # Use Decimal for stable rounding, convert back to formatted string
                    pred_conf = f"{Decimal(str(confidence_val * 100)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)}"
                except Exception:
                    pred_conf = "Error"

            prompt = f"""
                Act as a concise cricket analyst providing a brief insight based *only* on the provided pre-match data.

                **Match Context:**
                * Match: {input_data['team1']} vs {input_data['team2']}
                * Venue: {input_data['venue']}, {input_data['city']}
                * Toss: {input_data['toss_winner']} won, chose to {input_data['toss_decision']}.

                **Prediction Details:**
                * Predicted Winner: {predicted_winner} (Confidence: {pred_conf}%)

                **Relevant Pre-Match Statistics:**
                * Historical Win %: {input_data['team1']} ({t1_wp}%) vs {input_data['team2']} ({t2_wp}%)
                * Head-to-Head Win % ({input_data['team1']} first): {h2h_wp}%
                * Previous Match Score/Wickets: {input_data['team1']} ({t1_ps}/{t1_pw}) vs {input_data['team2']} ({t2_ps}/{t2_pw})
                * Team Recent Form (Avg):
                    * Bat SR: {input_data['team1']} ({t1_sr}) vs {input_data['team2']} ({t2_sr})
                    * Bowl Econ: {input_data['team1']} ({t1_ec}) vs {input_data['team2']} ({t2_ec})

                **Your Task:**
                Based *strictly* on the statistics provided above, identify ONE key statistical factor
                (e.g., historical win %, H2H record, recent form indicators, toss advantage) that likely favors the
                predicted winner ({predicted_winner}). Explain its potential significance in **one single, concise sentence.**

                **Constraints:**
                * **One sentence only.**
                * Base your reason **only** on the provided statistics.
                * **Do NOT** use external knowledge or hallucinate facts (e.g., player injuries, news, specific match events).
                * **Do NOT** mention the prediction model or its confidence.
                * Start the sentence directly (no preamble like "One reason is...").
                """

            prompt = "\n".join(line.strip() for line in prompt.strip().splitlines())
            explanation = get_llm_explanation(prompt=prompt)
            return_payload["explanation"] = explanation
        except Exception:
            return_payload["explanation"] = "Error generating explanation."
    else:
        if return_payload["explanation"] is None:
            return_payload["explanation"] = "Prediction could not be made."

    return return_payload


# --- End predict_winner ---


# --- Testing Block (__main__) ---
async def main_test():  # (Unchanged logic, uses asyncio.run)
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
        print(f"\n[FAILURE] Model file '{MODEL_FILENAME}' not found.")
    except Exception as e:
        print(f"\n[FAILURE] Prediction test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main_test())  # Use asyncio.run
