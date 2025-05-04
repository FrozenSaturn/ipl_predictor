# main_fastapi.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os
from typing import Dict, Any, Optional  # Added Union
from contextlib import asynccontextmanager

# --- Add src directory to Python path ---
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
# --- End Path Handling ---

# --- Import predictor functions ---
try:
    # Import both predictor functions and their loaders
    from ipl_predictor.predictor import (
        predict_winner,
        load_winner_pipeline_and_encoder,
        predict_score,
        load_score_pipeline,  # Added score predictor imports
    )

    PREDICTORS_LOADED = True
except ImportError as e:
    print(f"ERROR: Failed to import from 'ipl_predictor': {e}. Check sys.path/module.")
    PREDICTORS_LOADED = False

    # Define dummy functions if import fails, so app can start but endpoints fail
    async def predict_winner(*args, **kwargs):
        return {
            "prediction": None,
            "confidence": None,
            "explanation": "Predictor Unavailable",
        }

    async def predict_score(*args, **kwargs):
        return {"predicted_score": None, "error": "Predictor Unavailable"}

    def load_winner_pipeline_and_encoder():
        raise ImportError("Winner predictor failed to load")

    def load_score_pipeline():
        raise ImportError("Score predictor failed to load")


# --- End Imports ---


# --- Define Data Models (Pydantic) ---
# Input model (used by both endpoints)
class MatchInput(BaseModel):
    team1: str = Field(..., example="Chennai Super Kings")
    team2: str = Field(..., example="Rajasthan Royals")
    toss_winner: str = Field(..., example="Rajasthan Royals")
    toss_decision: str = Field(..., example="field", description="'field' or 'bat'")
    venue: str = Field(..., example="MA Chidambaram Stadium")
    city: str = Field(..., example="Chennai")
    match_date: str = Field(..., example="2024-05-10", description="Date (YYYY-MM-DD)")


# Output model for Winner Prediction
class WinnerPredictionOutput(BaseModel):
    prediction: Optional[str]  # Changed from predicted_winner for consistency
    confidence: Optional[float] = Field(None, example=0.75)
    explanation: Optional[str] = Field(None, example="Team X might be favored...")


# Output model for Score Prediction (NEW)
class ScorePredictionOutput(BaseModel):
    predicted_score: Optional[float] = Field(None, example=185.5)
    error: Optional[str] = Field(
        None, example="Feature retrieval failed."
    )  # Include potential errors


# --- End Data Models ---


# --- Lifespan Context Manager (Load BOTH models) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup:
    print("Application startup: Loading ML pipelines via lifespan...")
    models_loaded_successfully = True
    try:
        load_winner_pipeline_and_encoder()  # Load winner pipeline + encoder
        print("Winner pipeline and encoder loaded successfully via lifespan.")
    except Exception as e:
        models_loaded_successfully = False
        print(f"ERROR (lifespan): Failed to load winner pipeline/encoder: {e}")
    try:
        load_score_pipeline()  # Load score pipeline
        print("Score pipeline loaded successfully via lifespan.")
    except Exception as e:
        models_loaded_successfully = False
        print(f"ERROR (lifespan): Failed to load score pipeline: {e}")

    if not models_loaded_successfully:
        print(
            "WARNING: One or more models failed to load on startup. Related endpoints may fail."
        )

    yield  # The application runs while yielded

    # Code to run on shutdown (optional):
    print("Application shutdown.")


# --- End Lifespan Context Manager ---


# --- Initialize FastAPI App ---
app = FastAPI(
    title="IPL Predictor API (FastAPI)",
    description="API for predicting IPL match winners and first innings scores.",
    version="0.2.0",  # Incremented version
    lifespan=lifespan,  # Use lifespan for startup loading
)


# --- Winner Prediction Endpoint ---
@app.post(
    "/predict_winner",  # Renamed endpoint slightly for clarity
    response_model=WinnerPredictionOutput,
    summary="Predict IPL Match Winner with Explanation",
    tags=["Predictions"],
)
async def post_predict_winner(match_input: MatchInput):
    """Receives match details, predicts winner, confidence, and explanation."""
    print(f"Received winner prediction request: {match_input.model_dump()}")
    if not PREDICTORS_LOADED:
        raise HTTPException(
            status_code=503, detail="Predictor service unavailable (load failure)."
        )

    input_dict: Dict[str, Any] = match_input.model_dump()
    try:
        prediction_result = await predict_winner(input_data=input_dict)
        # Directly populate the Pydantic model
        return WinnerPredictionOutput(
            prediction=prediction_result.get("prediction"),
            confidence=prediction_result.get("confidence"),
            explanation=prediction_result.get("explanation"),
        )
    except Exception as e:  # Catch unexpected errors from predictor call
        print(f"UNEXPECTED ERROR during winner prediction: {e}")
        raise HTTPException(
            status_code=500, detail=f"Internal server error during prediction: {e}"
        )


# --- Score Prediction Endpoint (NEW) ---
@app.post(
    "/predict_score",
    response_model=ScorePredictionOutput,
    summary="Predict IPL First Innings Score",
    tags=["Predictions"],
)
async def post_predict_score(match_input: MatchInput):
    """Receives match details, predicts the likely first innings score."""
    print(f"Received score prediction request: {match_input.model_dump()}")
    if not PREDICTORS_LOADED:
        raise HTTPException(
            status_code=503, detail="Predictor service unavailable (load failure)."
        )

    input_dict: Dict[str, Any] = match_input.model_dump()
    try:
        score_result = await predict_score(input_data=input_dict)
        # Directly populate the Pydantic model
        return ScorePredictionOutput(
            predicted_score=score_result.get("predicted_score"),
            error=score_result.get("error"),  # Pass potential error message
        )
    except Exception as e:  # Catch unexpected errors from predictor call
        print(f"UNEXPECTED ERROR during score prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during score prediction: {e}",
        )


# --- Root Endpoint ---
@app.get("/", tags=["General"])
async def read_root():
    return {"message": "IPL Predictor API (FastAPI) is running. Use /docs for details."}
