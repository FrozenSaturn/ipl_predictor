# main_fastapi.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import sys
import os
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
# --- End Path Handling ---

# --- Import predictor functions ---
try:
    # Attempt to import after potentially modifying sys.path
    from ipl_predictor.predictor import predict_winner, load_pipeline
except ImportError as e:
    print(
        "ERROR: Failed to import from 'ipl_predictor'."
        " Check sys.path and module structure."
    )
    print(f"Current sys.path: {sys.path}")
    # Exit if essential imports fail, as the app cannot function
    sys.exit(f"ImportError: {e}")
# --- End Imports ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup:
    print("Application startup: Loading ML model pipeline via lifespan...")
    try:
        load_pipeline()  # Call the loader function from predictor.py
        print("ML model pipeline loaded successfully via lifespan.")
    except FileNotFoundError:
        print("ERROR (lifespan): Model file not found during startup.")
    except Exception as e:
        print(f"ERROR (lifespan): An unexpected error occurred loading model: {e}")

    yield  # The application runs while yielded

    # Code to run on shutdown (optional):
    print("Application shutdown.")


# --- Initialize FastAPI App ---
# Provides metadata for documentation (visible at /docs)
app = FastAPI(
    title="IPL Match Winner" " Predictor API",
    description="A simple API using a basic"
    " ML model to predict the winner of an IPL match.",
    version="0.1.0",
    lifespan=lifespan,
)
# --- End App Initialization ---


# --- Define Data Models (using Pydantic) ---
# Input model: Defines the structure and types expected in the request body
# Includes example values for documentation clarity
# --- Define Data Models ---
class MatchInput(BaseModel):
    team1: str = Field(..., example="Chennai Super Kings")
    team2: str = Field(..., example="Rajasthan Royals")
    toss_winner: str = Field(..., example="Rajasthan Royals")
    toss_decision: str = Field(
        ...,
        example="field",
        description="Decision made after winning toss ('field' or 'bat')",
    )
    venue: str = Field(..., example="MA Chidambaram Stadium")
    city: str = Field(..., example="Chennai")
    match_date: str = Field(
        ..., example="2024-05-10", description="Date of the match (YYYY-MM-DD)"
    )  # <-- ADDED FIELD


# Output model: Defines the structure of the response
class PredictionOutput(BaseModel):
    predicted_winner: str
    confidence: Optional[float] = Field(None, example=0.75)
    explanation: Optional[str] = Field(
        None,
        example="Mumbai Indians might be favored due to their strong batting lineup.",
    )


# --- End Data Models ---


# --- Prediction Endpoint ---
@app.post(
    "/predict",
    response_model=PredictionOutput,  # Response model now includes explanation
    summary="Predict IPL Match Winner with Explanation",  # Updated summary
    tags=["Predictions"],
)
async def post_predict_winner(match_input: MatchInput):
    """
    Receives match details, uses the ML model to predict the winner,
    queries an LLM for an explanation, and returns both.
    """
    print(f"Received prediction request for input: {match_input.model_dump()}")
    input_dict: Dict[str, Any] = match_input.model_dump()

    try:
        # Call the updated prediction function which returns a dictionary
        prediction_result = await predict_winner(input_data=input_dict)

        # winner: Optional[str] = prediction_result.get("prediction")
        # explanation: Optional[str] = prediction_result.get("explanation")

        # if winner is None:  # Check if ML prediction itself failed within predict_winner
        #     print("ERROR: Prediction function failed to return a winner.")
        #     raise HTTPException(
        #         status_code=500,
        #         detail="Prediction failed: Model did not return a winner.",
        #     )

        # print(
        #     f"Prediction successful: Winner='{winner}',"
        #     f"Explanation='{explanation if explanation else 'N/A'}'"
        # )

        return PredictionOutput(
            predicted_winner=prediction_result.get("prediction"),
            confidence=prediction_result.get("confidence"),
            explanation=prediction_result.get("explanation"),
        )

    except FileNotFoundError:
        error_detail = (
            "Model file not found. Please ensure the model is trained and available."
        )
        print(f"ERROR during prediction: {error_detail}")
        raise HTTPException(status_code=503, detail=error_detail)
    except (ValueError, RuntimeError) as e:
        error_detail = f"Prediction error: {str(e)}"
        print(f"ERROR during prediction: {error_detail}")
        # Use 400 for input errors (ValueError), 500 for runtime issues
        status_code = 400 if isinstance(e, ValueError) else 500
        raise HTTPException(status_code=status_code, detail=error_detail)
    except Exception as e:
        error_detail = f"An unexpected error occurred: {str(e)}"
        print(f"UNEXPECTED ERROR during prediction: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)


# --- End Prediction Endpoint ---


# --- Optional: Root Endpoint for Basic Check ---
@app.get("/", tags=["General"])
async def read_root():
    """Basic endpoint to check if the API is running."""
    return {
        "message": "IPL Predictor API is running."
        " Use the /docs endpoint for details."
    }


# --- End Root Endpoint ---
