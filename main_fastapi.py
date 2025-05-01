# main_fastapi.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import sys
import os
from typing import List, Dict, Any

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


# --- Initialize FastAPI App ---
# Provides metadata for documentation (visible at /docs)
app = FastAPI(
    title="IPL Match Winner" " Predictor API",
    description="A simple API using a basic"
    " ML model to predict the winner of an IPL match.",
    version="0.1.0",
)
# --- End App Initialization ---


# --- Define Data Models (using Pydantic) ---
# Input model: Defines the structure and types expected in the request body
# Includes example values for documentation clarity
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


# Output model: Defines the structure of the response
class PredictionOutput(BaseModel):
    predicted_winner: str


# --- End Data Models ---


# --- Application Startup Event ---
@app.on_event("startup")
async def startup_load_model():
    """
    Load the ML pipeline into memory when the FastAPI application starts.
    This avoids loading delay on the first prediction request.
    """
    print("Application startup: Loading ML model pipeline...")
    try:
        load_pipeline()  # Call the loader function from predictor.py
        print("ML model pipeline loaded successfully.")
    except FileNotFoundError:
        print(
            "ERROR: Model file not found during startup."
            " Predictions will fail until model is available."
        )
    except Exception as e:
        # Log other errors during loading
        print(
            f"ERROR: An unexpected error" f" occurred loading the model on startup: {e}"
        )


# --- End Startup Event ---


# --- Prediction Endpoint ---
@app.post(
    "/predict",
    response_model=PredictionOutput,
    summary="Predict IPL Match Winner",
    tags=["Predictions"],
)  # Tags group endpoints in the API docs
async def post_predict_winner(match_input: MatchInput):
    """
    Receives match details via POST "
    "request body, uses the trained model
    to predict the winner, and returns the prediction.
    """
    print(f"Received prediction request for input: {match_input.dict()}")

    # Convert the Pydantic model input into a standard dictionary
    input_dict: Dict[str, Any] = match_input.dict()

    try:
        # Call the prediction function from our predictor module
        prediction_list: List[str] = predict_winner(input_data=input_dict)

        # Basic validation of the prediction result
        if not prediction_list or len(prediction_list) == 0:
            print("ERROR: Prediction function returned empty list.")
            # Use HTTPException to return standard HTTP errors
            raise HTTPException(
                status_code=500,  # Internal Server Error
                detail="Prediction failed: Model did not return a winner.",
            )

        # Extract the winner (assuming the list contains one prediction)
        winner: str = prediction_list[0]
        print(f"Prediction successful: {winner}")

        # Return the prediction in the specified response format
        return PredictionOutput(predicted_winner=winner)

    except FileNotFoundError:
        error_detail = (
            "Model file not found. Please ensure" " the model is trained and available."
        )
        print(f"ERROR during prediction: {error_detail}")
        raise HTTPException(status_code=503, detail=error_detail)
    except (ValueError, RuntimeError) as e:
        error_detail = f"Prediction error: {str(e)}"
        print(f"ERROR during prediction: {error_detail}")
        raise HTTPException(
            status_code=400,
            detail=error_detail,
        )
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
