# from django.shortcuts import render

# predictor_api/views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status  # For HTTP status codes
from .serializers import (
    MatchInputSerializer,
    PredictionOutputSerializer,
)  # Import our serializers
import sys
import os

# --- Add src directory to Python path for robust imports ---
# Necessary for Django context to find the predictor module
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
# --- End Path Handling ---

# --- Import predictor functions ---
try:
    from ipl_predictor.predictor import (
        predict_winner,
        load_pipeline,
    )  # Import prediction function

    # Call load_pipeline here if not using lifespan/startup event equivalent in Django
    # Or rely on the lazy loading within predict_winner
    load_pipeline()  # Ensure model loads when this module is loaded by Django
    print("Django View: ML Pipeline loaded via load_pipeline() on module import.")
except ImportError as e:
    print("ERROR (Django View): Failed to import from 'ipl_predictor'. Check sys.path.")
    # Optionally raise error to prevent app from starting if model is critical
    raise ImportError(f"Could not import predictor: {e}") from e
    # For now, allow app to run, prediction will fail later if import failed
except FileNotFoundError as e:
    print(f"ERROR (Django View): Model file not found during initial load: {e}")
except Exception as e:
    print(f"ERROR (Django View): Unexpected error during initial load: {e}")
# --- End Imports ---


class PredictionView(APIView):
    """
    API View to handle IPL match prediction requests using Django REST Framework.
    """

    def post(self, request, *args, **kwargs):
        """
        Handles POST requests to the /api/v1/predict/ endpoint.
        Expects JSON data matching MatchInputSerializer.
        Returns prediction and explanation.
        """
        print(f"Django view received POST request data: {request.data}")

        # 1. Validate Input Data using the Serializer
        input_serializer = MatchInputSerializer(data=request.data)
        if not input_serializer.is_valid():
            print(f"Input validation failed: {input_serializer.errors}")
            return Response(input_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        # Validated data is in serializer.validated_data (it's a dictionary)
        validated_input_data = input_serializer.validated_data
        print(f"Validated input data: {validated_input_data}")

        # 2. Call Prediction Logic (Ensure predictor was imported)
        if predict_winner is None:
            print("ERROR: predict_winner function not available (import failed?).")
            return Response(
                {"error": "Prediction service is unavailable."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        try:
            prediction_result = predict_winner(input_data=validated_input_data)

        # Handle specific errors that predict_winner might raise
        except FileNotFoundError:
            error_detail = "Model file not found. Please ensure the model is trained."
            print(f"ERROR during Django prediction: {error_detail}")
            return Response(
                {"error": error_detail}, status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        except (ValueError, RuntimeError) as e:
            error_detail = f"Prediction error: {str(e)}"
            print(f"ERROR during Django prediction: {error_detail}")
            status_code = (
                status.HTTP_400_BAD_REQUEST
                if isinstance(e, ValueError)
                else status.HTTP_500_INTERNAL_SERVER_ERROR
            )
            return Response({"error": error_detail}, status=status_code)
        except Exception as e:
            error_detail = f"An unexpected error occurred during prediction: {str(e)}"
            print(f"UNEXPECTED ERROR during Django prediction: {error_detail}")
            return Response(
                {"error": error_detail}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # 3. Serialize Output Data
        output_serializer = PredictionOutputSerializer(prediction_result)

        print(
            f"Prediction successful." f" Returning response: {output_serializer.data}"
        )
        return Response(output_serializer.data, status=status.HTTP_200_OK)
