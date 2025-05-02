# predictor_api/views.py

import sys
import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, viewsets, permissions
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.filters import SearchFilter, OrderingFilter
from drf_spectacular.utils import extend_schema

from .models import Team, Venue, Match, Player
from .serializers import (
    TeamSerializer,
    VenueSerializer,
    MatchSerializer,
    MatchInputSerializer,  # Assuming this exists in serializers.py
    PredictionOutputSerializer,  # Assuming this exists in serializers.py
    PlayerSerializer,
)

# --- Add src directory to Python path for robust imports ---
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
# --- End Path Handling ---

# --- Import predictor functions and load model ---
# (Keep your existing robust import and loading logic)
try:
    from ipl_predictor.predictor import (
        predict_winner,
        load_pipeline,
    )

    load_pipeline()
    print("Django View: ML Pipeline loaded via load_pipeline() on module import.")
    PREDICTOR_AVAILABLE = True
except ImportError as e:
    print(
        f"ERROR (Django View): Failed to import from 'ipl_predictor': {e}."
        f" Prediction endpoint will be unavailable."
    )
    PREDICTOR_AVAILABLE = False
    # predict_winner = None  # Ensure predict_winner is defined but None
except FileNotFoundError as e:
    print(
        f"ERROR (Django View): Model file not found during"
        f" initial load: {e}. Prediction endpoint will be unavailable."
    )
    PREDICTOR_AVAILABLE = False
    # predict_winner = None
except Exception as e:
    print(
        f"ERROR (Django View): Unexpected error during initial load: {e}."
        f" Prediction endpoint will be unavailable."
    )
    PREDICTOR_AVAILABLE = False
    # predict_winner = None
# --- End Imports ---


# =============================================================================
# ViewSet for Database Model Access (Read Only)
# =============================================================================


class PlayerViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Player.objects.all()
    serializer_class = PlayerSerializer
    # permission_classes = [permissions.AllowAny]
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = ["name"]  # Allow filtering like /api/v1/players/?name=V%20Kohli
    search_fields = ["name"]  # Allow searching like /api/v1/players/?search=Kohli
    ordering_fields = ["name", "created_at"]
    ordering = ["name"]  # Default sort by name


class TeamViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Team.objects.all()
    serializer_class = TeamSerializer
    permission_classes = [permissions.AllowAny]
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = ["name", "short_name"]
    search_fields = ["name", "short_name"]
    ordering_fields = ["name", "short_name"]


class VenueViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Venue.objects.all()
    serializer_class = VenueSerializer
    permission_classes = [permissions.AllowAny]
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = ["name", "city", "country"]
    search_fields = ["name", "city"]
    ordering_fields = ["name", "city"]


class MatchViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Match.objects.select_related(
        "team1", "team2", "venue", "toss_winner", "winner"
    ).all()
    serializer_class = MatchSerializer
    permission_classes = [permissions.AllowAny]
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = {
        "season": ["exact", "in"],
        "date": ["exact", "gte", "lte", "range"],
        "venue__name": ["exact", "icontains"],
        "venue__city": ["exact", "icontains"],
        "team1__name": ["exact", "icontains"],
        "team2__name": ["exact", "icontains"],
        "winner__name": ["exact", "icontains", "isnull"],
        "result_type": ["exact"],
    }
    search_fields = ["team1__name", "team2__name", "venue__name", "venue__city"]
    ordering_fields = ["date", "season", "match_number", "venue__name"]
    ordering = ["-date"]


# =============================================================================
# API View for Predictions
# =============================================================================


class PredictionView(APIView):
    # permission_classes = [permissions.AllowAny]

    @extend_schema(
        summary="Predict IPL Match Winner",  # Add a concise summary
        description="Takes match details (teams, venue, toss) and returns"
        " the predicted winner along with an LLM explanation.",
        request=MatchInputSerializer,
        responses={
            200: PredictionOutputSerializer,
            400: {
                "description": "Bad Request: Input validation failed"
            },  # Example error response
            503: {
                "description": "Service Unavailable: Predictor not loaded or model file missing"
            },  # Example error response
            500: {
                "description": "Internal Server Error: Unexpected prediction error"
            },  # Example error response
        },
        tags=[
            "Predictions"
        ],  # Optional: Group this endpoint under a 'Predictions' tag in Swagger UI
    )
    def post(self, request, *args, **kwargs):
        """
        Handles POST requests. Expects JSON data matching MatchInputSerializer.
        Returns prediction and explanation.
        """
        print(f"Django PredictionView received POST request data: {request.data}")

        # Check if predictor loaded correctly
        if not PREDICTOR_AVAILABLE or predict_winner is None:
            print("ERROR: Prediction function not available (import/load failed?).")
            return Response(
                {
                    "error": "Prediction service is currently unavailable due to an internal error."
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        # 1. Validate Input Data
        input_serializer = MatchInputSerializer(data=request.data)
        if not input_serializer.is_valid():
            print(f"Input validation failed: {input_serializer.errors}")
            return Response(input_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        validated_input_data = input_serializer.validated_data
        print(f"Validated input data: {validated_input_data}")

        # 2. Call Prediction Logic
        try:
            # Pass the dictionary directly to the predictor function
            prediction_result = predict_winner(input_data=validated_input_data)

        except FileNotFoundError:
            error_detail = (
                "Model file not found during prediction. Service unavailable."
            )
            print(f"ERROR during Django prediction: {error_detail}")
            # Log the full error internally if possible
            return Response(
                {"error": error_detail}, status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        except (ValueError, RuntimeError) as e:
            # Handle errors expected from the prediction logic (e.g., invalid feature value)
            error_detail = f"Prediction error: {str(e)}"
            print(f"ERROR during Django prediction: {error_detail}")
            status_code = (
                status.HTTP_400_BAD_REQUEST  # Bad input leading to value error
                if isinstance(e, ValueError)
                else status.HTTP_500_INTERNAL_SERVER_ERROR  # Unexpected runtime issue
            )
            return Response({"error": error_detail}, status=status_code)
        except Exception as e:
            # Catch-all for other unexpected errors
            error_detail = "An unexpected error occurred during prediction."
            # Log the actual exception e internally for debugging
            print(f"UNEXPECTED ERROR during Django prediction: {e}", file=sys.stderr)
            return Response(
                {"error": error_detail}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # 3. Serialize Output Data
        try:
            # Instantiate serializer directly with the result dictionary/object
            output_serializer = PredictionOutputSerializer(prediction_result)
            # Access the serialized data directly via .data
            # NO NEED to call .is_valid() here
            response_data = output_serializer.data
        except Exception as e:
            # Keep error handling for potential issues during serialization itself
            error_detail = "An error occurred formatting the prediction response."
            print(f"ERROR during Django output serialization: {e}", file=sys.stderr)
            # Consider logging the actual prediction_result here for debugging
            # print(f"Data that failed serialization: {prediction_result}")
            return Response(
                {"error": error_detail}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        print(f"Prediction successful. Returning response: {response_data}")
        return Response(response_data, status=status.HTTP_200_OK)
