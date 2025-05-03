# predictor_api/views.py

import sys
import os
from datetime import date  # <-- Added import for date object type check
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import (
    status,
    viewsets,
)  # Keep permissions if needed later
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.filters import SearchFilter, OrderingFilter
from drf_spectacular.utils import extend_schema
from asgiref.sync import async_to_sync  # Keep this import

from .models import Team, Venue, Match, Player
from .serializers import (
    TeamSerializer,
    VenueSerializer,
    MatchSerializer,
    MatchInputSerializer,
    PredictionOutputSerializer,
    PlayerSerializer,
)

# --- Add src directory to Python path ---
# (Keep this block as is)
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
# --- End Path Handling ---

# --- Import predictor functions and load model ---
# (Keep this block as is, including PREDICTOR_AVAILABLE flag logic)
try:
    from ipl_predictor.predictor import predict_winner, load_pipeline

    load_pipeline()
    print("Django View: ML Pipeline loaded via load_pipeline() on module import.")
    PREDICTOR_AVAILABLE = True
    # Ensure predict_winner is defined even if import failed within try block for safety
    # Though PREDICTOR_AVAILABLE flag handles the logic flow
except ImportError as e:
    print(
        f"ERROR (Django View): Failed to import from 'ipl_predictor': {e}. Prediction endpoint will be unavailable."
    )
    PREDICTOR_AVAILABLE = False
    # predict_winner = None  # Define as None on failure
except FileNotFoundError as e:
    print(
        f"ERROR (Django View): Model file not found during initial load: {e}. Prediction endpoint will be unavailable."
    )
    PREDICTOR_AVAILABLE = False
    # predict_winner = None
except Exception as e:
    print(
        f"ERROR (Django View): Unexpected error during initial load: {e}. Prediction endpoint will be unavailable."
    )
    PREDICTOR_AVAILABLE = False
    # predict_winner = None
# --- End Imports ---


# =============================================================================
# ViewSet for Database Model Access (Read Only)
# =============================================================================
# (Keep your existing ViewSet definitions for Player, Team, Venue, Match - they are correct)


class PlayerViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Player.objects.all()
    serializer_class = PlayerSerializer
    # permission_classes = [permissions.IsAuthenticated] # Uncomment if needed
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = ["name"]
    search_fields = ["name"]
    ordering_fields = ["name", "created_at"]
    ordering = ["name"]


class TeamViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Team.objects.all()
    serializer_class = TeamSerializer
    # permission_classes = [permissions.IsAuthenticated] # Uncomment if needed
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = ["name", "short_name"]
    search_fields = ["name", "short_name"]
    ordering_fields = ["name", "short_name"]
    ordering = ["name"]


class VenueViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Venue.objects.all()
    serializer_class = VenueSerializer
    # permission_classes = [permissions.IsAuthenticated] # Uncomment if needed
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = ["name", "city", "country"]
    search_fields = ["name", "city"]
    ordering_fields = ["name", "city"]
    ordering = ["name"]


class MatchViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Match.objects.select_related(
        "team1", "team2", "venue", "toss_winner", "winner"
    ).all()
    serializer_class = MatchSerializer
    # permission_classes = [permissions.IsAuthenticated] # Uncomment if needed
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = {  # Using dictionary for more specific lookups
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
    """
    Handles requests to predict match winners using the ML model and LLM explanation.
    Requires Token Authentication by default (if set in settings.py).
    """

    # Make sure appropriate permissions are set, using project default for now
    # permission_classes = [permissions.IsAuthenticated] # Example

    @extend_schema(  # Keep documentation decorator
        summary="Predict IPL Match Winner with Explanation",
        description="Takes match details (teams, venue, toss, date) and returns "
        "the predicted winner along with an LLM explanation.",
        request=MatchInputSerializer,  # Serializer now includes match_date
        responses={
            200: PredictionOutputSerializer,
            400: {"description": "Bad Request: Input validation failed"},
            503: {"description": "Service Unavailable: Predictor/Model unavailable"},
            500: {"description": "Internal Server Error: Prediction failed"},
        },
        tags=["Predictions"],
    )
    def post(self, request, *args, **kwargs):
        """
        Handles POST requests. Expects JSON matching MatchInputSerializer.
        Returns prediction and explanation.
        """
        print(f"Django PredictionView received POST request data: {request.data}")

        if not PREDICTOR_AVAILABLE or predict_winner is None:
            print("ERROR: Prediction function not available (import/load failed?).")
            return Response(
                {"error": "Prediction service is currently unavailable."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        # 1. Validate Input Data
        input_serializer = MatchInputSerializer(data=request.data)
        if not input_serializer.is_valid():
            print(f"Input validation failed: {input_serializer.errors}")
            return Response(input_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        validated_input_data = input_serializer.validated_data
        print(f"Validated input data (before date conversion): {validated_input_data}")

        # --- Convert validated date object back to 'YYYY-MM-DD' string ---
        # The predict_winner function expects the date as a string
        if "match_date" in validated_input_data and isinstance(
            validated_input_data["match_date"], date
        ):
            validated_input_data["match_date"] = validated_input_data[
                "match_date"
            ].isoformat()
            print(f"Converted input data for predictor: {validated_input_data}")
        # --- End date conversion ---

        # 2. Call Prediction Logic (using async_to_sync)
        try:
            # Call the async predictor function from our synchronous view context
            prediction_result = async_to_sync(predict_winner)(
                input_data=validated_input_data
            )

            # Handle potential failure during feature retrieval within predict_winner
            if prediction_result.get(
                "prediction"
            ) is None and "Failed to retrieve features" in prediction_result.get(
                "explanation", ""
            ):
                print(
                    f"ERROR during Django prediction: {prediction_result.get('explanation')}"
                )
                return Response(
                    {"error": prediction_result.get("explanation")},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

        # --- Keep existing error handling ---
        except FileNotFoundError:
            error_detail = (
                "Model file not found during prediction. Service unavailable."
            )
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
            error_detail = "An unexpected error occurred during prediction."
            print(f"UNEXPECTED ERROR during Django prediction: {e}", file=sys.stderr)
            return Response(
                {"error": error_detail}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        # --- End error handling ---

        # 3. Serialize Output Data
        # Pass the dictionary returned by predict_winner to the output serializer
        output_serializer = PredictionOutputSerializer(prediction_result)
        # No need for .is_valid() on output usually, unless complex validation needed

        print(f"Prediction successful. Returning response: {output_serializer.data}")
        return Response(output_serializer.data, status=status.HTTP_200_OK)
