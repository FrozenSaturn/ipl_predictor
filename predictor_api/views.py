# predictor_api/views.py

import sys
import os
from datetime import date
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import action  # Keep for PlayerViewSet action
from rest_framework import status, viewsets, permissions, filters
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.filters import SearchFilter, OrderingFilter
from drf_spectacular.utils import extend_schema
from asgiref.sync import async_to_sync
from typing import Dict, Optional, Union

# Import all necessary models and serializers
from .models import Team, Venue, Match, Player, PlayerMatchPerformance
from .serializers import (
    TeamSerializer,
    VenueSerializer,
    MatchSerializer,
    PlayerSerializer,
    MatchInputSerializer,
    PredictionOutputSerializer,
    PlayerMatchPerformanceSerializer,
    ScorePredictionOutputSerializer,
    LLMQueryInputSerializer,
    LLMQueryOutputSerializer,
)

# --- Add src directory to Python path ---
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
# --- End Path Handling ---

# --- Import predictor functions and load models/encoder ---
PREDICTOR_AVAILABLE = False
SCORE_PREDICTOR_AVAILABLE = False
predict_winner = None
predict_score = None


# Dummy functions for type consistency on import failure
async def _dummy_winner_predictor(
    *args, **kwargs
) -> Dict[str, Optional[Union[str, float]]]:
    return {
        "prediction": None,
        "confidence": None,
        "explanation": "Predictor Unavailable",
    }


# --- Ensure LLM handler is imported (adjust path if needed) ---
try:
    from ipl_predictor.llm_handler import query_ollama_llm

    LLM_HANDLER_AVAILABLE = True
except ImportError:
    print(
        "ERROR (Django View): Failed to import 'query_ollama_llm' from 'ipl_predictor.llm_handler'. LLM query endpoint unavailable."
    )
    LLM_HANDLER_AVAILABLE = False

    def query_ollama_llm(prompt_text: str) -> str:
        return "LLM Handler not available."


async def _dummy_score_predictor(*args, **kwargs) -> Dict[str, Optional[float]]:
    return {"predicted_score": None, "error": "Predictor Unavailable"}


try:
    # Import winner predictor components
    from ipl_predictor.predictor import (
        predict_winner as actual_predict_winner,
        load_winner_pipeline_and_encoder,
    )

    load_winner_pipeline_and_encoder()  # Load winner model/encoder
    print("Django View: Winner Pipeline & Encoder loaded.")
    predict_winner = actual_predict_winner  # Assign actual function
    PREDICTOR_AVAILABLE = True

    # Try importing and loading score predictor components separately
    try:
        from ipl_predictor.predictor import (
            predict_score as actual_predict_score,
            load_score_pipeline,
        )

        load_score_pipeline()  # Load score model
        print("Django View: Score Pipeline loaded.")
        predict_score = actual_predict_score  # Assign actual function
        SCORE_PREDICTOR_AVAILABLE = True
    except (ImportError, FileNotFoundError, Exception) as score_e:
        print(
            f"ERROR (Django View): Failed score model load: {score_e}. Score endpoint unavailable."
        )
        predict_score = _dummy_score_predictor  # Assign dummy score predictor

except (ImportError, FileNotFoundError, Exception) as winner_e:
    print(
        f"ERROR (Django View): Failed winner predictor/model load: {winner_e}. Winner endpoint unavailable."
    )
    predict_winner = _dummy_winner_predictor  # Assign dummy winner predictor
    predict_score = (
        _dummy_score_predictor  # Score also unavailable if base import failed
    )
# --- End Imports ---


# =============================================================================
# ViewSet for Database Model Access (Read Only)
# =============================================================================
# (Keep PlayerViewSet, TeamViewSet, VenueViewSet, MatchViewSet exactly as you provided)


class PlayerViewSet(viewsets.ReadOnlyModelViewSet):
    """API endpoint that allows players to be viewed."""

    queryset = Player.objects.all()
    serializer_class = PlayerSerializer
    permission_classes = [permissions.IsAuthenticated]
    filter_backends = [filters.SearchFilter]
    filterset_fields = ["name"]
    search_fields = ["name"]
    ordering_fields = ["name", "created_at"]
    ordering = ["name"]
    pagination_class = None

    @action(
        detail=True,
        methods=["get"],
        url_path="recent-performance",
        url_name="recent-performance",
    )
    @extend_schema(
        summary="Get Player Recent Performance",
        description="Returns performance statistics for the player's last 5 matches.",
        responses={
            200: PlayerMatchPerformanceSerializer(many=True),
            404: {"description": "Player not found or no performance data available."},
        },
        tags=["Players"],
    )
    def recent_performance(self, request, pk=None):
        """Returns the performance stats for the player's last 5 matches."""
        try:
            player = self.get_object()
        except Player.DoesNotExist:
            return Response(
                {"detail": "Player not found."}, status=status.HTTP_404_NOT_FOUND
            )
        num_matches = 5
        recent_performances = (
            PlayerMatchPerformance.objects.filter(player=player)
            .select_related("match", "match__venue")
            .order_by("-match__date")[:num_matches]
        )
        if not recent_performances.exists():
            return Response(
                {"detail": "No performance data available."},
                status=status.HTTP_404_NOT_FOUND,
            )
        serializer = PlayerMatchPerformanceSerializer(recent_performances, many=True)
        return Response(serializer.data)


class TeamViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Team.objects.all()
    serializer_class = TeamSerializer
    permission_classes = [permissions.IsAuthenticated]
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = ["name", "short_name"]
    search_fields = ["name", "short_name"]
    ordering_fields = ["name", "short_name"]
    ordering = ["name"]


class VenueViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Venue.objects.all()
    serializer_class = VenueSerializer
    permission_classes = [permissions.IsAuthenticated]
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
    permission_classes = [permissions.IsAuthenticated]
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
# API View for Winner Predictions
# =============================================================================
# (Keep your existing PredictionView exactly as provided)
class PredictionView(APIView):  # Keeping original name as requested
    """Handles requests to predict match winners."""

    permission_classes = [permissions.IsAuthenticated]

    @extend_schema(
        summary="Predict IPL Match Winner with Explanation",
        description="Takes match details (teams, venue, toss, date) and returns the predicted winner, confidence score, and an LLM explanation.",
        request=MatchInputSerializer,
        responses={
            200: PredictionOutputSerializer,
            400: {"description": "Bad Request"},
            503: {"description": "Service Unavailable"},
            500: {"description": "Internal Server Error"},
        },
        tags=["Predictions"],
    )
    def post(self, request, *args, **kwargs):
        """Handles POST requests for winner prediction."""
        print(f"Django PredictionView received POST request data: {request.data}")

        if not PREDICTOR_AVAILABLE:  # Check winner predictor flag
            return Response(
                {"error": "Winner prediction service unavailable."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        input_serializer = MatchInputSerializer(data=request.data)
        if not input_serializer.is_valid():
            return Response(input_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        validated_input_data = input_serializer.validated_data
        print(f"Validated input data (before date conversion): {validated_input_data}")

        if "match_date" in validated_input_data and isinstance(
            validated_input_data["match_date"], date
        ):
            validated_input_data["match_date"] = validated_input_data[
                "match_date"
            ].isoformat()
            print(f"Converted input data for predictor: {validated_input_data}")

        try:
            prediction_result = async_to_sync(predict_winner)(
                input_data=validated_input_data
            )
            # Handle potential errors returned from predictor
            if (
                prediction_result.get("prediction") is None
                and prediction_result.get("explanation") is not None
            ):
                print(
                    f"ERROR during Django winner prediction: {prediction_result.get('explanation')}"
                )
                status_code = (
                    status.HTTP_503_SERVICE_UNAVAILABLE
                    if "Model loading failed"
                    in prediction_result.get("explanation", "")
                    else status.HTTP_500_INTERNAL_SERVER_ERROR
                )
                return Response(
                    {"error": prediction_result.get("explanation")}, status=status_code
                )

        except Exception as e:  # Catch errors from async_to_sync or predictor itself
            error_detail = f"Unexpected error during winner prediction: {str(e)}"
            print(
                f"UNEXPECTED ERROR during Django winner prediction: {e}",
                file=sys.stderr,
            )
            return Response(
                {"error": error_detail}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # Serialize Output Data using the original PredictionOutputSerializer
        output_serializer = PredictionOutputSerializer(prediction_result)
        print(
            f"Winner prediction successful. Returning response: {output_serializer.data}"
        )
        return Response(output_serializer.data, status=status.HTTP_200_OK)


# =============================================================================
# API View for Score Predictions
# =============================================================================
class ScorePredictionView(APIView):
    """Handles requests to predict first innings score."""

    permission_classes = [permissions.IsAuthenticated]  # Keep consistent auth

    @extend_schema(
        summary="Predict IPL First Innings Score",
        description="Takes match details (teams, venue, toss, date) and returns the predicted first innings score.",
        request=MatchInputSerializer,  # Uses the same input serializer
        responses={
            200: ScorePredictionOutputSerializer,  # Uses the new output serializer
            400: {"description": "Bad Request: Input validation failed"},
            503: {"description": "Service Unavailable: Score predictor not loaded"},
            500: {"description": "Internal Server Error: Unexpected prediction error"},
        },
        tags=["Predictions"],  # Group with other predictions
    )
    def post(self, request, *args, **kwargs):
        """Handles POST requests for score prediction."""
        print(f"Django ScorePredictionView received POST request data: {request.data}")

        if not SCORE_PREDICTOR_AVAILABLE:  # Check specific flag for score predictor
            return Response(
                {"error": "Score prediction service unavailable."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        # 1. Validate Input Data (using same serializer)
        input_serializer = MatchInputSerializer(data=request.data)
        if not input_serializer.is_valid():
            print(f"Input validation failed: {input_serializer.errors}")
            return Response(input_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        validated_input_data = input_serializer.validated_data
        print(f"Validated input data (before date conversion): {validated_input_data}")

        # Convert date object back to 'YYYY-MM-DD' string for predictor
        if "match_date" in validated_input_data and isinstance(
            validated_input_data["match_date"], date
        ):
            validated_input_data["match_date"] = validated_input_data[
                "match_date"
            ].isoformat()
            print(f"Converted input data for predictor: {validated_input_data}")

        # 2. Call Score Prediction Logic
        try:
            # Call the async score predictor function from sync context
            score_result = async_to_sync(predict_score)(input_data=validated_input_data)

            # Check for errors returned within the score_result dictionary
            if (
                score_result.get("predicted_score") is None
                and score_result.get("error") is not None
            ):
                print(
                    f"ERROR during Django score prediction: {score_result.get('error')}"
                )
                status_code = (
                    status.HTTP_503_SERVICE_UNAVAILABLE
                    if "model loading failed" in score_result.get("error", "").lower()
                    else status.HTTP_500_INTERNAL_SERVER_ERROR
                )
                return Response(
                    {"error": score_result.get("error")}, status=status_code
                )

        except Exception as e:  # Catch errors from async_to_sync or predictor itself
            error_detail = f"Unexpected error during score prediction: {str(e)}"
            print(
                f"UNEXPECTED ERROR during Django score prediction: {e}", file=sys.stderr
            )
            return Response(
                {"error": error_detail}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # 3. Serialize Output Data
        # Pass the dictionary returned by predict_score
        output_serializer = ScorePredictionOutputSerializer(score_result)

        print(
            f"Score prediction successful. Returning response: {output_serializer.data}"
        )
        return Response(output_serializer.data, status=status.HTTP_200_OK)


# =============================================================================
# API View for LLM Follow-up Queries
# =============================================================================
class LLMQueryView(APIView):
    """
    Handles follow-up questions about previous predictions using the LLM.
    """

    permission_classes = [permissions.IsAuthenticated]  # Apply token authentication

    @extend_schema(
        summary="Ask Follow-up Question about a Prediction",
        description="Sends a user's follow-up question and original prediction context to the LLM for a detailed answer.",
        request=LLMQueryInputSerializer,
        responses={
            200: LLMQueryOutputSerializer,
            400: {"description": "Bad Request: Input validation failed."},
            503: {
                "description": "Service Unavailable: LLM service could not be reached or failed."
            },
        },
        tags=["LLM Interaction"],
    )
    def post(self, request, *args, **kwargs):
        if not LLM_HANDLER_AVAILABLE:
            return Response(
                {"error": "LLM query service is currently unavailable."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        serializer = LLMQueryInputSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        validated_data = serializer.validated_data
        user_question = validated_data["user_question"]
        match_context = validated_data["match_context"]
        original_explanation = validated_data["original_explanation"]
        predicted_winner = validated_data.get("predicted_winner")  # Optional

        # Construct the prompt for Ollama
        prompt_parts = [
            f"Regarding the IPL match prediction for {match_context['team1']} vs {match_context['team2']} "
            f"at {match_context['venue']} ({match_context['city']}) on {match_context['match_date']}:"
        ]
        if predicted_winner:
            prompt_parts.append(f"- The predicted winner was: {predicted_winner}.")
        prompt_parts.append(
            f"- The initial reasoning provided was: '{original_explanation}'."
        )
        prompt_parts.append(
            f"\nPlease answer the following user question based ONLY on this context and general cricket"
            f" knowledge relevant to the context provided: '{user_question}'"
        )
        prompt_parts.append(
            "\nProvide a direct answer. Do not start with phrases like 'Based on the context...' or 'The user's question is...'."
        )

        prompt_text = "\n".join(prompt_parts)
        print(f"Constructed prompt for LLM Query: {prompt_text}")  # For debugging

        # Call the Ollama LLM (ensure query_ollama_llm handles errors and returns string)
        # This part might need to be async if query_ollama_llm is async, using async_to_sync
        # For simplicity, assuming query_ollama_llm is synchronous here.
        # If it's async: llm_answer = async_to_sync(query_ollama_llm)(prompt_text=prompt_text)
        llm_answer = query_ollama_llm(prompt_text=prompt_text)

        if llm_answer.startswith(
            "Error:"
        ):  # Check if the helper returned an error message
            return Response(
                {"error": llm_answer}, status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        output_data = {"answer": llm_answer}
        output_serializer = LLMQueryOutputSerializer(
            data=output_data
        )  # Pass as data= for validation
        if output_serializer.is_valid():  # Good to validate own output structure
            return Response(output_serializer.validated_data, status=status.HTTP_200_OK)
        else:
            # This should ideally not happen if llm_answer is a string
            print(
                f"ERROR: Failed to serialize LLM query output: {output_serializer.errors}"
            )
            return Response(
                {"error": "Failed to format LLM response."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
