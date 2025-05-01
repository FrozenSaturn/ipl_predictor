# predictor_api/serializers.py

from rest_framework import serializers


class MatchInputSerializer(serializers.Serializer):
    """
    Validates the input data for the prediction request.
    Mirrors the structure of FastAPI's MatchInput Pydantic model.
    """

    # Use CharField for string inputs. Equivalent to 'str' type hint.
    team1 = serializers.CharField(required=True, max_length=100)
    team2 = serializers.CharField(required=True, max_length=100)
    toss_winner = serializers.CharField(required=True, max_length=100)
    # Use ChoiceField to restrict toss_decision input
    toss_decision = serializers.ChoiceField(choices=["field", "bat"], required=True)
    venue = serializers.CharField(required=True, max_length=100)
    city = serializers.CharField(
        required=True, max_length=100, allow_blank=False
    )  # Explicitly disallow empty string if needed

    # Note: DRF serializers provide built-in validation based on field types and args.
    # Custom validation methods can be added if needed (e.g., validate_<field_name>).


class PredictionOutputSerializer(serializers.Serializer):
    """
    Defines the structure for the prediction response.
    Mirrors FastAPI's PredictionOutput Pydantic model.
    """

    predicted_winner = serializers.CharField(
        read_only=True
    )  # read_only as it's output only
    explanation = serializers.CharField(
        read_only=True, allow_null=True
    )  # Allow null if LLM fails
