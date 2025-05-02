# predictor_api/serializers.py

from rest_framework import serializers
from .models import Team, Venue, Match, Player

# =============================================================================
# Serializers for Database Models (Teams, Venues, Matches)
# =============================================================================


class PlayerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Player
        fields = ["id", "name", "created_at", "updated_at"]
        read_only_fields = fields  # Keep read-only for now


class TeamSerializer(serializers.ModelSerializer):
    class Meta:
        model = Team
        fields = ["id", "name", "short_name"]


class VenueSerializer(serializers.ModelSerializer):
    class Meta:
        model = Venue
        fields = ["id", "name", "city", "country"]


class MatchSerializer(serializers.ModelSerializer):
    team1 = serializers.StringRelatedField()
    team2 = serializers.StringRelatedField()
    venue = serializers.StringRelatedField()
    toss_winner = serializers.StringRelatedField()
    winner = serializers.StringRelatedField()

    class Meta:
        model = Match
        fields = [
            "id",
            "match_id_source",
            "season",
            "date",
            "match_number",
            "team1",
            "team2",
            "venue",
            "toss_winner",
            "toss_decision",
            "winner",
            "result_type",
            "result_margin",
            "created_at",
            "updated_at",
        ]
        read_only_fields = fields


# =============================================================================
# Serializers for Prediction Endpoint (Input and Output)
# =============================================================================


class MatchInputSerializer(serializers.Serializer):
    """
    Validates the input data required for making a prediction.
    Adjust fields based on what your 'predict_winner' function needs.
    """

    team1 = serializers.CharField(max_length=100)
    team2 = serializers.CharField(max_length=100)
    venue = serializers.CharField(max_length=200)
    city = serializers.CharField(max_length=100)
    toss_winner = serializers.CharField(max_length=100)
    toss_decision = serializers.ChoiceField(choices=["bat", "field"])
    # Add any other fields required by your ML model/predictor function
    # e.g., season = serializers.IntegerField(required=False)

    def validate_team1(self, value):
        """Example validation: Check if team exists in DB (optional but good)."""
        # Note: Requires querying the DB, consider performance implications.
        # You might skip this validation here and handle invalid teams in predictor.
        # if not Team.objects.filter(name=value).exists():
        #     raise serializers.ValidationError(f"Team '{value}' not found.")
        return value  # Remember to return the value

    def validate_team2(self, value):
        """Example validation for team2."""
        # if not Team.objects.filter(name=value).exists():
        #     raise serializers.ValidationError(f"Team '{value}' not found.")
        return value

    # Add similar validation for venue if desired


class PredictionOutputSerializer(serializers.Serializer):
    """
    Formats the output data returned by the prediction endpoint.
    Adjust fields based on what your 'predict_winner' function returns.
    """

    prediction = serializers.CharField(max_length=100)
    explanation = serializers.CharField()
    # Add other fields returned by your predictor function
    # e.g., confidence_score = serializers.FloatField(required=False)
