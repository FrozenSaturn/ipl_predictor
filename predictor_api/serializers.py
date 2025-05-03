# predictor_api/serializers.py

from rest_framework import serializers
from .models import Team, Venue, Match, Player, PlayerMatchPerformance

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
    match_date = serializers.DateField(input_formats=["%Y-%m-%d"], required=True)
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
    predicted_winner = serializers.CharField(
        source="prediction", read_only=True, allow_null=True
    )  # <-- Added source='prediction'
    confidence = serializers.FloatField(read_only=True, allow_null=True)
    explanation = serializers.CharField(read_only=True, allow_null=True)


# =============================================================================
# Serializer for Player Performance Data
# =============================================================================
class PlayerMatchPerformanceSerializer(serializers.ModelSerializer):
    """
    Serializer for PlayerMatchPerformance model, including calculated fields.
    """

    # Include related match info for context
    match_date = serializers.DateField(source="match.date", read_only=True)
    match_venue = serializers.CharField(source="match.venue.name", read_only=True)
    # Optionally add opponent? Requires fetching related match object again or different query optimization
    # opponent = serializers.SerializerMethodField()

    # Calculated fields
    strike_rate = serializers.SerializerMethodField()
    economy_rate = serializers.SerializerMethodField()

    class Meta:
        model = PlayerMatchPerformance
        fields = [
            # Match Info
            "match",  # Keep match ID for potential linking
            "match_date",
            "match_venue",
            # Batting
            "runs_scored",
            "balls_faced",
            "fours_hit",
            "sixes_hit",
            "dismissal_kind",
            "strike_rate",
            # Bowling
            "balls_bowled",
            "runs_conceded",
            "wickets_taken",
            "dots_bowled",
            "fours_conceded",
            "sixes_conceded",
            "economy_rate",
            # Add 'id' of the performance record itself if needed
            # 'id',
        ]
        read_only_fields = fields  # Mark all as read-only

    def get_strike_rate(self, obj):
        if obj.balls_faced > 0:
            return round((obj.runs_scored / obj.balls_faced) * 100, 2)
        return None  # Or 0.0? Can decide based on frontend needs

    def get_economy_rate(self, obj):
        if obj.balls_bowled > 0:
            # Calculate based on balls bowled for accuracy
            # Avoid division by zero if somehow 0 balls but runs conceded (shouldn't happen)
            overs_precise = obj.balls_bowled / 6.0
            if overs_precise > 0:
                return round(obj.runs_conceded / overs_precise, 2)
        return None  # Or indicate infinite/NA based on frontend needs

    # Optional: Method to get opponent team name
    # def get_opponent(self, obj):
    #     # Assumes obj.match is available
    #     requesting_player_team = # Need logic to determine which team the player was on for this match
    #     if obj.match.team1 == requesting_player_team:
    #         return obj.match.team2.name
    #     elif obj.match.team2 == requesting_player_team:
    #         return obj.match.team1.name
    #     return "Unknown"
