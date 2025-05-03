# predictor_api/models.py

from django.db import models

# from django.utils import timezone # Not currently used, keep commented


class Team(models.Model):
    """
    Represents a participating IPL team.
    """

    name = models.CharField(max_length=100, unique=True, help_text="Official team name")
    short_name = models.CharField(
        max_length=10,
        unique=True,
        blank=True,
        null=True,
        help_text="Common abbreviation (e.g., CSK, MI)",
    )

    def __str__(self):
        return self.name

    class Meta:
        ordering = ["name"]


class Player(models.Model):
    """
    Represents an individual player identified uniquely by name.
    """

    name = models.CharField(
        max_length=200, unique=True, help_text="Player's full name (must be unique)"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

    class Meta:
        ordering = ["name"]
        verbose_name_plural = "Players"


class Venue(models.Model):
    """
    Represents a cricket stadium/venue.
    """

    name = models.CharField(max_length=200, unique=True)
    city = models.CharField(max_length=100)
    country = models.CharField(max_length=100, default="India")

    def __str__(self):
        return f"{self.name}, {self.city}"

    class Meta:
        ordering = ["name"]


class Match(models.Model):
    """
    Represents a single IPL match, linking teams, venue, results,
    and pre-calculated features used for prediction.
    """

    # Core Match Info
    match_id_source = models.IntegerField(
        unique=True,
        help_text="Unique ID from the source data (e.g., Match_Info.csv match_number)",  # Updated help text
    )
    season = models.IntegerField(help_text="IPL Season year (e.g., 2023)")
    date = models.DateField(help_text="Date the match was played")
    match_number = models.IntegerField(
        blank=True, null=True, help_text="Match number within the season/tournament"
    )
    team1 = models.ForeignKey(
        Team, related_name="home_matches", on_delete=models.CASCADE
    )
    team2 = models.ForeignKey(
        Team, related_name="away_matches", on_delete=models.CASCADE
    )
    venue = models.ForeignKey(Venue, on_delete=models.CASCADE)
    toss_winner = models.ForeignKey(
        Team, related_name="toss_wins", on_delete=models.SET_NULL, null=True, blank=True
    )
    toss_decision = models.CharField(
        max_length=10,
        choices=[("bat", "Bat"), ("field", "Field")],
        null=True,
        blank=True,
    )
    winner = models.ForeignKey(
        Team,
        related_name="match_wins",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )
    result_type = models.CharField(
        max_length=20,
        null=True,
        blank=True,
        help_text="e.g., 'runs', 'wickets', 'tie', 'no result'",
    )
    result_margin = models.IntegerField(
        null=True, blank=True, help_text="Margin of victory (runs or wickets)"
    )

    # --- Pre-calculated Engineered Features ---
    # These will be calculated and populated by a management command based on historical data *before* this match occurred.
    team1_win_pct = models.FloatField(
        null=True,
        blank=True,
        help_text="Team 1's historical overall win % before this match",
    )
    team2_win_pct = models.FloatField(
        null=True,
        blank=True,
        help_text="Team 2's historical overall win % before this match",
    )
    team1_h2h_win_pct = models.FloatField(
        null=True,
        blank=True,
        help_text="Team 1's historical H2H win % against Team 2 before this match",
    )
    team1_prev_score = models.IntegerField(
        null=True,
        blank=True,
        help_text="Team 1's score in their immediately previous match",
    )
    team1_prev_wkts = models.IntegerField(
        null=True,
        blank=True,
        help_text="Team 1's wickets lost in their immediately previous match",
    )
    team2_prev_score = models.IntegerField(
        null=True,
        blank=True,
        help_text="Team 2's score in their immediately previous match",
    )
    team2_prev_wkts = models.IntegerField(
        null=True,
        blank=True,
        help_text="Team 2's wickets lost in their immediately previous match",
    )
    # --- Add more feature fields here as they are developed ---

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.team1} vs {self.team2} ({self.date})"

    class Meta:
        ordering = ["-date", "-match_number"]
        verbose_name_plural = "Matches"
