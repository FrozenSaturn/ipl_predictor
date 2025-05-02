# predictor_api/models.py

from django.db import models

# from django.utils import timezone  # Good practice for datetime fields


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
    # Add other relevant team fields if needed, e.g., home_ground, logo_url

    def __str__(self):
        return self.name

    class Meta:
        ordering = ["name"]  # Keep teams alphabetically sorted by default


class Player(models.Model):
    """
    Represents an individual player identified uniquely by name.
    We assume player names are unique across the dataset for simplicity initially.
    """

    # Using name as the primary way to identify players from the CSVs
    name = models.CharField(
        max_length=200, unique=True, help_text="Player's full name (must be unique)"
    )
    # You could add more fields later if data is available:
    # role = models.CharField(
    # max_length=50, blank=True, null=True, help_text="e.g.,
    # Batsman, Bowler, All-rounder"
    # )
    # date_of_birth = models.DateField(blank=True, null=True)
    # nationality = models.CharField(max_length=100, blank=True, null=True)

    # Timestamps
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
    country = models.CharField(
        max_length=100, default="India"
    )  # Assuming IPL context primarily

    def __str__(self):
        return f"{self.name}, {self.city}"

    class Meta:
        ordering = ["name"]


class Match(models.Model):
    """
    Represents a single IPL match, linking teams, venue, and results.
    """

    match_id_source = models.IntegerField(
        unique=True,
        help_text="Unique ID from the source data (e.g., Match_Info.csv ID)",
    )  # Important for linking back
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

    # Fields from Ball_By_Ball can be aggregated or linked if needed later,
    # but start with Match_Info basics.
    # Consider adding umpire names, player of the match etc. if required.

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.team1} vs {self.team2} ({self.date})"

    class Meta:
        ordering = ["-date", "-match_number"]  # Show most recent matches first
        verbose_name_plural = "Matches"  # Correct pluralization in Django admin
