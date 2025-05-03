# predictor_api/management/commands/load_match_data.py

import pandas as pd
import os
from collections import defaultdict
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings  # To get BASE_DIR potentially
from django.db import transaction  # For atomic operations
from predictor_api.models import Team, Player, Venue, Match  # Import your models

# --- Constants (adjust filenames if needed) ---
# Assuming 'data/raw' is relative to the Django project's BASE_DIR
RAW_DATA_DIR = os.path.join(settings.BASE_DIR, "data", "raw")  # Use BASE_DIR
MATCH_INFO_FILE = "Match_Info.csv"
BALL_BY_BALL_FILE = "Ball_By_Ball_Match_Data.csv"

# Team Name Normalization (keep consistent with data_processing.py)
TEAM_NAME_MAPPING: dict[str, str] = {
    "Kings XI Punjab": "Punjab Kings",
    "Delhi Daredevils": "Delhi Capitals",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    # Add others as needed
}
# --- End Constants ---


# --- Helper Function (from data_processing.py, adapted) ---
def normalize_team_names(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Applies consistent team names using TEAM_NAME_MAPPING."""
    print("Normalizing team names...")
    for col in columns:
        if col in df.columns:
            # Apply mapping, keep original if not in map
            original_dtype = df[col].dtype
            # Ensure operates on string representation for replacement
            df[col] = (
                df[col]
                .astype(str)
                .replace(TEAM_NAME_MAPPING)
                .astype(original_dtype, errors="ignore")
            )
    return df


# --- End Helper Function ---


class Command(BaseCommand):
    help = "Loads match data from CSV files into the database, calculating features."

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("Starting data loading process..."))

        # --- 1. Load Match Info Data ---
        match_info_path = os.path.join(RAW_DATA_DIR, MATCH_INFO_FILE)
        self.stdout.write(f"Loading Match Info from: {match_info_path}")
        try:
            # Define columns needed from Match Info
            match_cols_needed = [
                "match_number",
                "match_date",
                "team1",
                "team2",
                "toss_winner",
                "toss_decision",
                "venue",
                "city",
                "winner",
                "result",  # Keep 'result' for result_type
                "team1_players",
                "team2_players",  # Needed for Player population
            ]
            df_matches = pd.read_csv(
                match_info_path, usecols=match_cols_needed, parse_dates=["match_date"]
            )
            self.stdout.write(f"Loaded {len(df_matches)} match info rows.")
        except FileNotFoundError:
            raise CommandError(f"Match Info CSV file not found at {match_info_path}")
        except Exception as e:
            raise CommandError(f"Error loading Match Info CSV: {e}")

        # --- Basic Cleaning & Sorting ---
        df_matches.dropna(
            subset=["winner", "team1", "team2", "match_date", "match_number"],
            inplace=True,
        )
        df_matches.sort_values(by=["match_date", "match_number"], inplace=True)
        df_matches.reset_index(drop=True, inplace=True)
        self.stdout.write(
            f"Sorted and cleaned match info. Rows remaining: {len(df_matches)}"
        )

        # --- Normalize Team Names (Match Info) ---
        name_cols_to_normalize = ["team1", "team2", "toss_winner", "winner"]
        df_matches = normalize_team_names(df_matches, name_cols_to_normalize)

        # --- 2. Load Ball-by-Ball Data ---
        bbb_path = os.path.join(RAW_DATA_DIR, BALL_BY_BALL_FILE)
        self.stdout.write(f"Loading Ball-by-Ball data from: {bbb_path}")
        try:
            bbb_cols_needed = ["ID", "BattingTeam", "TotalRun", "IsWicketDelivery"]
            df_bbb = pd.read_csv(bbb_path, usecols=bbb_cols_needed)
            self.stdout.write(f"Loaded {len(df_bbb)} ball-by-ball rows.")

            # Filter based on valid match_numbers from df_matches
            valid_match_ids = set(df_matches["match_number"].unique())
            df_bbb = df_bbb[df_bbb["ID"].isin(valid_match_ids)].copy()
            self.stdout.write(f"Filtered ball-by-ball rows to {len(df_bbb)}.")

            # Normalize team names (BBB)
            df_bbb = normalize_team_names(df_bbb, ["BattingTeam"])

            # Ensure numeric types
            df_bbb["TotalRun"] = pd.to_numeric(
                df_bbb["TotalRun"], errors="coerce"
            ).fillna(0)
            df_bbb["IsWicketDelivery"] = pd.to_numeric(
                df_bbb["IsWicketDelivery"], errors="coerce"
            ).fillna(0)

        except FileNotFoundError:
            raise CommandError(f"Ball-by-Ball CSV file not found at {bbb_path}")
        except Exception as e:
            raise CommandError(f"Error loading Ball-by-Ball CSV: {e}")

        # --- 3. Pre-calculate Match Aggregates (from BBB) ---
        self.stdout.write("Calculating runs/wickets per team per match...")
        grouped = df_bbb.groupby(["ID", "BattingTeam"])
        runs = grouped["TotalRun"].sum().astype(int)
        wickets = grouped["IsWicketDelivery"].sum().astype(int)
        match_aggregates: dict[tuple[int, str], dict[str, int]] = defaultdict(
            lambda: {"score": 0, "wickets": 0}
        )
        for index, score in runs.items():
            match_aggregates[index]["score"] = score
        for index, wkts in wickets.items():
            match_aggregates[index]["wickets"] = wkts
        self.stdout.write(
            f"Calculated aggregates for {len(match_aggregates)} team-match entries."
        )

        # --- 4. Iterate and Populate Database with Features ---
        self.stdout.write(
            "Populating database (Teams, Venues, Players, Matches with features)..."
        )
        # Stores for historical feature calculation
        team_stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"wins": 0, "played": 0}
        )
        h2h_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        prev_match_perf: dict[str, dict[str, int]] = defaultdict(
            lambda: {"score": 150, "wickets": 5}
        )  # Default averages

        match_count = 0
        created_count = 0
        updated_count = 0

        # Use transaction.atomic for potentially faster bulk operations
        try:
            with transaction.atomic():
                total_rows = len(df_matches)
                for index, row in df_matches.iterrows():
                    if index % 100 == 0 and index > 0:
                        self.stdout.write(f"Processing matches: {index}/{total_rows}")

                    match_id_source = int(
                        row["match_number"]
                    )  # Use match_number as source ID
                    team1_name = row["team1"]
                    team2_name = row["team2"]
                    winner_name = row["winner"]  # Already normalized

                    # --- Get or Create related objects ---
                    # Using ignore_case might be useful if CSV has case variations missed by normalization
                    team1_obj, _ = Team.objects.get_or_create(name=team1_name)
                    team2_obj, _ = Team.objects.get_or_create(name=team2_name)
                    winner_obj, _ = (
                        Team.objects.get_or_create(name=winner_name)
                        if pd.notna(winner_name)
                        else (None, False)
                    )
                    toss_winner_obj, _ = (
                        Team.objects.get_or_create(name=row["toss_winner"])
                        if pd.notna(row["toss_winner"])
                        else (None, False)
                    )
                    venue_obj, _ = Venue.objects.get_or_create(
                        name=row["venue"], defaults={"city": row.get("city", "Unknown")}
                    )

                    # --- Populate Players (Keep existing logic from previous Task 2 steps) ---
                    player_list_str = (
                        str(row.get("team1_players", ""))
                        + ","
                        + str(row.get("team2_players", ""))
                    )
                    player_names = set(
                        p.strip()
                        for p in player_list_str.split(",")
                        if p and pd.notna(p) and p.strip()
                    )
                    for name in player_names:
                        if name:  # Ensure not empty after strip
                            Player.objects.get_or_create(name=name)

                    # --- Calculate Features *Before* This Match ---
                    t1_played = team_stats[team1_name]["played"]
                    t1_wins = team_stats[team1_name]["wins"]
                    t2_played = team_stats[team2_name]["played"]
                    t2_wins = team_stats[team2_name]["wins"]
                    t1_win_pct = (t1_wins / t1_played) if t1_played > 0 else 0.0
                    t2_win_pct = (t2_wins / t2_played) if t2_played > 0 else 0.0

                    h2h_played = (
                        h2h_stats[team1_name][team2_name]
                        + h2h_stats[team2_name][team1_name]
                    )
                    t1_h2h_wins = h2h_stats[team1_name][team2_name]
                    t1_h2h_win_pct = (
                        (t1_h2h_wins / h2h_played) if h2h_played > 0 else 0.5
                    )  # Default for no history

                    t1_prev_perf = prev_match_perf[team1_name]
                    t2_prev_perf = prev_match_perf[team2_name]

                    # --- Prepare data for Match model ---
                    match_defaults = {
                        "season": int(
                            pd.to_datetime(row["match_date"]).year
                        ),  # Extract year for season
                        "date": row["match_date"].date(),
                        "match_number": (
                            int(row["match_number"])
                            if pd.notna(row["match_number"])
                            else None
                        ),
                        "team1": team1_obj,
                        "team2": team2_obj,
                        "venue": venue_obj,
                        "toss_winner": toss_winner_obj,
                        "toss_decision": (
                            row["toss_decision"]
                            if pd.notna(row["toss_decision"])
                            else None
                        ),
                        "winner": winner_obj,
                        "result_type": (
                            row["result"] if pd.notna(row["result"]) else None
                        ),  # Use 'result' col for result_type
                        "result_margin": None,  # Keep as None - calculation logic not implemented here
                        # Add calculated features
                        "team1_win_pct": t1_win_pct,
                        "team2_win_pct": t2_win_pct,
                        "team1_h2h_win_pct": t1_h2h_win_pct,
                        "team1_prev_score": t1_prev_perf["score"],
                        "team1_prev_wkts": t1_prev_perf["wickets"],
                        "team2_prev_score": t2_prev_perf["score"],
                        "team2_prev_wkts": t2_prev_perf["wickets"],
                    }

                    # --- ADD DEBUGGING HERE ---
                    if index >= (total_rows - 5):  # Log for the last 5 matches
                        self.stdout.write(
                            f"\nDEBUG INFO FOR INDEX {index}:"
                        )  # Log index first
                        self.stdout.write(
                            f"  Match ID from row: {row['match_number']}"
                        )  # Log the actual match_number from the row
                        self.stdout.write(
                            f"  Raw Row Data (subset): team1={row['team1']}, team2={row['team2']}, winner={row['winner']}"
                        )
                        self.stdout.write(
                            f"  Calculated Win %: t1={t1_win_pct:.4f}, t2={t2_win_pct:.4f}, h2h={t1_h2h_win_pct:.4f}"
                        )
                        self.stdout.write(
                            f"  Retrieved Prev Perf: t1={t1_prev_perf}, t2={t2_prev_perf}"
                        )
                        # self.stdout.write(f"  Saving Defaults: {match_defaults}") # Optional
                    # --- END DEBUGGING ---

                    # --- Create or Update Match record ---
                    match_obj, created = Match.objects.update_or_create(
                        match_id_source=match_id_source, defaults=match_defaults
                    )
                    match_count += 1
                    if created:
                        created_count += 1
                    else:
                        updated_count += 1

                    # --- Update historical stats stores *after* processing match outcome ---
                    team_stats[team1_name]["played"] += 1
                    team_stats[team2_name]["played"] += 1
                    if winner_name == team1_name:
                        team_stats[team1_name]["wins"] += 1
                        h2h_stats[team1_name][team2_name] += 1
                    elif winner_name == team2_name:
                        team_stats[team2_name]["wins"] += 1
                        h2h_stats[team2_name][team1_name] += 1

                    # Update previous performance store
                    t1_actual_perf = match_aggregates.get(
                        (match_id_source, team1_name), t1_prev_perf
                    )  # Use previous if not found in aggregates
                    t2_actual_perf = match_aggregates.get(
                        (match_id_source, team2_name), t2_prev_perf
                    )  # Use previous if not found
                    prev_match_perf[team1_name] = t1_actual_perf
                    prev_match_perf[team2_name] = t2_actual_perf

        except Exception as e:
            raise CommandError(f"Error during database population: {e}")

        self.stdout.write(f"Finished processing {match_count} matches.")
        self.stdout.write(f"Created: {created_count}, Updated: {updated_count}")
        self.stdout.write(self.style.SUCCESS("Successfully loaded and processed data."))
