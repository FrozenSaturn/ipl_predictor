# build_features.py

import os
import django
import sys
import pandas as pd
from collections import defaultdict
from datetime import date
from typing import Dict, Optional, List, Set, Any
import numpy as np
import logging

# --- Django Environment Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logger.info("Setting up Django environment...")
# Assumes build_features.py is in a subdirectory of the project root (e.g., scripts/ or src/ipl_predictor/features/)
# Adjust path depth ('..') if necessary to reach the directory containing manage.py
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Example: If script is in scripts/
# Or uncomment and set absolute path:
project_root = "/Users/aryabhattacharjee/Desktop/Major Projects/ipl-predictor-project/"

if project_root not in sys.path:
    sys.path.append(project_root)

# Replace 'ipl_django_project.settings' if your settings file is elsewhere
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ipl_django_project.settings")

try:
    django.setup()
    logger.info("Django environment setup successfully.")
    from predictor_api.models import Match, Team, PlayerMatchPerformance
    from django.db.models import Q, Sum

    DJANGO_LOADED = True
except Exception as e:
    logger.error(f"Failed to setup Django environment: {e}", exc_info=True)
    sys.exit("Exiting: Cannot proceed without Django environment.")
# --- End Django Setup ---

# --- Configuration ---
# Assumes data directories are relative to the project root
BASE_DIR = project_root  # Use project root determined above
RAW_DATA_DIR: str = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR: str = os.path.join(BASE_DIR, "data", "processed")
MATCH_INFO_FILE: str = "Match_Info.csv"
# Saving new features to a different file to avoid overwriting old ones immediately
OUTPUT_FEATURE_FILE: str = "features_v2_with_rolling.csv"

NUM_RECENT_MATCHES = 5  # Window for rolling stats

# Default values for filling NaNs or handling initial matches
DEFAULT_WIN_PCT = 0.0
DEFAULT_H2H_WIN_PCT = 0.5
DEFAULT_BATTING_SR = 125.0  # Adjusted default
DEFAULT_BOWLING_ECON = 8.50  # Adjusted default
DEFAULT_AVG_RUNS = 25.0
DEFAULT_AVG_WICKETS = 1.0

# Team Name Normalization (Use the same mapping as in data loading)
TEAM_NAME_MAPPING: dict[str, str] = {
    "Kings XI Punjab": "Punjab Kings",
    "Delhi Daredevils": "Delhi Capitals",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
    # Ensure this includes ALL necessary mappings used during data loading
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
}
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
# --- End Configuration ---


def normalize_team_names(series: pd.Series) -> pd.Series:
    """Applies TEAM_NAME_MAPPING to a pandas Series."""
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        # Fill NA before replacing
        return series.fillna("Unknown").replace(TEAM_NAME_MAPPING)
    return series


# --- Helper Functions for Rolling Stats ---


def get_last_played_players(team_id: int, current_date: date) -> Set[int]:
    """Identifies player IDs from the team's most recent match before current_date."""
    previous_match = (
        Match.objects.filter(
            Q(team1_id=team_id) | Q(team2_id=team_id), date__lt=current_date
        )
        .order_by("-date", "-match_number")
        .first()
    )  # Added match_number for tie-breaking on date

    if not previous_match:
        # logger.debug(f"No previous match for team_id {team_id} before {current_date}")
        return set()

    player_ids = set(
        PlayerMatchPerformance.objects.filter(match=previous_match)
        .values_list("player_id", flat=True)
        .distinct()
    )  # Added distinct

    # logger.debug(f"Found {len(player_ids)} players for team_id {team_id} from match {previous_match.match_id_source} on {previous_match.date}")
    return player_ids


def get_player_rolling_stats(
    player_id: int, current_date: date, num_matches: int = NUM_RECENT_MATCHES
) -> Optional[Dict[str, Any]]:
    """Calculates rolling stats for a player's last N matches before current_date."""
    performances = PlayerMatchPerformance.objects.filter(
        player_id=player_id, match__date__lt=current_date
    ).order_by("-match__date")[:num_matches]

    count = performances.count()
    if count == 0:
        return None

    stats = performances.aggregate(
        total_runs=Sum("runs_scored", default=0),
        total_balls_faced=Sum("balls_faced", default=0),
        total_wickets=Sum("wickets_taken", default=0),
        total_balls_bowled=Sum("balls_bowled", default=0),
        total_runs_conceded=Sum("runs_conceded", default=0),
    )

    # Use Decimal for potentially better precision control if needed later
    total_runs = stats.get("total_runs", 0)
    total_balls_faced = stats.get("total_balls_faced", 0)
    total_wickets = stats.get("total_wickets", 0)
    total_balls_bowled = stats.get("total_balls_bowled", 0)
    total_runs_conceded = stats.get("total_runs_conceded", 0)

    # Calculate averages
    avg_runs = total_runs / count
    avg_wickets = total_wickets / count

    # Calculate aggregate rates safely
    strike_rate = (
        (total_runs * 100.0 / total_balls_faced) if total_balls_faced > 0 else None
    )
    economy_rate = (
        (total_runs_conceded * 6.0 / total_balls_bowled)
        if total_balls_bowled > 0
        else None
    )

    return {
        "avg_runs": avg_runs,
        "strike_rate": strike_rate,
        "avg_wickets": avg_wickets,
        "economy_rate": economy_rate,
        "matches_considered": count,
    }


def aggregate_team_stats(
    player_stats_list: List[Dict[str, Any]], team_prefix: str
) -> Dict[str, float]:
    """Aggregates list of player stats dicts into team-level average features."""
    team_features = {}
    num_players_with_stats = len(player_stats_list)

    # Define keys and their default values
    agg_map = {
        "avg_runs": DEFAULT_AVG_RUNS,
        "strike_rate": DEFAULT_BATTING_SR,
        "avg_wickets": DEFAULT_AVG_WICKETS,
        "economy_rate": DEFAULT_BOWLING_ECON,
    }

    if num_players_with_stats == 0:
        for key, default_val in agg_map.items():
            team_features[f"{team_prefix}_{key}_L{NUM_RECENT_MATCHES}"] = default_val
        return team_features

    # Calculate means, filtering out None values from individual calcs
    for key, default_val in agg_map.items():
        valid_values = [
            s[key]
            for s in player_stats_list
            if s is not None and s.get(key) is not None
        ]
        mean_val = np.mean(valid_values) if valid_values else default_val
        team_features[f"{team_prefix}_{key}_L{NUM_RECENT_MATCHES}"] = float(
            mean_val
        )  # Ensure float

    return team_features


# --- End Helper Functions ---


def calculate_and_save_features():
    """Loads data, calculates all features chronologically including player form, saves to CSV."""
    logger.info("Loading base match info...")
    match_info_path = os.path.join(RAW_DATA_DIR, MATCH_INFO_FILE)
    try:
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
        ]
        df_matches = pd.read_csv(
            match_info_path, usecols=match_cols_needed, parse_dates=["match_date"]
        )
    except Exception as e:
        logger.error(f"Error loading {MATCH_INFO_FILE}: {e}")
        return

    df_matches.dropna(
        subset=["winner", "team1", "team2", "match_date", "match_number"], inplace=True
    )
    df_matches.sort_values(by=["match_date", "match_number"], inplace=True)
    df_matches.reset_index(drop=True, inplace=True)
    logger.info(f"Loaded and sorted {len(df_matches)} matches.")

    logger.info("Normalizing team names...")
    for col in ["team1", "team2", "toss_winner", "winner"]:
        df_matches[col] = normalize_team_names(df_matches[col])
    df_matches["city"] = df_matches["city"].fillna("Unknown")

    # Pre-fetch Team objects for efficiency
    all_team_names = set(df_matches["team1"]) | set(df_matches["team2"])
    team_objs = {t.name: t for t in Team.objects.filter(name__in=all_team_names)}
    logger.info(f"Fetched {len(team_objs)} team objects from DB.")

    # --- Feature Calculation Loop ---
    logger.info("Starting feature calculation loop (Including Player Form)...")
    team_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"wins": 0, "played": 0}
    )
    h2h_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    all_features_list = []
    total_rows = len(df_matches)

    for index, row in df_matches.iterrows():
        if (index + 1) % 100 == 0:
            logger.info(f"  Processing features for match {index + 1}/{total_rows}...")

        match_id_source = int(row["match_number"])
        team1_name = row["team1"]
        team2_name = row["team2"]
        winner_name = row["winner"]
        current_date = row["match_date"].date()

        current_features = {"match_number": match_id_source, "winner": winner_name}
        for col in ["team1", "team2", "toss_winner", "toss_decision", "venue", "city"]:
            current_features[col] = row[col]

        try:
            team1_obj = team_objs.get(team1_name)
            team2_obj = team_objs.get(team2_name)
            if not team1_obj or not team2_obj:
                logger.warning(
                    f"Skipping match {match_id_source}: Team object not found in DB for {team1_name} or {team2_name}"
                )
                continue  # Skip row if teams not found

            # Calculate Historical Win %
            t1_played = team_stats[team1_name]["played"]
            t1_wins = team_stats[team1_name]["wins"]
            t2_played = team_stats[team2_name]["played"]
            t2_wins = team_stats[team2_name]["wins"]
            current_features["team1_win_pct"] = (
                (t1_wins / t1_played) if t1_played > 0 else DEFAULT_WIN_PCT
            )
            current_features["team2_win_pct"] = (
                (t2_wins / t2_played) if t2_played > 0 else DEFAULT_WIN_PCT
            )
            h2h_played = h2h_stats[team1_name].get(team2_name, 0) + h2h_stats[
                team2_name
            ].get(team1_name, 0)
            t1_h2h_wins = h2h_stats[team1_name].get(team2_name, 0)
            current_features["team1_h2h_win_pct"] = (
                (t1_h2h_wins / h2h_played) if h2h_played > 0 else DEFAULT_H2H_WIN_PCT
            )

            # Calculate Recent Player Form
            team1_players_ids = get_last_played_players(team1_obj.id, current_date)
            team2_players_ids = get_last_played_players(team2_obj.id, current_date)

            team1_rolling_stats = [
                get_player_rolling_stats(pid, current_date) for pid in team1_players_ids
            ]
            team1_rolling_stats = [
                s for s in team1_rolling_stats if s is not None
            ]  # Filter out None results

            team2_rolling_stats = [
                get_player_rolling_stats(pid, current_date) for pid in team2_players_ids
            ]
            team2_rolling_stats = [
                s for s in team2_rolling_stats if s is not None
            ]  # Filter out None results

            # Aggregate Team Form Features
            current_features.update(aggregate_team_stats(team1_rolling_stats, "team1"))
            current_features.update(aggregate_team_stats(team2_rolling_stats, "team2"))

            all_features_list.append(current_features)

            # Update history trackers for NEXT iteration
            team_stats[team1_name]["played"] += 1
            team_stats[team2_name]["played"] += 1
            if winner_name == team1_name:
                team_stats[team1_name]["wins"] += 1
                h2h_stats[team1_name][team2_name] = (
                    h2h_stats[team1_name].get(team2_name, 0) + 1
                )
            elif winner_name == team2_name:
                team_stats[team2_name]["wins"] += 1
                h2h_stats[team2_name][team1_name] = (
                    h2h_stats[team2_name].get(team1_name, 0) + 1
                )

        except Exception as e_row:
            logger.error(
                f"ERROR processing row {index} (Match #{match_id_source}): {e_row}",
                exc_info=True,
            )
            # Optionally add placeholder row or skip
            # all_features_list.append({"match_number": match_id_source})

    logger.info(
        f"Finished feature calculation loop for {len(all_features_list)} matches."
    )

    if not all_features_list:
        logger.error("No features were generated.")
        return

    df_final_features = pd.DataFrame(all_features_list)

    # Define final columns order (ensure these match the feature names generated)
    final_cols_order = [
        "match_number",
        "winner",  # Target
        "team1",
        "team2",
        "toss_winner",
        "toss_decision",
        "venue",
        "city",  # Categorical
        "team1_win_pct",
        "team2_win_pct",
        "team1_h2h_win_pct",  # Historical %
        # New Rolling Form Features (adjust names if changed in helpers)
        f"team1_avg_runs_L{NUM_RECENT_MATCHES}",
        f"team1_strike_rate_L{NUM_RECENT_MATCHES}",
        f"team1_avg_wickets_L{NUM_RECENT_MATCHES}",
        f"team1_economy_rate_L{NUM_RECENT_MATCHES}",
        f"team2_avg_runs_L{NUM_RECENT_MATCHES}",
        f"team2_strike_rate_L{NUM_RECENT_MATCHES}",
        f"team2_avg_wickets_L{NUM_RECENT_MATCHES}",
        f"team2_economy_rate_L{NUM_RECENT_MATCHES}",
    ]

    # Reorder and select columns, adding missing ones with defaults if necessary
    for col in final_cols_order:
        if col not in df_final_features.columns:
            logger.warning(
                f"Column '{col}' was missing after calculation, adding default."
            )
            # Determine default (crude mapping based on name)
            default_val = 0.0
            if "win_pct" in col:
                default_val = (
                    DEFAULT_WIN_PCT if "h2h" not in col else DEFAULT_H2H_WIN_PCT
                )
            elif "sr" in col:
                default_val = DEFAULT_BATTING_SR
            elif "econ" in col:
                default_val = DEFAULT_BOWLING_ECON
            elif "runs" in col:
                default_val = DEFAULT_AVG_RUNS
            elif "wickets" in col:
                default_val = DEFAULT_AVG_WICKETS
            df_final_features[col] = default_val

    df_final_features = df_final_features[final_cols_order]

    # Final check for NaNs and fill with defaults (should mostly be handled by aggregate_team_stats)
    nan_counts = df_final_features.isnull().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if not cols_with_nan.empty:
        logger.warning(
            f"NaN values found after calculation in columns: {list(cols_with_nan.index)}. Filling with defaults."
        )
        # Refined fillna based on type
        default_map_final = {
            f"team1_avg_runs_L{NUM_RECENT_MATCHES}": DEFAULT_AVG_RUNS,
            f"team1_strike_rate_L{NUM_RECENT_MATCHES}": DEFAULT_BATTING_SR,
            f"team1_avg_wickets_L{NUM_RECENT_MATCHES}": DEFAULT_AVG_WICKETS,
            f"team1_economy_rate_L{NUM_RECENT_MATCHES}": DEFAULT_BOWLING_ECON,
            f"team2_avg_runs_L{NUM_RECENT_MATCHES}": DEFAULT_AVG_RUNS,
            f"team2_strike_rate_L{NUM_RECENT_MATCHES}": DEFAULT_BATTING_SR,
            f"team2_avg_wickets_L{NUM_RECENT_MATCHES}": DEFAULT_AVG_WICKETS,
            f"team2_economy_rate_L{NUM_RECENT_MATCHES}": DEFAULT_BOWLING_ECON,
            "team1_win_pct": DEFAULT_WIN_PCT,
            "team2_win_pct": DEFAULT_WIN_PCT,
            "team1_h2h_win_pct": DEFAULT_H2H_WIN_PCT,
        }
        df_final_features.fillna(value=default_map_final, inplace=True)

    output_path = os.path.join(PROCESSED_DATA_DIR, OUTPUT_FEATURE_FILE)
    logger.info(
        f"Saving final features to: {output_path} (Shape: {df_final_features.shape})"
    )
    try:
        df_final_features.to_csv(output_path, index=False)
        logger.info("Features saved successfully.")
    except Exception as e:
        logger.error(f"ERROR saving features to CSV: {e}", exc_info=True)


if __name__ == "__main__":
    logger.info("Starting Feature Build Process (Adding Rolling Player Form)...")
    if DJANGO_LOADED:
        calculate_and_save_features()
    else:
        logger.error("Exiting: Django environment setup failed.")
    logger.info("Feature Build Process Finished.")
