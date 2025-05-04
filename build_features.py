# build_features.py

import pandas as pd
import os
import sys
from collections import defaultdict
from datetime import date
from typing import Dict, Optional

# --- Django Environment Setup ---
print("Setting up Django environment...")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ipl_django_project.settings")
try:
    import django

    django.setup()
    print("Django environment setup successfully.")
    from predictor_api.models import Match, Team, PlayerMatchPerformance
    from django.db.models import Q, Sum

    DJANGO_LOADED = True
except Exception as e:
    print(f"ERROR: Failed to setup Django environment: {e}")
    sys.exit("Exiting: Cannot proceed without Django environment.")
# --- End Django Setup ---

# --- Configuration ---
RAW_DATA_DIR: str = os.path.join(os.getcwd(), "data", "raw")
PROCESSED_DATA_DIR: str = os.path.join(os.getcwd(), "data", "processed")
MATCH_INFO_FILE: str = "Match_Info.csv"
BALL_BY_BALL_FILE: str = "Ball_By_Ball_Match_Data.csv"  # Needed for score calculation
OUTPUT_FEATURE_FILE: str = (
    "features_v3_with_scores.csv"  # New filename reflecting scores
)

N_RECENT_MATCHES_FORM: int = 5

DEFAULT_WIN_PCT = 0.0
DEFAULT_H2H_WIN_PCT = 0.5
DEFAULT_SCORE = 150
DEFAULT_WKTS = 5
DEFAULT_BATTING_SR = 130.0
DEFAULT_BOWLING_ECON = 8.5

TEAM_NAME_MAPPING: dict[str, str] = {
    "Kings XI Punjab": "Punjab Kings",
    "Delhi Daredevils": "Delhi Capitals",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
}
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
# --- End Configuration ---


def normalize_team_names(series: pd.Series) -> pd.Series:
    """Applies TEAM_NAME_MAPPING to a pandas Series."""
    original_dtype = series.dtype
    return (
        series.astype(str)
        .replace(TEAM_NAME_MAPPING)
        .astype(original_dtype, errors="ignore")
    )


# --- Helper Function to Get Aggregated Team Form (Unchanged from previous version) ---
def get_aggregated_team_form(
    team_obj: Optional[Team], prediction_date: date
) -> Dict[str, Optional[float]]:
    """Queries PlayerMatchPerformance for recent stats and aggregates them."""
    # ... (Keep the existing function body from response #107 / #109) ...
    team_form: Dict[str, Optional[float]] = {
        "avg_recent_sr": None,
        "avg_recent_econ": None,
    }
    if not DJANGO_LOADED or not team_obj:
        return team_form
    try:
        last_match_for_team = (
            Match.objects.filter(
                Q(team1=team_obj) | Q(team2=team_obj), date__lt=prediction_date
            )
            .order_by("-date", "-match_number")
            .first()
        )
        if not last_match_for_team:
            return team_form
        player_ids_last_match = list(
            PlayerMatchPerformance.objects.filter(match_id=last_match_for_team.id)
            .values_list("player_id", flat=True)
            .distinct()
        )
        if not player_ids_last_match:
            return team_form
        recent_performances = PlayerMatchPerformance.objects.filter(
            player_id__in=player_ids_last_match, match__date__lt=prediction_date
        )
        # Batting SR Aggregate
        calculated_sr: Optional[float] = None
        bat_agg = recent_performances.filter(balls_faced__gt=0).aggregate(
            total_runs=Sum("runs_scored"), total_balls=Sum("balls_faced")
        )
        if (
            bat_agg.get("total_balls") is not None
            and bat_agg["total_balls"] > 0
            and bat_agg.get("total_runs") is not None
        ):
            calculated_sr = float(
                (bat_agg["total_runs"] * 100.0) / bat_agg["total_balls"]
            )
        team_form["avg_recent_sr"] = calculated_sr
        # Bowling Economy Aggregate
        calculated_econ: Optional[float] = None
        bowl_agg = recent_performances.filter(balls_bowled__gt=0).aggregate(
            total_runs_conceded=Sum("runs_conceded"),
            total_balls_bowled=Sum("balls_bowled"),
        )
        if (
            bowl_agg.get("total_balls_bowled") is not None
            and bowl_agg["total_balls_bowled"] > 0
            and bowl_agg.get("total_runs_conceded") is not None
        ):
            overs = bowl_agg["total_balls_bowled"] / 6.0
            if overs > 0:
                calculated_econ = float(bowl_agg["total_runs_conceded"] / overs)
        team_form["avg_recent_econ"] = calculated_econ
    except Exception as e_form:
        print(
            f"Warning: Error calculating form stats for team {team_obj.name if team_obj else 'N/A'}: {e_form}"
        )
    return team_form


# --- End Helper Function ---


def calculate_and_save_features() -> None:
    """Loads data, calculates all features including first innings score, saves to CSV."""
    print("Loading base match info...")
    match_info_path = os.path.join(RAW_DATA_DIR, MATCH_INFO_FILE)
    try:  # ... (Load Match Info CSV as before) ...
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
        print(f"Error loading {MATCH_INFO_FILE}: {e}")
        return

    # Cleaning & Sorting (as before)
    df_matches.dropna(
        subset=["winner", "team1", "team2", "match_date", "match_number"], inplace=True
    )
    df_matches.sort_values(by=["match_date", "match_number"], inplace=True)
    df_matches.reset_index(drop=True, inplace=True)
    print(f"Loaded and sorted {len(df_matches)} matches.")
    print("Normalizing team names...")
    # ... (Normalize names as before) ...
    for col in ["team1", "team2", "toss_winner", "winner"]:
        df_matches[col] = normalize_team_names(df_matches[col])
    df_matches["city"] = df_matches["city"].fillna("Unknown")

    # --- Pre-calculate First Innings Scores ---
    print("Loading ball-by-ball data to calculate first innings scores...")
    bbb_path = os.path.join(RAW_DATA_DIR, BALL_BY_BALL_FILE)
    first_innings_scores = {}  # Dict to store {match_number: score}
    try:
        bbb_cols_needed = [
            "ID",
            "Innings",
            "TotalRun",
        ]  # Assuming 'ID' maps to 'match_number'
        df_bbb = pd.read_csv(bbb_path, usecols=bbb_cols_needed)
        # Ensure numeric, handle errors
        df_bbb["TotalRun"] = pd.to_numeric(df_bbb["TotalRun"], errors="coerce").fillna(
            0
        )
        df_bbb["Innings"] = pd.to_numeric(df_bbb["Innings"], errors="coerce")
        # Filter for first innings only
        df_first_innings = df_bbb[df_bbb["Innings"] == 1].copy()
        # Group by match ID and sum total runs
        scores = df_first_innings.groupby("ID")["TotalRun"].sum().astype(int)
        first_innings_scores = scores.to_dict()
        print(
            f"Calculated first innings scores for {len(first_innings_scores)} matches."
        )
    except Exception as e:
        print(
            f"Warning: Could not calculate first innings scores from {BALL_BY_BALL_FILE}: {e}"
        )
        print("First innings score feature will use default value.")
    # --- End Score Calculation ---

    # --- Feature Calculation Loop ---
    print("Starting feature calculation loop (Including Player Form)...")
    team_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"wins": 0, "played": 0}
    )
    h2h_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    all_features_list = []
    total_rows = len(df_matches)

    for index, row in df_matches.iterrows():
        if index % 100 == 0:
            print(f"  Processing features for match {index}/{total_rows}...")

        match_id_source = int(row["match_number"])
        team1_name = row["team1"]
        team2_name = row["team2"]
        winner_name = row["winner"]
        current_date = row["match_date"].date()

        # Initialize features dict for this row
        current_features = {"match_number": match_id_source, "winner": winner_name}
        for col in ["team1", "team2", "toss_winner", "toss_decision", "venue", "city"]:
            current_features[col] = row[col]

        # --- Add First Innings Score ---
        # Look up pre-calculated score, use default if not found for this match_id
        current_features["first_innings_score"] = first_innings_scores.get(
            match_id_source, DEFAULT_SCORE
        )

        try:  # --- Calculate/Fetch other features (Win%, H2H, Prev Match, Player Form) ---
            team1_obj = Team.objects.filter(name=team1_name).first()
            team2_obj = Team.objects.filter(name=team2_name).first()
            if not team1_obj or not team2_obj:
                raise ValueError("Team object not found in DB")

            # Historical Win % (from trackers)
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
            h2h_played = (
                h2h_stats[team1_name][team2_name] + h2h_stats[team2_name][team1_name]
            )
            t1_h2h_wins = h2h_stats[team1_name][team2_name]
            current_features["team1_h2h_win_pct"] = (
                (t1_h2h_wins / h2h_played) if h2h_played > 0 else DEFAULT_H2H_WIN_PCT
            )

            # Previous Match Performance (Fetch PRE-CALCULATED from DB Match fields)
            last_match_t1 = (
                Match.objects.filter(
                    Q(team1=team1_obj) | Q(team2=team1_obj), date__lt=current_date
                )
                .order_by("-date", "-match_number")
                .first()
            )
            last_match_t2 = (
                Match.objects.filter(
                    Q(team1=team2_obj) | Q(team2=team2_obj), date__lt=current_date
                )
                .order_by("-date", "-match_number")
                .first()
            )
            current_features["team1_prev_score"] = (
                (
                    last_match_t1.team1_prev_score
                    if last_match_t1 and last_match_t1.team1 == team1_obj
                    else last_match_t1.team2_prev_score
                )
                if last_match_t1
                else DEFAULT_SCORE
            )
            current_features["team1_prev_wkts"] = (
                (
                    last_match_t1.team1_prev_wkts
                    if last_match_t1 and last_match_t1.team1 == team1_obj
                    else last_match_t1.team2_prev_wkts
                )
                if last_match_t1
                else DEFAULT_WKTS
            )
            current_features["team2_prev_score"] = (
                (
                    last_match_t2.team1_prev_score
                    if last_match_t2 and last_match_t2.team1 == team2_obj
                    else last_match_t2.team2_prev_score
                )
                if last_match_t2
                else DEFAULT_SCORE
            )
            current_features["team2_prev_wkts"] = (
                (
                    last_match_t2.team1_prev_wkts
                    if last_match_t2 and last_match_t2.team1 == team2_obj
                    else last_match_t2.team2_prev_wkts
                )
                if last_match_t2
                else DEFAULT_WKTS
            )
            for key in [
                "team1_prev_score",
                "team1_prev_wkts",
                "team2_prev_score",
                "team2_prev_wkts",
            ]:
                if current_features.get(key) is None:
                    current_features[key] = (
                        DEFAULT_SCORE if "score" in key else DEFAULT_WKTS
                    )

            # Recent Player Form (Call Helper)
            team1_form = get_aggregated_team_form(team1_obj, current_date)
            team2_form = get_aggregated_team_form(team2_obj, current_date)
            current_features["team1_avg_recent_bat_sr"] = team1_form.get(
                "avg_recent_sr", DEFAULT_BATTING_SR
            )
            current_features["team1_avg_recent_bowl_econ"] = team1_form.get(
                "avg_recent_econ", DEFAULT_BOWLING_ECON
            )
            current_features["team2_avg_recent_bat_sr"] = team2_form.get(
                "avg_recent_sr", DEFAULT_BATTING_SR
            )
            current_features["team2_avg_recent_bowl_econ"] = team2_form.get(
                "avg_recent_econ", DEFAULT_BOWLING_ECON
            )
            for key in [
                "team1_avg_recent_bat_sr",
                "team1_avg_recent_bowl_econ",
                "team2_avg_recent_bat_sr",
                "team2_avg_recent_bowl_econ",
            ]:
                if current_features.get(key) is None:
                    current_features[key] = (
                        DEFAULT_BATTING_SR if "sr" in key else DEFAULT_BOWLING_ECON
                    )

            # --- Store features for this row ---
            all_features_list.append(current_features)

            # --- Update history trackers for NEXT iteration ---
            team_stats[team1_name]["played"] += 1
            team_stats[team2_name]["played"] += 1
            if winner_name == team1_name:
                team_stats[team1_name]["wins"] += 1
                h2h_stats[team1_name][team2_name] += 1
            elif winner_name == team2_name:
                team_stats[team2_name]["wins"] += 1
                h2h_stats[team2_name][team1_name] += 1

        except Exception as e_row:
            print(f"ERROR processing row {index} (Match #{match_id_source}): {e_row}")
            all_features_list.append(
                {"match_number": match_id_source}
            )  # Add placeholder

    print(f"Finished feature calculation loop for {len(all_features_list)} matches.")

    # --- Create Final DataFrame and Save ---
    if not all_features_list:
        print("ERROR: No features were generated.")
        return

    df_final_features = pd.DataFrame(all_features_list)

    # Define final columns order (including target, score target, and all features)
    final_cols_order = [
        "match_number",
        "winner",
        "first_innings_score",  # IDs and Targets
        "team1",
        "team2",
        "toss_winner",
        "toss_decision",
        "venue",
        "city",  # Categorical (6)
        "team1_win_pct",
        "team2_win_pct",
        "team1_h2h_win_pct",  # Historical (3)
        "team1_prev_score",
        "team1_prev_wkts",
        "team2_prev_score",
        "team2_prev_wkts",  # Prev Match (4)
        "team1_avg_recent_bat_sr",
        "team1_avg_recent_bowl_econ",  # Team Form (4)
        "team2_avg_recent_bat_sr",
        "team2_avg_recent_bowl_econ",
    ]  # Total 1 + 2 + 6 + 3 + 4 + 4 = 20 columns

    # Ensure all expected columns exist, fill missing columns with defaults
    # (Copying the default filling logic from previous version)
    default_map = {
        "team1_win_pct": DEFAULT_WIN_PCT,
        "team2_win_pct": DEFAULT_WIN_PCT,
        "team1_h2h_win_pct": DEFAULT_H2H_WIN_PCT,
        "team1_prev_score": DEFAULT_SCORE,
        "team1_prev_wkts": DEFAULT_WKTS,
        "team2_prev_score": DEFAULT_SCORE,
        "team2_prev_wkts": DEFAULT_WKTS,
        "team1_avg_recent_bat_sr": DEFAULT_BATTING_SR,
        "team1_avg_recent_bowl_econ": DEFAULT_BOWLING_ECON,
        "team2_avg_recent_bat_sr": DEFAULT_BATTING_SR,
        "team2_avg_recent_bowl_econ": DEFAULT_BOWLING_ECON,
        "first_innings_score": DEFAULT_SCORE,
    }  # Added default for score
    for col in final_cols_order:
        if col not in df_final_features.columns:
            print(f"Warning: Column '{col}' missing, adding default.")
            df_final_features[col] = default_map.get(
                col, None
            )  # Use specific default or None

    df_final_features = df_final_features[final_cols_order]  # Select and order

    # Final check for NaNs in numerical columns and fill with defaults
    numerical_cols_final = [
        col
        for col in final_cols_order
        if col
        not in [
            "match_number",
            "winner",
            "team1",
            "team2",
            "toss_winner",
            "toss_decision",
            "venue",
            "city",
        ]
    ]
    for col in numerical_cols_final:
        if col in df_final_features.columns:
            df_final_features[col] = df_final_features[col].fillna(default_map[col])
        else:
            print(f"ERROR: Column {col} still missing before NaN fill.")

    output_path = os.path.join(PROCESSED_DATA_DIR, OUTPUT_FEATURE_FILE)
    print(
        f"Saving final features (including scores) to: {output_path} (Shape: {df_final_features.shape})"
    )
    try:
        df_final_features.to_csv(output_path, index=False)
        print("Features saved successfully.")
    except Exception as e:
        print(f"ERROR saving features to CSV: {e}")


if __name__ == "__main__":
    print("Starting Feature Build Process (Step 3 - Adding Scores)...")
    if DJANGO_LOADED:
        calculate_and_save_features()
    else:
        print("Exiting: Django environment setup failed.")
    print("Feature Build Process (Step 3) Finished.")
