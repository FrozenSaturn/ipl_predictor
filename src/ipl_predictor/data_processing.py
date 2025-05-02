# src/ipl_predictor/data_processing.py

import pandas as pd
import os
from typing import Tuple, List, Dict  # Added Optional
from collections import defaultdict

# --- Configuration Constants ---
RAW_DATA_DIR: str = "data/raw"
MATCH_INFO_FILE: str = "Match_Info.csv"
BALL_BY_BALL_FILE: str = "Ball_By_Ball_Match_Data.csv"  # <-- Add Ball-by-Ball filename

INITIAL_LOAD_COLUMNS_MATCH: List[str] = [  # Renamed for clarity
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
INITIAL_LOAD_COLUMNS_BBB: List[str] = [  # Columns needed from Ball-by-Ball
    "ID",  # Assuming this matches 'match_number' from Match_Info
    "Innings",  # May not be needed for simple aggregates, but good context
    "BattingTeam",
    "TotalRun",
    "IsWicketDelivery",
]
# Features to keep for the final model input (X)
FINAL_FEATURE_COLUMNS: List[str] = [
    "team1",
    "team2",
    "toss_winner",
    "toss_decision",
    "venue",
    "city",  # Original categoricals
    "team1_win_pct",
    "team2_win_pct",
    "team1_h2h_win_pct",  # Previous numerical
    "team1_prev_score",
    "team1_prev_wkts",
    "team2_prev_score",
    "team2_prev_wkts",  # New numerical
]
TARGET_COLUMN: str = "winner"

TEAM_NAME_MAPPING: Dict[str, str] = {  # Keep this updated
    "Kings XI Punjab": "Punjab Kings",
    "Delhi Daredevils": "Delhi Capitals",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
}
# --- End Configuration ---


# --- Keep normalize_team_names function ---
def normalize_team_names(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    # ... (keep existing code) ...
    print("Normalizing team names...")
    for col in columns:
        if col in df.columns:
            df[col] = df[col].replace(TEAM_NAME_MAPPING)
    return df


def load_and_prepare_match_info(file_path: str) -> pd.DataFrame:
    """Loads, sorts, cleans basic missing values from Match_Info.csv."""
    print(f"Attempting to load match info from: {file_path}")
    try:
        df = pd.read_csv(
            file_path, usecols=INITIAL_LOAD_COLUMNS_MATCH, parse_dates=["match_date"]
        )
        print(f"Successfully loaded {len(df)} match info rows.")
    except FileNotFoundError:  # ... (keep error handling) ...
        raise
    except Exception as e:  # ... (keep error handling) ...
        raise e

    # initial_rows = len(df)
    df.dropna(
        subset=[TARGET_COLUMN, "team1", "team2", "match_date", "match_number"],
        inplace=True,
    )
    # ... (keep dropna logging) ...

    print("Sorting match info by date and match_number...")
    df.sort_values(by=["match_date", "match_number"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    name_cols_to_normalize = ["team1", "team2", "toss_winner", "winner"]
    df = normalize_team_names(df, name_cols_to_normalize)

    if "city" in df.columns and df["city"].isnull().any():
        df["city"] = df["city"].fillna("Unknown")
        print("Filled missing 'city' values with 'Unknown'.")

    print(f"Match Info preparation complete. Shape: {df.shape}")
    return df


def load_and_prepare_bbb_data(file_path: str, match_ids: pd.Series) -> pd.DataFrame:
    """Loads and prepares Ball-by-Ball data, filtering by relevant match IDs."""
    print(f"Attempting to load ball-by-ball data from: {file_path}")
    try:
        # Load only necessary columns
        df_bbb = pd.read_csv(file_path, usecols=INITIAL_LOAD_COLUMNS_BBB)
        print(f"Successfully loaded {len(df_bbb)} ball-by-ball rows.")

        # Assuming 'ID' in BBB corresponds to 'match_number' in Match_Info
        valid_match_ids = set(match_ids.unique())
        initial_bbb_rows = len(df_bbb)
        df_bbb = df_bbb[
            df_bbb["ID"].isin(valid_match_ids)
        ].copy()  # Use .copy() to avoid SettingWithCopyWarning
        print(
            f"Filtered ball-by-ball data to {len(df_bbb)} rows matching"
            f" valid match IDs (removed {initial_bbb_rows - len(df_bbb)} rows)."
        )

        # Normalize team names here too
        df_bbb = normalize_team_names(df_bbb, ["BattingTeam"])

        # Convert relevant columns to numeric, coercing errors
        df_bbb["TotalRun"] = pd.to_numeric(df_bbb["TotalRun"], errors="coerce").fillna(
            0
        )
        df_bbb["IsWicketDelivery"] = pd.to_numeric(
            df_bbb["IsWicketDelivery"], errors="coerce"
        ).fillna(0)

        return df_bbb

    except FileNotFoundError:  # ... (keep error handling) ...
        raise
    except Exception as e:  # ... (keep error handling) ...
        raise e


def calculate_match_aggregates(
    df_bbb: pd.DataFrame,
) -> Dict[Tuple[int, str], Dict[str, int]]:
    """Calculates total runs and wickets for each team in each match."""
    print(
        "Calculating aggregates (runs, wickets)"
        " per team per match from ball-by-ball data..."
    )
    # Group by Match ID and Batting Team
    grouped = df_bbb.groupby(["ID", "BattingTeam"])

    # Calculate total runs and wickets
    runs = grouped["TotalRun"].sum().astype(int)
    wickets = grouped["IsWicketDelivery"].sum().astype(int)

    match_aggregates: Dict[Tuple[int, str], Dict[str, int]] = defaultdict(
        lambda: {"score": 0, "wickets": 0}
    )
    for index, score in runs.items():
        match_id, team_name = index
        match_aggregates[(match_id, team_name)]["score"] = score
    for index, wkts in wickets.items():
        match_id, team_name = index
        match_aggregates[(match_id, team_name)]["wickets"] = wkts

    print(f"Calculated aggregates for {len(match_aggregates)} team-match entries.")
    return match_aggregates


def calculate_historical_and_prev_match_stats(
    df: pd.DataFrame, match_aggregates: Dict[Tuple[int, str], Dict[str, int]]
) -> pd.DataFrame:
    """
    Calculates historical win percentages AND previous match performance.
    Iterates chronologically through the main match DataFrame.
    """
    print("Calculating historical win stats and previous match performance stats...")
    # Stores for historical win %
    team_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"wins": 0, "played": 0}
    )
    h2h_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # Stores for previous match performance {team: {'score': S, 'wickets': W}}
    prev_match_perf: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"score": 150, "wickets": 5}
    )  # Use default average? Or 0? Let's use ~avg

    # Lists to store calculated stats for each row
    team1_win_pcts, team2_win_pcts, team1_h2h_win_pcts = [], [], []
    team1_prev_scores, team1_prev_wkts, team2_prev_scores, team2_prev_wkts = (
        [],
        [],
        [],
        [],
    )

    total_rows = len(df)
    for index, row in df.iterrows():
        if index % 100 == 0:
            print(f"Processing combined stats: {index}/{total_rows}")

        match_id = row["match_number"]  # Use match_number as the ID
        team1 = row["team1"]
        team2 = row["team2"]
        winner = row["winner"]

        # --- Calculate stats *before* this match ---
        # Overall Win % (as before)
        t1_played = team_stats[team1]["played"]
        t1_wins = team_stats[team1]["wins"]
        t2_played = team_stats[team2]["played"]
        t2_wins = team_stats[team2]["wins"]
        t1_win_pct = (t1_wins / t1_played) if t1_played > 0 else 0.0
        t2_win_pct = (t2_wins / t2_played) if t2_played > 0 else 0.0

        # H2H Win % (as before)
        h2h_played = h2h_stats[team1][team2] + h2h_stats[team2][team1]
        t1_h2h_wins = h2h_stats[team1][team2]
        t1_h2h_win_pct = (t1_h2h_wins / h2h_played) if h2h_played > 0 else 0.5

        # Previous Match Performance (NEW)
        # Get the performance stored from the *last* time this team played
        t1_prev_perf = prev_match_perf[team1]
        t2_prev_perf = prev_match_perf[team2]

        # Append calculated stats for the *current* row
        team1_win_pcts.append(t1_win_pct)
        team2_win_pcts.append(t2_win_pct)
        team1_h2h_win_pcts.append(t1_h2h_win_pct)
        team1_prev_scores.append(t1_prev_perf["score"])
        team1_prev_wkts.append(t1_prev_perf["wickets"])
        team2_prev_scores.append(t2_prev_perf["score"])
        team2_prev_wkts.append(t2_prev_perf["wickets"])

        # --- Update historical stats *after* processing this match's outcome ---
        # Win/Played Counts
        team_stats[team1]["played"] += 1
        team_stats[team2]["played"] += 1
        if winner == team1:
            team_stats[team1]["wins"] += 1
        elif winner == team2:
            team_stats[team2]["wins"] += 1
        # H2H Counts
        if winner == team1:
            h2h_stats[team1][team2] += 1
        elif winner == team2:
            h2h_stats[team2][team1] += 1

        # --- Update previous performance store *after* this match ---

        t1_actual_perf = match_aggregates.get(
            (match_id, team1),
            {"score": t1_prev_perf["score"], "wickets": t1_prev_perf["wickets"]},
        )  # Default to prev if not found
        t2_actual_perf = match_aggregates.get(
            (match_id, team2),
            {"score": t2_prev_perf["score"], "wickets": t2_prev_perf["wickets"]},
        )  # Default to prev if not found
        prev_match_perf[team1] = t1_actual_perf
        prev_match_perf[team2] = t2_actual_perf

    print(f"Finished processing combined stats: {total_rows}/{total_rows}")

    # Add all calculated stats as new columns
    df["team1_win_pct"] = team1_win_pcts
    df["team2_win_pct"] = team2_win_pcts
    df["team1_h2h_win_pct"] = team1_h2h_win_pcts
    df["team1_prev_score"] = team1_prev_scores
    df["team1_prev_wkts"] = team1_prev_wkts
    df["team2_prev_score"] = team2_prev_scores
    df["team2_prev_wkts"] = team2_prev_wkts

    print("Added historical win percentage and previous match performance columns.")
    return df


def get_prepared_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Orchestrates loading, cleaning, feature"""
    """engineering, returns features (X) and target (y)."""
    print("--- Starting Data Preparation Pipeline (Adv. Feature Engineering) ---")
    match_info_path = os.path.join(RAW_DATA_DIR, MATCH_INFO_FILE)
    bbb_path = os.path.join(RAW_DATA_DIR, BALL_BY_BALL_FILE)

    # Load and prepare base match info
    df_matches = load_and_prepare_match_info(file_path=match_info_path)

    # Load and prepare ball-by-ball data, filtered by matches we have info for
    df_bbb = load_and_prepare_bbb_data(
        file_path=bbb_path, match_ids=df_matches["match_number"]
    )

    # Pre-calculate aggregates from ball-by-ball
    match_aggregates = calculate_match_aggregates(df_bbb=df_bbb)

    # Calculate historical and previous match features
    df_featured = calculate_historical_and_prev_match_stats(
        df=df_matches, match_aggregates=match_aggregates
    )

    # Select final features and target
    try:
        X = df_featured[FINAL_FEATURE_COLUMNS]
        y = df_featured[TARGET_COLUMN]
    except KeyError as e:  # ... (keep error handling) ...
        raise e

    print(
        f"Data preparation complete. Features shape: {X.shape}, Target shape: {y.shape}"
    )
    print("--- End Data Preparation Pipeline ---")
    return X, y


# --- Testing Block ---
if __name__ == "__main__":
    print("\n[INFO] Running data_processing.py directly for testing...")
    try:
        X_data, y_data = get_prepared_data()
        print("\n[SUCCESS] Data loading and preparation test finished.")
        print("\nSample Features (X head) including new stats:")
        print(X_data.head())
        new_cols = [
            "team1_prev_score",
            "team1_prev_wkts",
            "team2_prev_score",
            "team2_prev_wkts",
        ]
        print("\nCheck for NaN/Inf in new performance features:")
        print(X_data[new_cols].isnull().sum())
        print(X_data[new_cols].describe())
        # print("\nSample Target (y head):") # Optional
        # print(y_data.head())
    except Exception as e:
        print(f"\n[FAILURE] Data loading and preparation test failed: {e}")
        # raise e
