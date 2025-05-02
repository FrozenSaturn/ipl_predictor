# predictor_api/management/commands/load_match_data.py

import pandas as pd
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

# Ensure Player is imported
from predictor_api.models import Team, Venue, Match, Player

# from dateutil import parser as date_parser
import logging

# import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Team Name Normalization ---
TEAM_NAME_MAPPING = {
    "Rising Pune Supergiant": "Rising Pune Supergiants",
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
}


def normalize_team_name(name):
    if pd.isna(name):
        return None
    return TEAM_NAME_MAPPING.get(name, name)


# --- Function to process player list string ---
def process_player_list(player_string):
    """Parses the player list string and yields cleaned player names."""
    if pd.isna(player_string) or not isinstance(player_string, str):
        return  # Skip if NaN or not a string

    # Split by comma, strip whitespace from each resulting item
    player_names = [name.strip() for name in player_string.split(",")]

    # Yield non-empty names
    for name in player_names:
        if name:  # Ensure name is not an empty string
            yield name


class Command(BaseCommand):
    help = "Loads match data from Match_Info.csv and populates Player table."

    def add_arguments(self, parser):
        parser.add_argument(
            "csv_file_path", type=str, help="Path to the Match_Info.csv file"
        )

    def handle(self, *args, **options):
        csv_file_path = options["csv_file_path"]
        logger.info(f"Starting data loading process from: {csv_file_path}")

        try:
            logger.info("Reading CSV file...")
            df = pd.read_csv(csv_file_path)
            logger.info(f"Read {len(df)} rows from CSV.")

            critical_cols = [
                "match_number",
                "match_date",
                "team1",
                "team2",
                "venue",
                "toss_winner",
                "toss_decision",
            ]
            # Also check player columns exist, though we handle NaNs later
            # player_cols = ["team1_players", "team2_players"]
            initial_rows = len(df)
            df = df.dropna(subset=critical_cols)
            if len(df) < initial_rows:
                logger.warning(
                    f"Dropped {initial_rows - len(df)} rows due to missing critical"
                    f" data in columns: {critical_cols}"
                )

            df = df.sort_values(by="match_date")

            logger.info("Normalizing team names...")
            # ... (keep team name normalization) ...
            df["Team1_Normalized"] = df["team1"].apply(normalize_team_name)
            df["Team2_Normalized"] = df["team2"].apply(normalize_team_name)
            df["TossWinner_Normalized"] = df["toss_winner"].apply(normalize_team_name)
            df["Winner_Normalized"] = df["winner"].apply(normalize_team_name)

            logger.info("Starting database population...")
            # Keep track of players created in this run to avoid excessive logging
            created_players_in_run = set()
            player_created_count = 0

            with transaction.atomic():
                match_count = 0
                skipped_count = 0
                # created_teams = set()
                # created_venues = set()

                for index, row in df.iterrows():
                    try:
                        # --- Process Venue, Teams (Keep existing logic) ---
                        venue_name = row["venue"].strip()
                        city_name = row.get("city")
                        city_name = city_name if pd.notna(city_name) else "Unknown"
                        venue, venue_created = Venue.objects.get_or_create(
                            name=venue_name, defaults={"city": city_name}
                        )
                        # ... log venue creation ...

                        team1, team1_created = Team.objects.get_or_create(
                            name=row["Team1_Normalized"]
                        )
                        # ... log team1 creation ...
                        team2, team2_created = Team.objects.get_or_create(
                            name=row["Team2_Normalized"]
                        )
                        # ... log team2 creation ...
                        toss_winner_team, toss_winner_created = (
                            Team.objects.get_or_create(
                                name=row["TossWinner_Normalized"]
                            )
                        )
                        # ... log toss_winner_team creation ...
                        winner_team = None
                        if row["Winner_Normalized"]:
                            winner_team, winner_created = Team.objects.get_or_create(
                                name=row["Winner_Normalized"]
                            )
                            # ... log winner_team creation ...
                        # --- End Venue/Team Processing ---

                        # --- Process Players ---
                        if "team1_players" in row:
                            for player_name in process_player_list(
                                row["team1_players"]
                            ):
                                player_obj, created = Player.objects.get_or_create(
                                    name=player_name
                                )
                                if created:
                                    player_created_count += 1
                                    if player_name not in created_players_in_run:
                                        logger.debug(f"Created Player: {player_name}")
                                        created_players_in_run.add(player_name)

                        if "team2_players" in row:
                            for player_name in process_player_list(
                                row["team2_players"]
                            ):
                                player_obj, created = Player.objects.get_or_create(
                                    name=player_name
                                )
                                if created:
                                    player_created_count += 1
                                    if player_name not in created_players_in_run:
                                        logger.debug(f"Created Player: {player_name}")
                                        created_players_in_run.add(player_name)
                        # --- End Player Processing ---

                        # --- Process Match (Keep existing logic) ---
                        try:
                            match_date = pd.to_datetime(row["match_date"]).date()
                        except ValueError:
                            logger.warning(
                                f"Skipping match row {index} due to "
                                f"invalid date: {row['match_date']}"
                            )
                            skipped_count += 1
                            continue
                        try:
                            match_id_src = int(row["match_number"])
                        except (ValueError, TypeError):
                            logger.warning(
                                f"Skipping match row {index} due to "
                                f"invalid match_number: {row['match_number']}"
                            )
                            skipped_count += 1
                            continue
                        result_type_val = row.get("result")
                        result_type_str = (
                            str(result_type_val).lower()
                            if pd.notna(result_type_val)
                            else "normal"
                        )
                        margin_value = None

                        match_obj, match_created = Match.objects.update_or_create(
                            match_id_source=match_id_src,
                            defaults={
                                "season": row.get("Season", match_date.year),
                                "date": match_date,
                                "match_number": match_id_src,
                                "team1": team1,
                                "team2": team2,
                                "venue": venue,
                                "toss_winner": toss_winner_team,
                                "toss_decision": row["toss_decision"].lower(),
                                "winner": winner_team,
                                "result_type": result_type_str,
                                "result_margin": margin_value,
                            },
                        )
                        # --- End Match Processing ---

                        if match_created:
                            match_count += 1
                            if match_count % 100 == 0:
                                logger.info(f"Processed {match_count} matches...")
                        # else: (optional logging for updated matches)
                        #    logger.debug(f"Updated existing match with Source ID: {match_id_src}")

                    except Exception as e:
                        logger.error(
                            f"Error processing row {index}"
                            f" (Match #{row.get('match_number', 'N/A')}): {e}",
                            exc_info=True,
                        )
                        skipped_count += 1

                logger.info("Finished processing matches.")
                if player_created_count > 0:
                    logger.info(f"Created {player_created_count} new player entries.")
                logger.info(
                    f"Total distinct players found/created: {Player.objects.count()}"
                )  # Log total players

            logger.info(f"Successfully created/updated {match_count} matches.")
            if skipped_count > 0:
                logger.warning(f"Skipped {skipped_count} match rows due to errors.")
            self.stdout.write(
                self.style.SUCCESS(f"Successfully loaded data from {csv_file_path}")
            )

        except FileNotFoundError:
            raise CommandError(f'Error: CSV file not found at "{csv_file_path}"')
        except KeyError as e:
            logger.error(f"A required column is missing from the CSV: {e}")
            raise CommandError(f"Missing column in CSV: {e}")
        except Exception as e:
            logger.exception("An unexpected error occurred during data loading.")
            raise CommandError(f"An error occurred: {e}")
