# predictor_api/management/commands/load_performance_data.py

import pandas as pd
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction, IntegrityError
from predictor_api.models import Match, Player, PlayerMatchPerformance
import logging
from collections import defaultdict
import time

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants defining cricket rules interpretation for aggregation
EXTRAS_NO_BALL_FACED = ["wides"]
EXTRAS_NO_LEGAL_BALL_BOWLED = ["wides", "noballs"]
WICKET_KINDS_NOT_BOWLER = [
    "run out",
    "retired hurt",
    "retired out",
    "obstructing the field",
    "hit wicket",
    "handled the ball",
]


class Command(BaseCommand):
    help = "Loads ball-by-ball data, aggregates player performance per match, and populates PlayerMatchPerformance table."

    def add_arguments(self, parser):
        parser.add_argument(
            "csv_file_path",
            type=str,
            help="Path to the Ball_By_Ball_Match_Data.csv file",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=10000,
            help="Batch size for bulk DB operations.",
        )

    def handle(self, *args, **options):
        start_time = time.time()
        csv_file_path = options["csv_file_path"]
        batch_size = options["batch_size"]
        logger.info(f"Starting performance data loading from: {csv_file_path}")

        try:
            logger.info("Pre-loading Match and Player data into memory...")
            match_map = {m.match_id_source: m.pk for m in Match.objects.all()}
            player_map = {p.name: p.pk for p in Player.objects.all()}
            initial_player_count = len(player_map)
            logger.info(
                f"Loaded {len(match_map)} matches and {initial_player_count} players."
            )

            logger.info("Reading ball-by-ball CSV file...")
            try:
                # Define dtypes for potentially problematic columns
                dtypes = {
                    "ExtraType": str,
                    "Kind": str,
                    "PlayerOut": str,
                    "FieldersInvolved": str,
                    "BattingTeam": str,
                    "ID": int,
                }
                df = pd.read_csv(
                    csv_file_path, dtype=dtypes, keep_default_na=False
                )  # Handle empty strings directly
                df.rename(columns={"ID": "match_id_source"}, inplace=True)
            except Exception as e:
                logger.error(f"Error reading CSV file: {e}")
                raise CommandError("Failed to read CSV. Check file path and format.")

            # Explicitly handle NaNs or empty strings post-load if keep_default_na=False wasn't sufficient
            for col in ["ExtraType", "Kind", "PlayerOut", "FieldersInvolved"]:
                if col in df.columns:
                    df[col] = df[col].fillna("")  # Ensure NAs become empty strings

            # Ensure numeric columns are correctly typed, handling potential errors
            numeric_cols = [
                "Innings",
                "Overs",
                "BallNumber",
                "BatsmanRun",
                "ExtrasRun",
                "TotalRun",
                "IsWicketDelivery",
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = (
                        pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
                    )
                else:
                    logger.warning(
                        f"Numeric column {col} not found in CSV. Assuming 0."
                    )
                    df[col] = 0

            # Ensure player name columns are strings and handle potential 'nan' strings
            for col in ["Batter", "Bowler", "NonStriker", "PlayerOut"]:
                if col in df.columns:
                    df[col] = (
                        df[col]
                        .astype(str)
                        .replace("nan", "")
                        .replace("<NA>", "")
                        .strip()
                    )  # Clean player names

            logger.info(f"Read and cleaned {len(df)} rows from ball-by-ball CSV.")

            # Use a single dictionary to aggregate stats, keyed by (player_pk, match_pk)
            all_match_performances = defaultdict(lambda: defaultdict(int))
            processed_rows = 0
            skipped_balls_match_missing = 0
            skipped_balls_processing_error = 0
            player_created_count = 0

            logger.info("Aggregating performance data by match...")
            for match_id_source, match_df in df.groupby("match_id_source"):
                match_pk = match_map.get(match_id_source)
                if not match_pk:
                    skipped_balls_match_missing += len(match_df)
                    continue  # Skip balls for matches not found in the Match table

                for index, ball in match_df.iterrows():
                    processed_rows += 1
                    try:
                        batter_name = ball["Batter"]
                        bowler_name = ball["Bowler"]
                        player_out_name = ball["PlayerOut"]

                        # Skip row if essential names are missing after cleaning
                        if not batter_name or not bowler_name:
                            skipped_balls_processing_error += 1
                            continue

                        # --- Get or Create Player PKs (with basic IntegrityError handling) ---
                        batter_pk = player_map.get(batter_name)
                        if not batter_pk:
                            try:
                                player_obj, created = Player.objects.get_or_create(
                                    name=batter_name
                                )
                                if created:
                                    player_created_count += 1
                                batter_pk = player_obj.pk
                                player_map[batter_name] = batter_pk
                            except IntegrityError:
                                logger.warning(
                                    f"Integrity error getting/creating player {batter_name}, fetching again."
                                )
                                batter_pk = Player.objects.get(name=batter_name).pk
                                player_map[batter_name] = (
                                    batter_pk  # Ensure map is updated
                                )

                        bowler_pk = player_map.get(bowler_name)
                        if not bowler_pk:
                            try:
                                player_obj, created = Player.objects.get_or_create(
                                    name=bowler_name
                                )
                                if created:
                                    player_created_count += 1
                                bowler_pk = player_obj.pk
                                player_map[bowler_name] = bowler_pk
                            except IntegrityError:
                                logger.warning(
                                    f"Integrity error getting/creating player {bowler_name}, fetching again."
                                )
                                bowler_pk = Player.objects.get(name=bowler_name).pk
                                player_map[bowler_name] = bowler_pk

                        player_out_pk = None
                        if player_out_name:  # Only process if there is a player name
                            player_out_pk = player_map.get(player_out_name)
                            if not player_out_pk:
                                try:
                                    player_obj, created = Player.objects.get_or_create(
                                        name=player_out_name
                                    )
                                    if created:
                                        player_created_count += 1
                                    player_out_pk = player_obj.pk
                                    player_map[player_out_name] = player_out_pk
                                except IntegrityError:
                                    logger.warning(
                                        f"Integrity error getting/creating player {player_out_name}, fetching again."
                                    )
                                    player_out_pk = Player.objects.get(
                                        name=player_out_name
                                    ).pk
                                    player_map[player_out_name] = player_out_pk
                        # --- End Player PK handling ---

                        perf_key_batter = (batter_pk, match_pk)
                        perf_key_bowler = (bowler_pk, match_pk)

                        # Use setdefault to initialize if key doesn't exist
                        current_batting_stats = all_match_performances.setdefault(
                            perf_key_batter, defaultdict(int)
                        )
                        current_bowling_stats = all_match_performances.setdefault(
                            perf_key_bowler, defaultdict(int)
                        )

                        # --- Accumulate Batting Stats ---
                        current_batting_stats["runs_scored"] += ball["BatsmanRun"]
                        if ball["ExtraType"] not in EXTRAS_NO_BALL_FACED:
                            current_batting_stats["balls_faced"] += 1
                        if ball["BatsmanRun"] == 4:
                            current_batting_stats["fours_hit"] += 1
                        if ball["BatsmanRun"] == 6:
                            current_batting_stats["sixes_hit"] += 1

                        # --- Accumulate Bowling Stats ---
                        if ball["ExtraType"] not in EXTRAS_NO_LEGAL_BALL_BOWLED:
                            current_bowling_stats["balls_bowled"] += 1
                        bowler_extras_conceded = (
                            ball["ExtrasRun"]
                            if ball["ExtraType"] in ["wides", "noballs"]
                            else 0
                        )
                        current_bowling_stats["runs_conceded"] += (
                            ball["BatsmanRun"] + bowler_extras_conceded
                        )
                        if (
                            ball["IsWicketDelivery"] == 1
                            and ball["Kind"] not in WICKET_KINDS_NOT_BOWLER
                        ):
                            current_bowling_stats["wickets_taken"] += 1
                        if ball["BatsmanRun"] == 0 and bowler_extras_conceded == 0:
                            current_bowling_stats["dots_bowled"] += 1
                        if ball["BatsmanRun"] == 4:
                            current_bowling_stats["fours_conceded"] += 1
                        if ball["BatsmanRun"] == 6:
                            current_bowling_stats["sixes_conceded"] += 1

                        # --- Record Dismissal ---
                        if ball["IsWicketDelivery"] == 1 and player_out_pk:
                            perf_key_player_out = (player_out_pk, match_pk)
                            # Ensure player exists in dict before setting dismissal
                            current_out_stats = all_match_performances.setdefault(
                                perf_key_player_out, defaultdict(int)
                            )
                            current_out_stats["dismissal_kind"] = ball[
                                "Kind"
                            ]  # Overwrites previous if multiple? Should be ok.

                    except Exception as e:
                        # Log specific error but continue processing other balls
                        logger.error(
                            f"Error processing ball row {index} for Match ID {match_id_source}: {e}",
                            exc_info=False,
                        )
                        skipped_balls_processing_error += 1

                    # Log progress less frequently for large files
                    if processed_rows > 0 and processed_rows % 50000 == 0:
                        logger.info(
                            f"Processed {processed_rows // 1000}k ball records..."
                        )

            logger.info("Aggregation complete.")
            if player_created_count > 0:
                logger.info(
                    f"Created {player_created_count} new players found only in ball-by-ball data."
                )

            logger.info("Preparing performance objects for database...")
            final_perf_objects = []
            for (player_pk, match_pk), stats in all_match_performances.items():
                # Set dismissal kind to 'not out' for batters who faced balls but weren't dismissed
                if stats.get("balls_faced", 0) > 0 and "dismissal_kind" not in stats:
                    stats["dismissal_kind"] = "not out"

                # Final check for valid PKs before creating instance
                if player_pk is not None and match_pk is not None:
                    final_perf_objects.append(
                        PlayerMatchPerformance(
                            player_id=player_pk,
                            match_id=match_pk,
                            runs_scored=stats.get("runs_scored", 0),
                            balls_faced=stats.get("balls_faced", 0),
                            fours_hit=stats.get("fours_hit", 0),
                            sixes_hit=stats.get("sixes_hit", 0),
                            dismissal_kind=stats.get(
                                "dismissal_kind"
                            ),  # Allows None if never batted/bowled
                            balls_bowled=stats.get("balls_bowled", 0),
                            runs_conceded=stats.get("runs_conceded", 0),
                            wickets_taken=stats.get("wickets_taken", 0),
                            dots_bowled=stats.get("dots_bowled", 0),
                            fours_conceded=stats.get("fours_conceded", 0),
                            sixes_conceded=stats.get("sixes_conceded", 0),
                        )
                    )
                else:
                    logger.warning(
                        f"Skipping record due to missing PKs: PlayerPK={player_pk}, MatchPK={match_pk}"
                    )

            logger.info(f"Prepared {len(final_perf_objects)} performance objects.")

            logger.info("Starting database update using bulk methods...")
            # Fetch only the necessary fields for the lookup
            existing_lookup = {
                (p.player_id, p.match_id): p.pk
                for p in PlayerMatchPerformance.objects.filter(
                    match_id__in=match_map.values()
                ).only("pk", "player_id", "match_id")
            }
            logger.info(
                f"Fetched {len(existing_lookup)} existing performance record keys for comparison."
            )

            objs_to_create = []
            objs_to_update = []
            update_fields = [  # Fields included in bulk_update
                "runs_scored",
                "balls_faced",
                "fours_hit",
                "sixes_hit",
                "dismissal_kind",
                "balls_bowled",
                "runs_conceded",
                "wickets_taken",
                "dots_bowled",
                "fours_conceded",
                "sixes_conceded",
                # 'updated_at' is handled by auto_now=True if using obj.save(), NOT by bulk_update
            ]

            for perf_obj in final_perf_objects:
                key = (perf_obj.player_id, perf_obj.match_id)
                existing_pk = existing_lookup.get(key)
                if existing_pk:
                    perf_obj.pk = existing_pk  # Set PK for update operation
                    objs_to_update.append(perf_obj)
                else:
                    objs_to_create.append(perf_obj)

            logger.info(
                f"Identified {len(objs_to_create)} records to create and {len(objs_to_update)} records to update."
            )

            # Perform DB operations within a single transaction
            with transaction.atomic():
                if objs_to_create:
                    created_objs = PlayerMatchPerformance.objects.bulk_create(
                        objs_to_create, batch_size=batch_size
                    )
                    logger.info(
                        f"Bulk created {len(created_objs)} performance records."
                    )
                if objs_to_update:
                    updated_objs_count = PlayerMatchPerformance.objects.bulk_update(
                        objs_to_update, update_fields, batch_size=batch_size
                    )
                    logger.info(
                        f"Bulk updated {updated_objs_count} performance records."
                    )

            logger.info("Database update complete.")
            if skipped_balls_match_missing > 0:
                logger.warning(
                    f"Total balls skipped due to missing match links: {skipped_balls_match_missing}"
                )
            if skipped_balls_processing_error > 0:
                logger.warning(
                    f"Total balls skipped due to processing errors: {skipped_balls_processing_error}"
                )

            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"Performance data loading finished in {duration:.2f} seconds.")
            self.stdout.write(
                self.style.SUCCESS(
                    f"Successfully processed ball-by-ball data from {csv_file_path}"
                )
            )

        except FileNotFoundError:
            raise CommandError(
                f'Error: Ball-by-ball CSV file not found at "{csv_file_path}"'
            )
        except Exception as e:
            logger.exception(
                "An unexpected error occurred during performance data loading."
            )
            raise CommandError(f"An error occurred: {e}")
