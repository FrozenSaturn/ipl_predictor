# Example: scripts/check_data.py
import pandas as pd
import os

RAW_DATA_DIR = "data/raw"
# Adjust filename based on your download
matches_file = os.path.join(
    RAW_DATA_DIR, "Ball_By_Ball_Match_Data.csv"
)  # Example filename

try:
    df_matches = pd.read_csv(matches_file)
    print("Matches data loaded successfully:")
    print(df_matches.info())
    print("\nFirst 5 rows:")
    print(df_matches.head())
    print(f"\nShape: {df_matches.shape}")
    print("\nColumns:", df_matches.columns.tolist())

    # Add similar checks for other files like ball-by-ball data if downloaded.

except FileNotFoundError:
    print(f"Error: File not found at {matches_file}." f"Please download the data.")
except Exception as e:
    print(f"An error occurred: {e}")
