"""
Title: NBA Player Biographical & Birthday Paradox Analysis
Description: A modular Python exercise script for loading, sanitizing, and analyzing
             NBA player birth dates to explore shared birthdays and player lookups.

"""


# Step 0: Import necessary libraries
# TODO: Import pandas, datetime, and any required modules
import pandas as pd
import numpy as np



def phase_1_discovery_and_sanitization():
    """
    [Phase 1] Data Discovery, Column Standardization & Sanitization (5 Marks)
    - Goal: Load the NBA player CSV file.
    - Standardize column names (remove whitespace, lowercase, snake_case).
    - Sanitize 'Birth Date' into standard datetime format.
    """
    print("\n[PHASE 1] Discovery & Data Sanitization (5 Marks)")
    print("-" * 65)
    # TODO: Load CSV file using pandas (handling potential quote formatting)
    # TODO: Standardize column headers (e.g., "Birth Date" -> "birth_date")
    # TODO: Parse 'birth_date' column into datetime objects using pd.to_datetime()
    # TODO: Inspect dataset shape and check for missing birth dates


    df = pd.read_csv("player_data.csv", quotechar='"')
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_")
    )

    df['birth_date'] = pd.to_datetime(df['birth_date'], errors='coerce')

    missing_bday = df['birth_date'].isnull().sum()
    print(f"Dataset shape: {df.shape}")
    print(f"Missing birth dates: {missing_bday}")

    return df


def phase_2_birthday_paradox_analysis(df):
    """
    [Phase 2] Birthday Paradox & Distribution Analysis (10 Marks)
    - Goal: Analyze the distribution of birth dates across months and days.
    - Calculate and demonstrate the probability of shared birthdays (Birthday Paradox)
      within subsets of NBA players.
    """
    print("\n[PHASE 2] Birthday Paradox & Date Distribution (10 Marks)")
    print("-" * 65)
    # TODO: Extract month and day components from 'birth_date'
    # TODO: Count frequency of birthdays per calendar day
    # TODO: Identify dates shared by multiple NBA players in the dataset

    df['birth_month'] = df['birth_date'].dt.month
    df['birth_day'] = df['birth_date'].dt.day

    # bday_freq = df['birth_date'].value_counts().sort_index()
    # print("Birthday Frequency Distribution:", bday_freq)


    df['birthday'] = df['birth_date'].dt.strftime('%m-%d')

    bday_freq=df["birthday"].value_counts().sort_index()

    shared_dates = bday_freq[bday_freq > 1]
    print("Shared Birth Dates:", shared_dates)
    print(shared_dates.sum())
    return df


def phase_3_player_lookup_system(df, player_name):
    """
    [Phase 3] Player Detail Retrieval (10 Marks)
    - Goal: Search for a specific player by name (case-insensitive or partial match).
    - Retrieve their biographical details (Position, College, Birth Date).
    """
    print(f"\n[PHASE 3] Player Lookup: Searching for '{player_name}' (10 Marks)")
    print("-" * 65)
    # TODO: Filter dataframe for the specified player name
    # TODO: Extract and return player details (Position, Ht, Wt, Colleges, Birth Date)

    df['player'] = df['player'].str.lower().str.strip()
    player_name = player_name.lower()

    result = df[df['player'].str.contains(player_name, case=False, na=False)]
    # print(result)
    # if player_name in df['player']:
    #
    #     print(df[df['player' , 'ht' , 'wt' , 'colleges' , 'birth_date']])

    details = result[['player', 'ht' , 'wt' , 'colleges' , 'birth_date']]
    print(details)
    return details




def phase_4_shared_birthday_cohort(df, player_name):
    """
    [Phase 4] Shared Birthday Peer Finder (10 Marks)
    - Goal: Identify the target player's exact birth date.
    - List all other NBA players who share the exact same date of birth.
    """
    print(f"\n[PHASE 4] Shared Birthday Cohort for '{player_name}' (10 Marks)")
    print("-" * 65)
    # TODO: Retrieve the birth date of the target player from Phase 3
    # TODO: Query the dataset for all other players born on that exact month and day (or year)
    # TODO: Output the list of peer players sharing the birthday

    player_birthday = df.loc[df['player'].str.contains(player_name, case=False, na=False) , 'birthday'].iloc[0]
    print(player_birthday)

    birthday_match = df[
        (df['birthday'] == player_birthday) &
        (df['player'] != player_name.lower())
    ]

    print(birthday_match[['player','birthday']])

    return df


def phase_5_evaluation_and_summary(df):
    """
    [Phase 5] Evaluation & Reporting (5 Marks)
    - Goal: Summarize statistical findings on player longevity, birth months,
      and birthday coincidences.
    """
    print("\n[PHASE 5] Evaluation & Summary Reporting (5 Marks)")
    print("-" * 65)
    # TODO: Print final summary statistics regarding dataset coverage and duplicate birthdays
    pass


# ==========================================
# THE FACADE FUNCTION (Orchestrator)
# ==========================================

def run_analytics_lifecycle():
    """
    Facade function coordinating the NBA analytics and birthday lookup workflow.
    """
    print("=" * 65)
    print("STARTING NBA BIRTHDAY & BIOGRAPHICAL LIFECYCLE (Out of 40 Marks)")
    print("=" * 65)

    # Target player to test lookup and shared birthdays
    target_player = "Kareem Abdul-Jabbar"

    # Executing phases chronologically
    df = phase_1_discovery_and_sanitization()
    df = phase_2_birthday_paradox_analysis(df)
    phase_3_player_lookup_system(df, target_player)
    phase_4_shared_birthday_cohort(df, target_player)
    # phase_5_evaluation_and_summary(df)
    pd.set_option('display.max_columns', None)
    pd.set_option("display.width", None)
    print(df)


# ==========================================
# MAIN GUARD
# ==========================================
if __name__ == "__main__":
    run_analytics_lifecycle()