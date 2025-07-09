import pandas as pd
from sklearn.ensemble import RandomForestClassifier

matches = pd.read_csv('matches.csv', index_col=0)

# Remove the "total" season stats
matches_cleaned = matches.dropna(subset=['Rk']).copy()

matches_cleaned["Date"] = pd.to_datetime(matches_cleaned["Date"])
# Make the result an int for ML model to work
matches_cleaned["Target"] = matches_cleaned["Rslt"].astype("category").cat.codes
# Home game = -1, Away game = 0
matches_cleaned["Venue_Code"] = matches_cleaned["Unnamed: 5_level_1"].astype("category").cat.codes
# Assign codes 0-31 for all the teams
matches_cleaned["Opp_Code"] = matches_cleaned["Opp"].astype("category").cat.codes
# Day code for day of week 0-6 = Monday-Sunday
matches_cleaned["Day_Code"] = matches_cleaned["Date"].dt.dayofweek



