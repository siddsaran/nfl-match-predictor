import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

matches = pd.read_csv('matches.csv', index_col=0)

# Remove the "total" season stats
matches_cleaned = matches.dropna(subset=['Rk']).copy()

# Convert data to datetime object
matches_cleaned["Date"] = pd.to_datetime(matches_cleaned["Date"])
# Make the result an int for ML model to work
matches_cleaned["Target"] = (matches_cleaned["Rslt"] == "W").astype("int")
# Home game = 1, Away game = 0
matches_cleaned["Venue_Code"] = matches_cleaned["Unnamed: 5_level_1"].astype("category").cat.codes
matches_cleaned["Venue_Code"] = abs(matches_cleaned["Venue_Code"])
# Assign codes 0-31 for all the teams
matches_cleaned["Opp_Code"] = matches_cleaned["Opp"].astype("category").cat.codes
# Day code for day of week 0-6 = Monday-Sunday
matches_cleaned["Day_Code"] = matches_cleaned["Date"].dt.dayofweek

rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
# Train on 2022 and 2023 seasons, Test on 2024 season
train = matches_cleaned[matches_cleaned["Date"] < "2024-02-01"]
test = matches_cleaned[matches_cleaned["Date"] > "2024-02-01"]

# Predictor metrics
predictors = ["Venue_Code", "Opp_Code", "Day_Code"]

# Train and test
rf.fit(train[predictors], train["Target"])
preds = rf.predict(test[predictors])
acc = accuracy_score(test["Target"], preds)
'''
Accuracy: 0.5551470588235294
'''


combined = pd.DataFrame(dict(actual=test["Target"], prediction=preds))
tab = pd.crosstab(index = combined["actual"], columns=combined["prediction"])
'''
Tab: 0 = L/T, 1 = W
prediction   0    1
actual
0           150   122
1           120   152
'''

# Precision score: When model predicts win, percent of time
# that they actually won
p_score = precision_score(test["Target"], preds)
'''
Precision Score: 0.5547445255474452, not great
'''

grouped_matches = matches_cleaned.groupby("Team")
group = grouped_matches.get_group("BAL")

# Take into account how team is doing before game
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("Date")
    # Take current week out and compute rolling averages of 3 games before
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset = new_cols)
    return group

# Cols to consider for rolling averages
cols = ["Pts", "PtsO", "Cmp%", "PYds", "PTD", "PY/A", "RYds", "RTD", "RY/A"]
new_cols = [f"{c}_rolling" for c in cols]

# Apply to all teams and return
matches_rolling = matches_cleaned.groupby("Team").apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling = matches_rolling.droplevel("Team")
matches_rolling.index = range(matches_rolling.shape[0])
print(matches_rolling)