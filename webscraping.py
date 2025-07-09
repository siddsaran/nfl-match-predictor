import requests
from bs4 import BeautifulSoup
import time
import pandas as pd

# Initial setup
url = "https://www.pro-football-reference.com/years/2024/"
data = requests.get(url)
soup = BeautifulSoup(data.text, "html.parser")

# On the website, the AFC and NFC standings are in separate categories
# in the HTML part, so we need to go through 2 different lists
afc_standings = soup.select('table.sortable')[0]
nfc_standings = soup.select('table.sortable')[1]
afc_links = afc_standings.find_all('a')
nfc_links = nfc_standings.find_all('a')

# Gather all the links of each team
links = []
for l in afc_links:
    links.append(l.get("href"))
for l in nfc_links:
    links.append(l.get("href"))
team_urls = [f"https://pro-football-reference.com{l}" for l in links]

# For each team, go to their page and go to their schedule page and extract
# that table to a DataFrame, and add team name to the DataFrame
official_list = []
for team_url in team_urls:
    data = requests.get(team_url)
    soup = BeautifulSoup(data.text, features='html.parser')
    hyperlinks = soup.select('ul.hoversmooth')
    schedule_link = None
    temp = []
    for l in hyperlinks:
        temp.extend(l.find_all('a'))
    for l in temp:
        temp_link = l.get('href')
        if 'gamelog' in temp_link:
            schedule_link = temp_link
            break
    team = schedule_link.split("/")[2].upper()
    link = f"https://pro-football-reference.com{schedule_link}"
    data = requests.get(link)

    # Read in the table and convert to a DataFrame
    df = pd.read_html(data.text, match="2024 Regular Season")[0]
    df.columns = df.columns.droplevel()
    df["Team"] = team
    official_list.append(df)
    # Wait 10 seconds between every 2 scrapes to ensure we don't
    # get blocked from web scraping
    time.sleep(10)

# Combine all teams' DataFrames and convert to csv for preparation
# for prediction model
match_df = pd.concat(official_list)
match_df.to_csv("matches.csv")




