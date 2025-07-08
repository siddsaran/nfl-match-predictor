import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://www.pro-football-reference.com/years/2024/"
data = requests.get(url)
soup = BeautifulSoup(data.text, features='html.parser')
afc_standings = soup.select('table.sortable')[0]
nfc_standings = soup.select('table.sortable')[1]
afc_links = afc_standings.find_all('a')
nfc_links = nfc_standings.find_all('a')

links = []
for l in afc_links:
    links.append(l.get("href"))
for l in nfc_links:
    links.append(l.get("href"))
team_urls = [f"https://pro-football-reference.com{l}" for l in links]
#print(team_urls)
gamelog_urls = []
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
    gamelog_urls.append(schedule_link)
team_urls = [f"https://pro-football-reference.com{l}" for l in gamelog_urls]
print(team_urls)





