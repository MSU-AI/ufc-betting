import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from utils.team_enum import NBATeam

# MongDB connection
load_dotenv()
mongo_url = os.getenv("MONGODB_URI")
client = MongoClient(mongo_url)
db = client["nba_stats"]
collection = db["recent_game_stats"]


def isHome(versusStr):
    return versusStr == "vs"


# Scraping
for team in NBATeam:
    url = f"https://www.statmuse.com/nba/ask/{team.value}-game-log-last-10-games"
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")

        table = soup.find("table", {"class": "whitespace-nowrap w-full"})

        if table:
            tbody = table.find("tbody")
            rows = tbody.find_all("tr")  # should be 10 rows/games

            for row in rows:
                cols = row.find_all("td")

                boxscore_stats = {
                    "team": cols[4].find("div").text.strip(),
                    "opponent": cols[6].find("div").text.strip(),
                    "date": cols[3].find("span").text.strip(),
                    "isHome": isHome(cols[5].find("div").find("span").text.strip()),
                    "result": cols[7].find("div", class_="w-5").text.strip(),
                    "points": int(cols[9].find("div").find("span").text.strip()),
                    "rebounds": int(cols[10].find("div").find("span").text.strip()),
                    "assists": int(cols[11].find("div").find("span").text.strip()),
                    "steals": int(cols[12].find("div").find("span").text.strip()),
                    "blocks": int(cols[13].find("div").find("span").text.strip()),
                    "fg_pct": float(cols[16].find("div").find("span").text.strip()),
                    "fg3_pct": float(cols[19].find("div").find("span").text.strip()),
                    "ft_pct": float(cols[22].find("div").find("span").text.strip()),
                }

                # TODO: uncomment to add to collection
                result = collection.insert_one(boxscore_stats)
                print(f"Data inserted with ID: {result.inserted_id}")
        else:
            print(
                "Could not find the table with class 'stats_table sortable' and id 'per_game_stats'."
            )
    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
