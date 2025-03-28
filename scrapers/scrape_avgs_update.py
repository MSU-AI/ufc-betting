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
collection = db["team_averages"]


def update_scrape():
    for team in NBATeam:
        filter_query = {"name": team.name}
        document = collection.find_one(filter_query)

        if document:
            update_doc = document.copy()
            stats = {}  # init here for the field loop below

            url = f"https://www.basketball-reference.com/teams/{team.name}/2025.html?sr&utm_source=direct&utm_medium=Share&utm_campaign=ShareTool#all_team_and_opponent"
            response = requests.get(url)

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")

                table = soup.find(
                    "table",
                    {"class": "stats_table sortable soc", "id": "per_game_stats"},
                )

                if table:
                    tfoot = table.find("tfoot")

                    stats = {
                        "games": int(
                            tfoot.find("td", {"data-stat": "games"}).text.strip()
                        ),
                        "fg_pct": float(
                            tfoot.find("td", {"data-stat": "fg_pct"}).text.strip()
                        ),
                        "fg3_pct": float(
                            tfoot.find("td", {"data-stat": "fg3_pct"}).text.strip()
                        ),
                        "ft_pct": float(
                            tfoot.find("td", {"data-stat": "ft_pct"}).text.strip()
                        ),
                        "avg_reb": float(
                            tfoot.find("td", {"data-stat": "trb_per_g"}).text.strip()
                        ),
                        "avg_ast": float(
                            tfoot.find("td", {"data-stat": "ast_per_g"}).text.strip()
                        ),
                        "avg_stl": float(
                            tfoot.find("td", {"data-stat": "stl_per_g"}).text.strip()
                        ),
                        "avg_blk": float(
                            tfoot.find("td", {"data-stat": "blk_per_g"}).text.strip()
                        ),
                        "avg_pts": float(
                            tfoot.find("td", {"data-stat": "pts_per_g"}).text.strip()
                        ),
                    }
                else:
                    print(
                        "Could not find the table with class 'stats_table sortable' and id 'per_game_stats'."
                    )
            else:
                print(
                    f"Failed to retrieve the webpage. Status code: {response.status_code}"
                )

            for key, val in document.items():
                for statKey, statVal in stats.items():
                    if key == statKey:
                        update_doc[key] = statVal

            update_doc["_id"] = document["_id"]

            collection.update_one({"_id": document["_id"]}, {"$set": update_doc})
        else:
            print(f"Doc not found")


update_scrape()  # Run immediately
