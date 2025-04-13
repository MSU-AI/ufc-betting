import requests
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import sys
from typing import List, Dict
from pymongo.errors import ServerSelectionTimeoutError, OperationFailure
from pymongo.operations import ReplaceOne
from datetime import datetime, timedelta
from dateutil import parser
from typing import Tuple

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # Add the project root to sys.path

from utils.team_enum import NBATeam

load_dotenv()


def connect_to_mongodb():
    try:
        client = MongoClient(os.getenv("MONGODB_URI"), serverSelectionTimeoutMS=5000)

        # Test the connection
        client.server_info()
        db = client["nba_stats"]
        collection = db["ev_results"]
        return collection
    except Exception as e:
        print(f"Error: Could not connect to MongoDB. {str(e)}")
        return None
    except ServerSelectionTimeoutError:
        print(
            "Error: Could not connect to MongoDB server. Check if the server is running."
        )
        return None
    except OperationFailure as e:
        print(f"Error: Authentication failed. {str(e)}")
        return None


def get_team_code(full_team_name: str) -> str:
    last_word = full_team_name.split()[-1]
    for team in NBATeam:
        if team.value == last_word:
            return team.name
    return None


# Restructure results to be inserted into db
def insert_results(results: List[Dict]):
    collection = connect_to_mongodb()
    if collection is None:
        return None

    bulk_results = []

    for game in results:
        game_info = game["game_info"]
        home_team = game_info["home_team"]
        away_team = game_info["away_team"]
        commence_time = game_info["commence_time"]
        commence_time = parser.isoparse(
            commence_time
        )  # Force commence_time to be a datetime
        home_win_prob = game_info["model_probabilities"]["home_win"]
        away_win_prob = game_info["model_probabilities"]["away_win"]

        home_code = get_team_code(home_team)
        away_code = get_team_code(away_team)

        # Log and skip if home or away code is None
        if home_code is None or away_code is None:
            print(f"Skipping game: {home_team} vs {away_team} - Missing team code")
            continue

        for bookmaker in game["bookmaker_odds"]:
            bookmaker_name = bookmaker["bookmaker"]
            home_odds = bookmaker["odds"]["home"]
            away_odds = bookmaker["odds"]["away"]
            home_ev = bookmaker["expected_value"]["home"]
            away_ev = bookmaker["expected_value"]["away"]
            home_kelly = bookmaker["kelly_fractions"]["home_kelly"]
            away_kelly = bookmaker["kelly_fractions"]["away_kelly"]

            game_result = {
                "home_team": home_team,
                "away_team": away_team,
                "home_code": home_code,
                "away_code": away_code,
                "commence_time": commence_time,
                "home_win_prob": round(home_win_prob, 4),
                "away_win_prob": round(away_win_prob, 4),
                "bookmaker": bookmaker_name,
                "home_odds": home_odds,
                "away_odds": away_odds,
                "home_ev": home_ev,
                "away_ev": away_ev,
                "home_kelly": home_kelly,
                "away_kelly": away_kelly,
            }

            time_window_start = commence_time - timedelta(hours=6)
            time_window_end = commence_time + timedelta(hours=6)

            existing_game = collection.find_one(
                {
                    "home_code": home_code,
                    "away_code": away_code,
                    "bookmaker": bookmaker_name,
                    "commence_time": {
                        "$gte": time_window_start,
                        "$lte": time_window_end,
                    },
                }
            )

            if existing_game:
                print("Existing game found!")
                filter_criteria = {"_id": existing_game["_id"]}
                # game_result["commence_time"] = existing_game["commence_time"] -- Uncomment to preserve old game time
            else:
                filter_criteria = {
                    "home_code": home_code,
                    "away_code": away_code,
                    "commence_time": commence_time,
                    "bookmaker": bookmaker_name,
                }

            bulk_results.append(ReplaceOne(filter_criteria, game_result, upsert=True))

    if bulk_results:
        collection.bulk_write(bulk_results)

    print("Generated EV insertion completed succesfully")
