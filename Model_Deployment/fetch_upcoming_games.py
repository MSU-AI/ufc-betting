import os
import requests
import pymongo
from dotenv import load_dotenv
from datetime import datetime

load_dotenv(".env.local")

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = "nba_stats"
COLLECTION_NAME = "upcoming_games"
NBA_SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"

def connect_to_mongo():
    """Connect to MongoDB"""
    client = pymongo.MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    return db[COLLECTION_NAME]

def fetch_nba_schedule():
    response = requests.get(NBA_SCHEDULE_URL)
    if response.status_code != 200:
        print("Failed to fetch NBA schedule:", response.status_code)
        return None
    return response.json()

# this should extract the upcoming games from the nba schedule
def extract_upcoming_games(data):
    upcoming_games = []
    today = datetime.utcnow().date()

    if "leagueSchedule" in data and "gameDates" in data["leagueSchedule"]:
        for game_date_entry in data["leagueSchedule"]["gameDates"]:
            for game in game_date_entry.get("games", []):  # Ensure games exist
                try:
                    game_date = datetime.strptime(game["gameDateUTC"], "%Y-%m-%dT%H:%M:%SZ").date()

                    if game_date >= today:
                        upcoming_games.append({
                            "game_id": game["gameId"],
                            "date": game_date.strftime("%Y-%m-%d"),
                            "home_team": game["homeTeam"]["teamName"],
                            "away_team": game["awayTeam"]["teamName"],
                            "arena": game["arenaName"],
                            "game_time": game["gameDateTimeUTC"]
                        })
                except KeyError as e:
                    print(f"Missing key: {e}")
                except ValueError as e:
                    print(f"Date parsing error: {e}")

    return upcoming_games

def store_games_in_mongo(games):
    collection = connect_to_mongo()
    if not games:
        print("No upcoming games found.")
        return
    collection.delete_many({}) 
    collection.insert_many(games)
    print(f"Inserted {len(games)} upcoming games into MongoDB.")

def main():
    data = fetch_nba_schedule()
    if data:
        upcoming_games = extract_upcoming_games(data)
        store_games_in_mongo(upcoming_games)

if __name__ == "__main__":
    main()
