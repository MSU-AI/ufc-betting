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


NBA_TEAM_IDS = {
    1610612737, 1610612738, 1610612739, 1610612740, 1610612741, 1610612742, 1610612743, 1610612744,
    1610612745, 1610612746, 1610612747, 1610612748, 1610612749, 1610612750, 1610612751, 1610612752,
    1610612753, 1610612754, 1610612755, 1610612756, 1610612757, 1610612758, 1610612759, 1610612760,
    1610612761, 1610612762, 1610612763, 1610612764, 1610612765, 1610612766
}


def extract_upcoming_games(data):
    upcoming_games = []
    today = datetime.utcnow().date()

    if "leagueSchedule" in data and "gameDates" in data["leagueSchedule"]:
        for game_date_entry in data["leagueSchedule"]["gameDates"]:
            for game in game_date_entry.get("games", []):  
                try:
                    game_date = datetime.strptime(game["gameDateUTC"], "%Y-%m-%dT%H:%M:%SZ").date()

                    # Skip past games
                    if game_date < today:
                        continue

                    home_team_id = game["homeTeam"]["teamId"]
                    away_team_id = game["awayTeam"]["teamId"]

                    # Ignore non-NBA teams
                    if home_team_id not in NBA_TEAM_IDS or away_team_id not in NBA_TEAM_IDS:
                        continue

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

    for game in games:
        existing_game = collection.find_one({"game_id": game["game_id"]})

        if existing_game:
            # Update only if the game details have changed
            if existing_game != game:
                collection.update_one({"game_id": game["game_id"]}, {"$set": game})
                print(f"Updated game {game['game_id']}")
        else:
            # Insert new game if it doesn't exist
            collection.insert_one(game)
            print(f"Inserted new game {game['game_id']}")


def main():
    data = fetch_nba_schedule()
    if data:
        upcoming_games = extract_upcoming_games(data)
        store_games_in_mongo(upcoming_games)


if __name__ == "__main__":
    main()
