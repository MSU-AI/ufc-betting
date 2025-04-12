from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, OperationFailure
import os
from datetime import datetime
from dotenv import load_dotenv
from pprint import pprint
from expected_val import calc_expected_val

load_dotenv()


def connect_to_mongodb(collection_name: str):
    try:
        client = MongoClient(os.getenv("MONGODB_URI"), serverSelectionTimeoutMS=5000)

        # Test the connection
        client.server_info()
        db = client["nba_stats"]
        collection = db[collection_name]
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


def get_recent_win_pcts(team_abbr):
    try:
        collection = connect_to_mongodb("recent_game_stats")
        if collection is None:
            return None

        games = list(
            collection.find({"team": team_abbr}, {"_id": 0, "date": 1, "result": 1})
        )

        if not games:
            print(f"No recent games found for team: {team_abbr}")
            return None

        # Sort manually by parsed date
        games.sort(key=lambda x: datetime.strptime(x["date"], "%m/%d/%Y"), reverse=True)

        return {
            "winpct_last_3": compute_win_pct(3, games),
            "winpct_last_5": compute_win_pct(5, games),
            "winpct_last_10": compute_win_pct(10, games),
            "latest_wl": games[0].get("result"),
        }
    except Exception as e:
        print(f"Error in get_recent_win_pcts: {str(e)}")


def compute_win_pct(numGames, games):
    recent = games[:numGames]
    wins = sum(1 for g in recent if g.get("result") == "W")
    return round(wins / numGames, 3)


def reorg_data(data):
    # reorganize data into a dictionary with team names as keys
    team_stats = {}
    for team in data:
        # remove the name field
        team_name = team.pop("name")
        team_stats[team_name] = team
    return team_stats


def load_data():
    try:
        collection = connect_to_mongodb("team_averages")
        if collection is None:
            return None

        # Convert MongoDB cursor to list of dictionaries
        data = list(collection.find({}, {"_id": 0}))

        if not data:
            print("Error: No data found in collection")
            return None

        print(f"\nSuccessfully loaded {len(data)} team stats")
        return reorg_data(data)

    except Exception as e:
        print(f"Unexpected error while loading data: {str(e)}")
        return None


# Test the function
if __name__ == "__main__":
    team_stats = load_data()
    if team_stats is None:
        print("Failed to load team stats")
    else:
        print("Successfully loaded team stats")
        pprint(team_stats)
