import os
import requests
import pymongo
import logging
from datetime import datetime

# from dotenv import load_dotenv
# load_dotenv(".env.local")

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = "nba_stats"
COLLECTION_NAME = "h2h_records"
NBA_SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"

def connect_to_mongo():
    """Connect to MongoDB"""
    client = pymongo.MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    return db[COLLECTION_NAME]

def fetch_nba_schedule():
    response = requests.get(NBA_SCHEDULE_URL)
    if response.status_code != 200:
        logging.error("Failed to fetch NBA schedule: %s", response.status_code)
        return None
    return response.json()

NBA_TEAM_IDS = {
    1610612737, 1610612738, 1610612739, 1610612740, 1610612741, 1610612742, 1610612743, 1610612744,
    1610612745, 1610612746, 1610612747, 1610612748, 1610612749, 1610612750, 1610612751, 1610612752,
    1610612753, 1610612754, 1610612755, 1610612756, 1610612757, 1610612758, 1610612759, 1610612760,
    1610612761, 1610612762, 1610612763, 1610612764, 1610612765, 1610612766
}

def get_full_team_name(city, name):
    """Generate the full team name given the team city and team name."""
    if city == "LA" and name == "Clippers":
        return "Los Angeles Clippers"
    return f"{city} {name}"

def extract_h2h_records(data):
    """Extracts head-to-head (H2H) records for NBA teams using full team names."""
    h2h_records = {}

    if "leagueSchedule" in data and "gameDates" in data["leagueSchedule"]:
        for game_date_entry in data["leagueSchedule"]["gameDates"]:
            for game in game_date_entry.get("games", []):
                try:
                    game_date = datetime.strptime(game["gameDateUTC"], "%Y-%m-%dT%H:%M:%SZ").date()
                    today = datetime.utcnow().date()
                    if game_date >= today:
                        continue
                    home_team = get_full_team_name(game["homeTeam"]["teamCity"], game["homeTeam"]["teamName"])
                    away_team = get_full_team_name(game["awayTeam"]["teamCity"], game["awayTeam"]["teamName"])
                    home_team_id = game["homeTeam"]["teamId"]
                    away_team_id = game["awayTeam"]["teamId"]
                    if home_team_id not in NBA_TEAM_IDS or away_team_id not in NBA_TEAM_IDS:
                        continue  
                    home_score = game["homeTeam"]["score"]
                    away_score = game["awayTeam"]["score"]
                    if home_score > away_score:
                        winner, loser = home_team, away_team
                    elif away_score > home_score:
                        winner, loser = away_team, home_team
                    else:
                        continue  
                    if home_team not in h2h_records:
                        h2h_records[home_team] = {}
                    if away_team not in h2h_records:
                        h2h_records[away_team] = {}
                    if away_team not in h2h_records[home_team]:
                        h2h_records[home_team][away_team] = {"wins": 0, "losses": 0}
                    if home_team not in h2h_records[away_team]:
                        h2h_records[away_team][home_team] = {"wins": 0, "losses": 0}    
                    h2h_records[winner][loser]["wins"] += 1
                    h2h_records[loser][winner]["losses"] += 1

                except KeyError as e:
                    logging.error("Missing key: %s", e)
                except ValueError as e:
                    logging.error("Date parsing error: %s", e)

    return h2h_records

def store_h2h_in_mongo(h2h_data):
    """Inserts or updates H2H records in MongoDB."""
    collection = connect_to_mongo()

    if not h2h_data:
        logging.warning("No H2H records found.")
        return

    for team, opponents in h2h_data.items():
        existing_record = collection.find_one({"team": team})
        if existing_record:
            if existing_record["h2h"] != opponents:
                collection.update_one({"team": team}, {"$set": {"h2h": opponents}})
                logging.info("Updated H2H record for %s", team)
        else:
            collection.insert_one({"team": team, "h2h": opponents})
            logging.info("Inserted H2H record for %s", team)

def main():
    data = fetch_nba_schedule()
    if data:
        h2h_data = extract_h2h_records(data)
        store_h2h_in_mongo(h2h_data)

if __name__ == "__main__":
    main()

