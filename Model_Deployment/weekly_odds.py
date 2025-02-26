import requests
from dotenv import load_dotenv
import os
import json
from pprint import pprint


# MongoDB connection
load_dotenv()
api_key = os.getenv("ODDS-API")

sport = "basketball_nba"
regions = "us"
markets = "h2h"
format = "american"
# timeFrame = "2025-02-24T03:00:00Z"  # TODO: dynamically update this
bookmakers = "betonlineag,betmgm,betrivers,draftkings,fliff,espnbet,hardrockbet"
url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds/?apiKey={api_key}&markets={markets}&oddsFormat={format}&bookmakers={bookmakers}"


def get_upcoming_games():
    """Get upcoming games and their odds"""
    response = requests.get(url)

    if response.status_code == 200:
        try:
            raw_data = response.json()
            organized_games = []
            
            for game in raw_data:
                game_data = {
                    'game_info': {
                        'home_team': game['home_team'],
                        'away_team': game['away_team'],
                        'commence_time': game['commence_time'],
                        'id': game['id']
                    },
                    'odds': {}
                }
                
                # Organize odds by bookmaker
                for bookmaker in game['bookmakers']:
                    if 'markets' in bookmaker and bookmaker['markets']:
                        market = bookmaker['markets'][0]  # Get h2h market
                        if 'outcomes' in market:
                            # Create odds dictionary for this bookmaker
                            odds_dict = {}
                            for outcome in market['outcomes']:
                                team_name = outcome['name']
                                price = outcome['price']
                                odds_dict[team_name] = price
                            
                            game_data['odds'][bookmaker['key']] = odds_dict
                
                organized_games.append(game_data)
            
            return organized_games
            
        except json.JSONDecodeError:
            print("Failed to parse JSON response:", response.text)
            return None
    else:
        print("Request failed with status code:", response.status_code)
        return None

# Test the function
if __name__ == "__main__":
    upcoming_games = get_upcoming_games()
    pprint(upcoming_games)
