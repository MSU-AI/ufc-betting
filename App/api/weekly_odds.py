import requests
from dotenv import load_dotenv
import os
import json

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

response = requests.get(url)

if response.status_code == 200:
    try:
        data = response.json()
        print(json.dumps(data, indent=4))
    except json.JSONDecodeError:
        print("Failed to parse JSON response:", response.text)
else:
    print("Request failed with status code:", response.status_code)
