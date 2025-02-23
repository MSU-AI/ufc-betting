import requests
import os
import json
from dotenv import load_dotenv
from balldontlie import BalldontlieAPI

load_dotenv()
api_key = os.getenv('BALLDONTLIE-API')
api = BalldontlieAPI(api_key=api_key)


def get_team_id():
    """
    Prompts the user to enter an NBA team name and retrieves the corresponding team ID from the API.
    """
    # Get user input
    team_name = input("Enter an NBA team name: ").strip().lower()

    response = api.nba.teams.list()

    try:
        teams_data = json.loads(response.model_dump_json())
        teams = teams_data['data']
    except requests.exceptions.JSONDecodeError:
        print("Failed to parse JSON response.")
        return None
    
    if teams:
        for team in teams:
            if team['name'].lower() == team_name:
                print(f"\nTeam ID for '{team['full_name']}' is: {team['id']}")
                return team['id']
            
        print("\nTeam not found. Please check the name and try again.")
    else:
        print(f"Failed to fetch teams. Status Code: {response.status_code}")

def get_players_from_team(team_id):
    """
    Fetches players from a specified team using the team_id.
    """
    # Call the API with team_id as a query parameter

    url = f"https://api.balldontlie.io/v1/players?team_ids[]={team_id}"

    headers = {
        'Authorization': api_key,
    }

    response = requests.request("GET", url, headers=headers)
    
    try:
        players_data = response.json()
        players = players_data['data']
    except requests.exceptions.JSONDecodeError:
        print("Failed to parse JSON response.")
        return None
    
    player_ids = []
    if players:
        print(f"\nPlayers from team ID {team_id}:")
        for player in players:
            print(f"{player['first_name']} {player['last_name']}")
            player_ids.append(player['id'])

        return player_ids
    else:
        print(f"\nNo players found for team ID {team_id}.")
        return []

def get_dummy_stats(player_ids, season):
    """
    Fetches dummy stats for a list of player_ids from dummy_stats.json.
    """
    with open('dummy_player_stats.json', 'r') as file:
        data = json.load(file)
        
        # Collect stats for all matching player_ids
        matching_stats = []
        
        for stat in data['data']:
            if stat['player']['id'] in player_ids and stat['season'] == season:
                matching_stats.append(stat)
    
    return matching_stats if matching_stats else None