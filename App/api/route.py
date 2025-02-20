import requests
import os
from dotenv import load_dotenv
import json
from utils.route_utils import *
from balldontlie import BalldontlieAPI

load_dotenv()
api_key = os.getenv('BALLDONTLIE-API')
api = BalldontlieAPI(api_key=api_key)


team_id = get_team_id()
players = get_players_from_team(team_id)
print(players)

season = 2024
stats = get_dummy_stats([players[0], players[1]], season)
print(json.dumps(stats, indent=4))

# Testing Teams endpoint
response = api.nba.teams.get(team_id=9)

# Pretty-print the JSON response
try:
    data = json.loads(response.model_dump_json())
    # TODO: uncomment to view json response
    # print(json.dumps(data, indent=4))
except json.JSONDecodeError:
    print("Failed to parse JSON response:", response.text)
    exit()

# Extract the required fields
if data:
    team_stats = {
        "total_matches_played": 0,
        "total_wins": 0,
        "total_losses": 0,
        "total_points": 0,
        "total_rebounds": 0,
        "total_assists": 0,
        "total_steals": 0,
        "total_blocks": 0,
        "total_fg_made": 0,
        "total_fg_attempted": 0,
        "total_3pt_made": 0,
        "total_3pt_attempted": 0,
        "total_ft_made": 0,
        "total_ft_attempted": 0,
    }

    # Iterate over each game
    # for game in data:
    #     team_stats["total_matches_played"] += game.get("Games", 0)
    #     team_stats["total_wins"] += game.get("Wins", 0)
    #     team_stats["total_losses"] += game.get("Losses", 0)
    #     team_stats["total_points"] += game.get("Points", 0)
    #     team_stats["total_rebounds"] += game.get("Rebounds", 0)
    #     team_stats["total_assists"] += game.get("Assists", 0)
    #     team_stats["total_steals"] += game.get("Steals", 0)
    #     team_stats["total_blocks"] += game.get("BlockedShots", 0)
    #     team_stats["total_fg_made"] += game.get("FieldGoalsMade", 0)
    #     team_stats["total_fg_attempted"] += game.get("FieldGoalsAttempted", 0)
    #     team_stats["total_3pt_made"] += game.get("ThreePointersMade", 0)
    #     team_stats["total_3pt_attempted"] += game.get("ThreePointersAttempted", 0)
    #     team_stats["total_ft_made"] += game.get("FreeThrowsMade", 0)
    #     team_stats["total_ft_attempted"] += game.get("FreeThrowsAttempted", 0)

    team_stats = {key: round(value, 2) for key, value in team_stats.items()}

    print(json.dumps(team_stats, indent=4))
else:
    print("No data available for this team.")

#     # Calculate averages
#     wl_percent = round(team_stats["total_awarded_matches"] / team_stats["total_matches_played"], 2)
#     avg_pts = round(team_stats["points"] / team_stats["total_matches_played"], 2)
#     avg_reb = round(team_stats["total_rebounds"] / team_stats["total_matches_played"], 2)
#     avg_ast = round(team_stats["total_assists"] / team_stats["total_matches_played"], 2)
#     avg_stl = round(team_stats["total_steals"] / team_stats["total_matches_played"], 2)
#     avg_blk = round(team_stats["total_blocks"] / team_stats["total_matches_played"], 2)
#     avg_fg_pct = round(team_stats["field_goal_percentage"], 2)
#     avg_fg3_pct = round(team_stats["three_point_percentage"], 2)
#     avg_ft_pct = round(team_stats["free_throw_percentage"], 2)