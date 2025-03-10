import os
import sys
import joblib
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
from weekly_odds import get_upcoming_games
from load_data import load_data
from utils.team_name_converter import convert_team_name
from pprint import pprint
from feature_engineering import engineer_features
from pydantic import BaseModel, Field
from typing import Dict
import json
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
import joblib
from expected_val import calc_expected_val

# Get path to model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'best_model.joblib')

class GameStats(BaseModel):
    home_avg_pts: float
    home_avg_reb: float
    home_avg_ast: float
    home_avg_stl: float
    home_avg_blk: float
    home_avg_fg_pct: float
    home_avg_fg3_pct: float
    home_avg_ft_pct: float
    away_avg_pts: float
    away_avg_reb: float
    away_avg_ast: float
    away_avg_stl: float
    away_avg_blk: float
    away_avg_fg_pct: float
    away_avg_fg3_pct: float
    away_avg_ft_pct: float

def validate_stats(home_stats: Dict, away_stats: Dict) -> Dict:
    """Validate and convert team stats to proper format"""
    try:
        model_input = {
            'home_avg_pts': float(home_stats['avg_pts']),
            'home_avg_reb': float(home_stats['avg_reb']),
            'home_avg_ast': float(home_stats['avg_ast']),
            'home_avg_stl': float(home_stats['avg_stl']),
            'home_avg_blk': float(home_stats['avg_blk']),
            'home_avg_fg_pct': float(home_stats['fg_pct']),
            'home_avg_fg3_pct': float(home_stats['fg3_pct']),
            'home_avg_ft_pct': float(home_stats['ft_pct']),
            'away_avg_pts': float(away_stats['avg_pts']),
            'away_avg_reb': float(away_stats['avg_reb']),
            'away_avg_ast': float(away_stats['avg_ast']),
            'away_avg_stl': float(away_stats['avg_stl']),
            'away_avg_blk': float(away_stats['avg_blk']),
            'away_avg_fg_pct': float(away_stats['fg_pct']),
            'away_avg_fg3_pct': float(away_stats['fg3_pct']),
            'away_avg_ft_pct': float(away_stats['ft_pct'])
        }
        
        # Validate with Pydantic
        validated_stats = GameStats(**model_input)
        return validated_stats.model_dump()
        
    except ValueError as e:
        print(f"\nValidation error: {str(e)}")
        return None
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        return None

def prepare_stats(home_stats: Dict, away_stats: Dict) -> pd.DataFrame:
    """Prepare stats for model input"""
    # Validate stats
    validated_stats = validate_stats(home_stats, away_stats)
    if validated_stats is None:
        return None
    
    validated_stats = engineer_features(validated_stats)
    
    # Create DataFrame and convert to float32
    return pd.DataFrame([validated_stats], dtype=np.float32)

def generate_ev():
    # Load model
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    team_stats = load_data()
    if team_stats is None:
        return None
    
    # Get upcoming games
    upcoming_games = get_upcoming_games()
    results = []
    
    # Process each game
    for game in upcoming_games:
        home_team = game['game_info']['home_team']
        away_team = game['game_info']['away_team']
        
        # Get team stats
        home_stats = team_stats.get(convert_team_name(home_team))
        away_stats = team_stats.get(convert_team_name(away_team))
        
        if not home_stats or not away_stats:
            continue
        
        # Prepare stats for model
        stats_df = prepare_stats(home_stats, away_stats)
        if stats_df is None:
            continue
        
        try:
            # Make prediction
            prediction = model.predict_proba(stats_df)
            away_win_prob = prediction[0][0]
            home_win_prob = prediction[0][1]
            
            game_result = {
                "game_info": {
                    "home_team": home_team,
                    "away_team": away_team,
                    "commence_time": game['game_info']['commence_time'],
                    "model_probabilities": {
                        "home_win": float(home_win_prob),
                        "away_win": float(away_win_prob)
                    }
                },
                "bookmaker_odds": []
            }
            
            # Process odds for each bookmaker
            for bookmaker, odds in game['odds'].items():
                if home_team in odds and away_team in odds:
                    home_odds = odds[home_team]
                    away_odds = odds[away_team]
                    
                    # Calculate EV for each bet
                    home_ev = calc_expected_val(home_team, home_odds, home_win_prob)
                    away_ev = calc_expected_val(away_team, away_odds, away_win_prob)
                    
                    bookmaker_result = {
                        "bookmaker": bookmaker,
                        "odds": {
                            "home": home_odds,
                            "away": away_odds
                        },
                        "expected_value": {
                            "home": home_ev[home_team]['edge'],
                            "away": away_ev[away_team]['edge']
                        }
                    }
                    
                    game_result["bookmaker_odds"].append(bookmaker_result)
            
            results.append(game_result)
            
        except Exception as e:
            print(f"\nError processing game {home_team} vs {away_team}: {str(e)}")
            continue
    
    return results

if __name__ == "__main__":
    results = generate_ev()
    
    pprint(results)
    
    #output results to json file
    with open('Model_Deployment/results.json', 'w') as f:
        json.dump(results, f, indent=2)









