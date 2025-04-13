import os
import sys
import joblib
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
from weekly_odds import get_upcoming_games
from load_data import load_data, get_recent_win_pcts
from utils.team_name_converter import convert_team_name
from pprint import pprint
from feature_engineering import engineer_features
from scaling import transform_features
from pydantic import BaseModel, Field
from typing import Dict
import json
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
import joblib
from expected_val import calc_expected_val
from results_insert import insert_results

# Get path to model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "best_model.joblib")


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
    home_winpct_last_3: float
    away_winpct_last_3: float
    home_winpct_last_5: float
    away_winpct_last_5: float
    home_winpct_last_10: float
    away_winpct_last_10: float


def validate_stats(home_stats: Dict, away_stats: Dict) -> Dict:
    """Validate and convert team stats to proper format"""
    try:
        model_input = {
            "home_avg_pts": float(home_stats["avg_pts"]),
            "home_avg_reb": float(home_stats["avg_reb"]),
            "home_avg_ast": float(home_stats["avg_ast"]),
            "home_avg_stl": float(home_stats["avg_stl"]),
            "home_avg_blk": float(home_stats["avg_blk"]),
            "home_avg_fg_pct": float(home_stats["fg_pct"]),
            "home_avg_fg3_pct": float(home_stats["fg3_pct"]),
            "home_avg_ft_pct": float(home_stats["ft_pct"]),
            "away_avg_pts": float(away_stats["avg_pts"]),
            "away_avg_reb": float(away_stats["avg_reb"]),
            "away_avg_ast": float(away_stats["avg_ast"]),
            "away_avg_stl": float(away_stats["avg_stl"]),
            "away_avg_blk": float(away_stats["avg_blk"]),
            "away_avg_fg_pct": float(away_stats["fg_pct"]),
            "away_avg_fg3_pct": float(away_stats["fg3_pct"]),
            "away_avg_ft_pct": float(away_stats["ft_pct"]),
            "home_winpct_last_3": float(home_stats["winpct_last_3"]),
            "away_winpct_last_3": float(away_stats["winpct_last_3"]),
            "home_winpct_last_5": float(home_stats["winpct_last_5"]),
            "away_winpct_last_5": float(away_stats["winpct_last_5"]),
            "home_winpct_last_10": float(home_stats["winpct_last_10"]),
            "away_winpct_last_10": float(away_stats["winpct_last_10"]),
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


def prepare_stats(
    home_stats: Dict, away_stats: Dict, home_abbr, away_abbr, date, target, game_id
) -> pd.DataFrame:
    """Prepare stats for model input"""
    # Validate stats
    validated_stats = validate_stats(home_stats, away_stats)
    if validated_stats is None:
        return None

    # Add metadata to validated_stats for features
    validated_stats["date"] = date
    validated_stats["game_id"] = metadata_to_numeric(game_id)
    validated_stats["season"] = 2025
    validated_stats["team_id_away"] = metadata_to_numeric(away_abbr)
    validated_stats["team_id_home"] = metadata_to_numeric(home_abbr)
    validated_stats["target"] = target

    df = pd.DataFrame([validated_stats])
    df = engineer_features(df, include_rolling=False)

    return df.drop(
        columns=[
            "date",
            "target",
        ],
        errors="ignore",
    ).astype(np.float32)


def metadata_to_numeric(team_abbr):
    return sum(ord(c) for c in team_abbr)


def calculate_kelly(p_model: float, odds: int) -> float:
    # Convert odds to decimal odds
    decimal_odds = 0
    if odds < 0:  # 'minus' odds
        decimal_odds = (100 / -(odds)) + 1
    else:
        decimal_odds = (odds / 100) + 1

    b = decimal_odds - 1
    q = 1 - p_model
    kelly = (b * p_model - q) / b if b != 0 else 0
    kelly *= 0.05  # Multiply kelly by const

    return round(kelly, 4)  # Neg kelly means to take the other side


def run_inference_pipeline(model, scaler) -> list:
    team_stats = load_data()
    if team_stats is None:
        return None

    # Get upcoming games
    upcoming_games = get_upcoming_games()
    results = []

    # Process each game
    for game in upcoming_games:
        home_team = game["game_info"]["home_team"]
        away_team = game["game_info"]["away_team"]
        date = game["game_info"]["commence_time"].split("T")[0]
        game_id = game["game_info"]["id"]

        # Get team stats
        home_stats = team_stats.get(convert_team_name(home_team))
        away_stats = team_stats.get(convert_team_name(away_team))

        if not home_stats or not away_stats:
            continue

        # Add rolling winning pcts
        home_abbr = convert_team_name(home_team, use_bkn=True)
        away_abbr = convert_team_name(away_team, use_bkn=True)

        home_win_pcts = get_recent_win_pcts(home_abbr)
        away_win_pcts = get_recent_win_pcts(away_abbr)

        home_stats.update(
            {
                "winpct_last_3": home_win_pcts["winpct_last_3"],
                "winpct_last_5": home_win_pcts["winpct_last_5"],
                "winpct_last_10": home_win_pcts["winpct_last_10"],
            }
        )
        target = int(home_win_pcts["latest_wl"] == "W")

        away_stats.update(
            {
                "winpct_last_3": away_win_pcts["winpct_last_3"],
                "winpct_last_5": away_win_pcts["winpct_last_5"],
                "winpct_last_10": away_win_pcts["winpct_last_10"],
            }
        )

        # Prepare stats for model
        stats_df = prepare_stats(
            home_stats, away_stats, home_abbr, away_abbr, date, target, game_id
        )
        if stats_df is None:
            continue

        # Scale stats df and drop cols unnecessary for inference
        stats_df = transform_features(stats_df, scaler)
        stats_df = stats_df.drop(
            columns=["game_id", "season", "team_id_home", "team_id_away"],
            errors="ignore",
        )

        try:
            # Make prediction
            prediction = model.predict_proba(stats_df)
            away_win_prob = prediction[0][0]
            home_win_prob = prediction[0][1]

            game_result = {
                "game_info": {
                    "home_team": home_team,
                    "away_team": away_team,
                    "commence_time": game["game_info"]["commence_time"],
                    "model_probabilities": {
                        "home_win": float(home_win_prob),
                        "away_win": float(away_win_prob),
                    },
                },
                "bookmaker_odds": [],
            }

            # Process odds for each bookmaker
            for bookmaker, odds in game["odds"].items():
                if home_team in odds and away_team in odds:
                    home_odds = odds[home_team]
                    away_odds = odds[away_team]

                    # Calculate EV for each bet
                    home_ev = calc_expected_val(
                        home_team, home_odds, float(home_win_prob)
                    )
                    away_ev = calc_expected_val(
                        away_team, away_odds, float(away_win_prob)
                    )

                    home_kelly = calculate_kelly(float(home_win_prob), home_odds)
                    away_kelly = calculate_kelly(float(away_win_prob), away_odds)

                    bookmaker_result = {
                        "bookmaker": bookmaker,
                        "odds": {"home": home_odds, "away": away_odds},
                        "expected_value": {
                            "home": home_ev[home_team]["edge"],
                            "away": away_ev[away_team]["edge"],
                        },
                        "kelly_fractions": {
                            "home_kelly": home_kelly,
                            "away_kelly": away_kelly,
                        },
                    }

                    game_result["bookmaker_odds"].append(bookmaker_result)

            results.append(game_result)

        except Exception as e:
            print(f"\nError processing game {home_team} vs {away_team}: {str(e)}")
            continue

    return results
