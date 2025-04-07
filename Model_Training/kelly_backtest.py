import sys
import os
from pathlib import Path

# Add parent directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
import joblib
import warnings
from utils.feature_engineering import engineer_features
from datetime import datetime
from team_name_converter import convert_team_name
import json
from train_models import FFNClassifier
from train_models import ResFFNClassifier
from train_models import ResidualBlock
import os



warnings.filterwarnings('ignore', category=FutureWarning)  # For pandas FutureWarnings
warnings.filterwarnings('ignore', category=UserWarning)    # For UserWarnings
warnings.filterwarnings('ignore', category=RuntimeWarning) # For RuntimeWarnings

def convert_american_to_decimal(american_odds):
    """Convert American odds to decimal odds."""
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1

def kelly_criterion(probability, decimal_odds):
    """Calculate Kelly Criterion fraction."""
    q = 1 - probability
    b = decimal_odds - 1
    kelly = (b * probability - q) / b
    return max(0, min(1, kelly))  # Ensure between 0 and 1

def backtest_kelly(model_path, odds_data_path, stats_data_path, max_bet_percentage=0.05, 
                   min_kelly_threshold=0.05, min_ev_threshold=0.1, use_thresholds=True,
                   kelly_fraction_multiplier=0.25):
    """Run backtesting using Kelly Criterion for bankroll management.
    
    Parameters:
    - model_path: Path to the trained model file
    - odds_data_path: Path to the odds data CSV
    - stats_data_path: Path to the stats data CSV
    - max_bet_percentage: Maximum percentage of bankroll to bet
    - min_kelly_threshold: Minimum Kelly criterion value to place a bet
    - min_ev_threshold: Minimum expected value to place a bet
    - use_thresholds: Whether to apply minimum Kelly and EV thresholds
    - kelly_fraction_multiplier: Fraction of Kelly to bet
    """
    print("\nStarting backtest execution...")
    
    # Load model and data
    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    print("Loading odds and stats data...")
    odds_df = pd.read_csv(odds_data_path)
    stats_df = pd.read_csv(stats_data_path)
    
    print(f"Initial data shapes - Odds: {odds_df.shape}, Stats: {stats_df.shape}")
    
    # Convert win/loss to binary
    stats_df['target'] = (stats_df['wl_home'] == 'W').astype(int)
    
    # Convert date columns to datetime# Remove time information
    odds_df['date'] = pd.to_datetime(odds_df['date']).dt.date  # Remove time information
    stats_df['date'] = pd.to_datetime(stats_df['date']).dt.date  # Remove time information
    
    # Initialize tracking variables
    bankroll = 1000  # Starting bankroll
    max_bankroll = bankroll
    min_bankroll = bankroll
    total_bets = 0
    winning_bets = 0
    total_profit = 0
    bet_history = []
    skipped_games = []  # List to store skipped games information
    
    # Sort data by date
    odds_df = odds_df.sort_values('date')
    stats_df = stats_df.sort_values('date')
    
    #remove games from odds_df that before the first game in stats_df
    odds_df = odds_df[odds_df['date'] >= stats_df['date'].min()]
    
    # Process each game
    print("\nStarting game-by-game analysis...")
    for idx, game in enumerate(odds_df.iterrows(), 1):
        game = game[1]  # Get the row data
        if idx % 100 == 0:  # Print progress every 100 games
            print(f"Processing game {idx}/{len(odds_df)} - Current bankroll: ${bankroll:.2f}")
        
        # Find corresponding stats for this game using team abbreviations
        game_stats = stats_df[
            (stats_df['date'] == game['date']) & 
            (stats_df['team_abbreviation_home'] == game['team']) & 
            (stats_df['team_abbreviation_away'] == game['opponent'])
        ]
        
        if game_stats.empty:
            # Document the mismatched game
            skipped_games.append({
                'date': game['date'],
                'team': game['team'],
                'opponent': game['opponent']
            })
            # Print skipped game information
            print(f"Skipping game {idx} - No matching stats found for Date: {game['date']}, Team: {game['team']}, Opponent: {game['opponent']}")
            continue
            
        # Prepare features for prediction
        features = game_stats.copy()
        features = features.drop(columns=['target', 'team_abbreviation_home', 'team_abbreviation_away', 'game_id', 'date', 'team_id_home', 'team_id_away', 'season', 'wl_home'])
        
    
        
        home_win_prob = model.predict_proba(features)[0][1]
        away_win_prob = 1 - home_win_prob  # Away team's probability is complement of home
        
        # Calculate confidence metrics (how far from random 0.5 probability)
        home_confidence = abs(home_win_prob - 0.5) * 2  # Scale to 0-1 range
        away_confidence = abs(away_win_prob - 0.5) * 2  # Scale to 0-1 range
        
        # Convert odds to decimal for both teams
        home_decimal_odds = convert_american_to_decimal(game['moneyLine'])
        away_decimal_odds = convert_american_to_decimal(game['opponentMoneyLine'])
        
        # Calculate Kelly fraction for both teams
        home_kelly_fraction = kelly_criterion(home_win_prob, home_decimal_odds)
        away_kelly_fraction = kelly_criterion(away_win_prob, away_decimal_odds)
        
        # Calculate expected value for both teams
        home_ev = home_win_prob * (home_decimal_odds - 1) - (1 - home_win_prob)
        away_ev = away_win_prob * (away_decimal_odds - 1) - (1 - away_win_prob)
        
        # Determine viability based on thresholds
        if use_thresholds:
            home_viable = home_kelly_fraction >= min_kelly_threshold and home_ev >= min_ev_threshold
            away_viable = away_kelly_fraction >= min_kelly_threshold and away_ev >= min_ev_threshold
        else:
            # If not using thresholds, consider all bets with positive Kelly and EV
            home_viable = home_kelly_fraction > 0 and home_ev > 0
            away_viable = away_kelly_fraction > 0 and away_ev > 0
        
        if not (home_viable or away_viable):
            continue
            
        # Calculate bet size for both teams (capped at current max_bet)
        current_max_bet = bankroll * max_bet_percentage  # Dynamic max bet based on current bankroll
        
        home_bet_size = 0 if not home_viable else min(
            bankroll * home_kelly_fraction * kelly_fraction_multiplier, 
            current_max_bet
        )
        away_bet_size = 0 if not away_viable else min(
            bankroll * away_kelly_fraction * kelly_fraction_multiplier, 
            current_max_bet
        )
        
        # Skip if both bet sizes are too small
        if home_bet_size < 1 and away_bet_size < 1:
            continue
            
        # Determine which side to bet (choose the one with higher expected value)
        if home_ev > away_ev and home_bet_size >= 1:
            bet_side = 'home'
            bet_size = home_bet_size
            win_prob = home_win_prob
            decimal_odds = home_decimal_odds
            kelly_fraction = home_kelly_fraction
            original_odds = game['moneyLine']
            bet_ev = home_ev
        elif away_bet_size >= 1:
            bet_side = 'away'
            bet_size = away_bet_size
            win_prob = away_win_prob
            decimal_odds = away_decimal_odds
            kelly_fraction = away_kelly_fraction
            original_odds = game['opponentMoneyLine']
            bet_ev = away_ev
        else:
            if idx % 100 == 0:  # Only print skipped bets every 100 games
                print(f"Skipping game {idx} - No viable bet found")
            continue

        # Determine if bet won
        won = (game['score'] > game['opponentScore']) if bet_side == 'home' else (game['score'] < game['opponentScore'])
        
        # Calculate profit/loss
        if won:
            profit = bet_size * (decimal_odds - 1)
            winning_bets += 1
        else:
            profit = -bet_size
            
        # Update bankroll
        bankroll += profit
        total_profit += profit
        total_bets += 1
        
        # Track max/min bankroll
        max_bankroll = max(max_bankroll, bankroll)
        min_bankroll = min(min_bankroll, bankroll)
        
        # Record bet (keep confidence metrics for information only)
        bet_history.append({
            'date': game['date'],
            'team': game['team'],
            'opponent': game['opponent'],
            'bet_side': bet_side,
            'win_prob': win_prob,
            'confidence': home_confidence if bet_side == 'home' else away_confidence,
            'odds': original_odds,
            'decimal_odds': decimal_odds,
            'kelly_fraction': kelly_fraction,
            'expected_value': bet_ev,
            'bet_size': bet_size,
            'won': won,
            'profit': profit,
            'bankroll': bankroll,
            'home_win_prob': home_win_prob,
            'away_win_prob': away_win_prob,
            'home_odds': game['moneyLine'],
            'away_odds': game['opponentMoneyLine']
        })

        if idx % 500 == 0:  # Print detailed bet info every 500 games
            print(f"\nDetailed bet {idx}:")
            print(f"Date: {game['date']}")
            print(f"Match: {game['team']} vs {game['opponent']}")
            print(f"Bet Side: {bet_side}")
            print(f"Bet Size: ${bet_size:.2f}")
            print(f"Win Probability: {win_prob:.2%}")
            print(f"Decimal Odds: {decimal_odds:.2f}")
    
    # Calculate percentage of games skipped
    total_games = len(odds_df)
    skipped_percentage = (len(skipped_games) / total_games) * 100

    print("\nBacktest completed!")
    print(f"Processed {total_games} total games")
    print(f"Made bets on {total_bets} games")
    print(f"Skipped {len(skipped_games)} games ({skipped_percentage:.2f}%)")
    
    # Calculate performance metrics
    win_rate = winning_bets / total_bets if total_bets > 0 else 0
    roi = (total_profit / 1000) * 100  # ROI as percentage of initial bankroll
    max_drawdown = ((max_bankroll - min_bankroll) / max_bankroll) * 100
    
    # Create results DataFrame
    results_df = pd.DataFrame(bet_history)
    
    return {
        'total_bets': total_bets,
        'winning_bets': winning_bets,
        'win_rate': win_rate,
        'total_profit': total_profit,
        'roi': roi,
        'final_bankroll': bankroll,
        'max_bankroll': max_bankroll,
        'min_bankroll': min_bankroll,
        'max_drawdown': max_drawdown,
        'bet_history': results_df,
        'skipped_games': skipped_games,
        'skipped_percentage': skipped_percentage
    }

if __name__ == "__main__":
    print("Starting Kelly Criterion parameter test...")
    
    # Define test parameters
    test_params = [
        # Conservative approach
        {'kelly_fraction_multiplier': 0.25, 'test_name': '15% kelly No Threshold', 'max_bet_percentage': 0.1, 
         'use_thresholds': False, 'min_kelly_threshold': 0.08, 'min_ev_threshold': 0.12},
    
    ]
    
    results = []
    
    # Determine script directory for file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_file_path = os.path.join(script_dir, 'backtest_results.csv')
    
    # Load existing results if available
    existing_results = None
    if os.path.exists(results_file_path):
        try:
            existing_results = pd.read_csv(results_file_path)
            print(f"Loaded {len(existing_results)} existing test results")
        except Exception as e:
            print(f"Error loading existing results: {e}")
            existing_results = None
    
    # Test each parameter combination
    for i, params in enumerate(test_params):
        test_name = params.pop('test_name')  # Extract test name before passing params
        print(f"\nRunning test {i+1}: {test_name}")
        print(f"Parameters: {params}")
        
        # Create a param key for identification
        param_key = f"{params['max_bet_percentage']}_{int(params['use_thresholds'])}_{params['min_kelly_threshold']}_{params['min_ev_threshold']}_{params['kelly_fraction_multiplier']}"
        
        # Check if this combination already exists
        if existing_results is not None:
            # Create param keys for existing results
            existing_results['param_key'] = (
                existing_results['max_bet_percentage'].astype(str) + '_' +
                existing_results['use_thresholds'].astype(int).astype(str) + '_' +
                existing_results['min_kelly_threshold'].astype(str) + '_' +
                existing_results['min_ev_threshold'].astype(str) + '_' +
                existing_results['kelly_fraction_multiplier'].astype(str)
            )
            
            if param_key in existing_results['param_key'].values:
                print(f"Skipping test with param key {param_key} - already tested")
                continue
        
        results_data = backtest_kelly(
            model_path='best_model.joblib',
            odds_data_path='Model_Training/processed_odds_data.csv',
            stats_data_path='Model_Training/test_data.csv',
            **params
        )
        
        # Store key metrics and parameters
        results.append({
            'test_name': test_name,
            'total_bets': results_data['total_bets'],
            'win_rate': results_data['win_rate'],
            'roi': results_data['roi'],
            'max_drawdown': results_data['max_drawdown'],
            'sharpe_ratio': results_data['roi'] / (results_data['max_drawdown'] if results_data['max_drawdown'] > 0 else 1),
            'final_bankroll': results_data['final_bankroll'],
            'max_bet_percentage': params['max_bet_percentage'],
            'use_thresholds': params['use_thresholds'],
            'min_kelly_threshold': params['min_kelly_threshold'],
            'min_ev_threshold': params['min_ev_threshold'],
            'kelly_fraction_multiplier': params['kelly_fraction_multiplier'],
            'param_key': param_key,
            'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        # Save bet history for this test (using param key instead of index)
        history_file = os.path.join(script_dir, f'test_history_{param_key}.csv')
        results_data['bet_history'].to_csv(history_file, index=False)
        print(f"Saved bet history to {history_file}")
    
    # If we have new results to add
    if results:
        # Create summary DataFrame for new results
        new_results = pd.DataFrame(results)
        
        # Combine with existing results if available
        if existing_results is not None:
            # Drop the temporary param_key column from existing results
            if 'param_key' in existing_results.columns:
                existing_results = existing_results.drop(columns=['param_key'])
            
            # Combine existing and new results
            combined_results = pd.concat([existing_results, new_results], ignore_index=True)
        else:
            combined_results = new_results
        
        # Save combined results
        combined_results.to_csv(results_file_path, index=False)
        print(f"\nSaved {len(combined_results)} test results to {results_file_path}")
        
        # For display, drop param_key column
        display_results = combined_results.drop(columns=['param_key']) if 'param_key' in combined_results.columns else combined_results
        
        # Print summary of all results, sorted by ROI
        print("\nAll Backtest Results (sorted by ROI):")
        sorted_by_roi = display_results.sort_values('roi', ascending=False)
        print(sorted_by_roi[['test_name', 'total_bets', 'win_rate', 'roi', 'max_drawdown', 'sharpe_ratio']].head(10))
        
        # Print summary of all results, sorted by Sharpe ratio
        print("\nAll Backtest Results (sorted by Sharpe Ratio):")
        sorted_by_sharpe = display_results.sort_values('sharpe_ratio', ascending=False)
        print(sorted_by_sharpe[['test_name', 'total_bets', 'win_rate', 'roi', 'max_drawdown', 'sharpe_ratio']].head(10))
    else:
        print("\nNo new parameter combinations tested.")
    
    # If we have combined results, print best performers
    if 'combined_results' in locals():
        # Print best result by ROI
        best_roi = combined_results.loc[combined_results['roi'].idxmax()]
        print("\nBest Strategy by ROI:")
        print(f"Test Name: {best_roi['test_name']}")
        print(f"Parameters: max_bet={best_roi['max_bet_percentage']}, use_thresholds={best_roi['use_thresholds']}, "
              f"min_kelly={best_roi['min_kelly_threshold']}, min_ev={best_roi['min_ev_threshold']}, "
              f"kelly_multiplier={best_roi['kelly_fraction_multiplier']}")
        print(f"ROI: {best_roi['roi']:.2f}%")
        print(f"Win Rate: {best_roi['win_rate']:.2%}")
        print(f"Total Bets: {best_roi['total_bets']}")
        print(f"Max Drawdown: {best_roi['max_drawdown']:.2f}%")
        print(f"Final Bankroll: ${best_roi['final_bankroll']:.2f}")
        
        # Print best result by Sharpe ratio
        best_sharpe = combined_results.loc[combined_results['sharpe_ratio'].idxmax()]
        print("\nBest Strategy by Sharpe Ratio:")
        print(f"Test Name: {best_sharpe['test_name']}")
        print(f"Parameters: max_bet={best_sharpe['max_bet_percentage']}, use_thresholds={best_sharpe['use_thresholds']}, "
              f"min_kelly={best_sharpe['min_kelly_threshold']}, min_ev={best_sharpe['min_ev_threshold']}, "
              f"kelly_multiplier={best_sharpe['kelly_fraction_multiplier']}")
        print(f"Sharpe Ratio: {best_sharpe['sharpe_ratio']:.2f}")
        print(f"ROI: {best_sharpe['roi']:.2f}%")
        print(f"Win Rate: {best_sharpe['win_rate']:.2%}")
        print(f"Total Bets: {best_sharpe['total_bets']}")
        print(f"Max Drawdown: {best_sharpe['max_drawdown']:.2f}%") 