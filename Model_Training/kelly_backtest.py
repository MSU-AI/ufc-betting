import pandas as pd
import numpy as np
import joblib
import warnings
from feature_engineering import engineer_features
from datetime import datetime
from team_name_converter import convert_team_name
import json
from train_models import FFNClassifier


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

def backtest_kelly(model_path, odds_data_path, stats_data_path, max_bet_percentage=0.05):
    """Run backtesting using Kelly Criterion for bankroll management."""
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
        features = engineer_features(game_stats.copy())
        features = features.drop(columns=['target', 'team_abbreviation_home', 'team_abbreviation_away', 'game_id', 'date', 'team_id_home', 'team_id_away', 'season', 'wl_home'])
        
        # Get model prediction
        home_win_prob = model.predict_proba(features)[0][1]
        away_win_prob = 1 - home_win_prob  # Away team's probability is complement of home
        
        # Convert odds to decimal for both teams
        home_decimal_odds = convert_american_to_decimal(game['moneyLine'])
        away_decimal_odds = convert_american_to_decimal(game['opponentMoneyLine'])
        
        # Calculate Kelly fraction for both teams
        home_kelly_fraction = kelly_criterion(home_win_prob, home_decimal_odds)
        away_kelly_fraction = kelly_criterion(away_win_prob, away_decimal_odds)
        
        # Skip if Kelly fractions are below threshold
        min_kelly_threshold = 0.05
        if home_kelly_fraction < min_kelly_threshold and away_kelly_fraction < min_kelly_threshold:
            continue
            
        # Calculate bet size for both teams (capped at current max_bet)
        current_max_bet = bankroll * max_bet_percentage  # Dynamic max bet based on current bankroll
        home_bet_size = min(bankroll * home_kelly_fraction * 1/8, 20)
        away_bet_size = min(bankroll * away_kelly_fraction * 1/8, 20)
        # Skip if both bet sizes are too small
        if home_bet_size < 1 and away_bet_size < 1:
            continue
            
        # Determine which side to bet (choose the one with higher expected value)
        home_ev = home_win_prob * (home_decimal_odds - 1) - (1 - home_win_prob)
        away_ev = away_win_prob * (away_decimal_odds - 1) - (1 - away_win_prob)
        
        if home_ev > away_ev and home_bet_size >= 1:
            bet_side = 'home'
            bet_size = home_bet_size
            win_prob = home_win_prob
            decimal_odds = home_decimal_odds
            kelly_fraction = home_kelly_fraction
            original_odds = game['moneyLine']
        elif away_bet_size >= 1:
            bet_side = 'away'
            bet_size = away_bet_size
            win_prob = away_win_prob
            decimal_odds = away_decimal_odds
            kelly_fraction = away_kelly_fraction
            original_odds = game['opponentMoneyLine']
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
        
        # Record bet
        bet_history.append({
            'date': game['date'],
            'team': game['team'],
            'opponent': game['opponent'],
            'bet_side': bet_side,
            'win_prob': win_prob,
            'odds': original_odds,
            'decimal_odds': decimal_odds,
            'kelly_fraction': kelly_fraction,
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
    print("Starting Kelly Criterion backtest script...")
    
    # Run backtest
    results = backtest_kelly(
        model_path='best_model.joblib',
        odds_data_path='Model_Training/processed_odds_data.csv',
        stats_data_path='Model_Training/test_data.csv',
        max_bet_percentage=0.05
    )
    
    # Print results
    print("\nBacktesting Results:")
    print(f"Total Bets: {results['total_bets']}")
    print(f"Winning Bets: {results['winning_bets']}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Total Profit: ${results['total_profit']:.2f}")
    print(f"ROI: {results['roi']:.2f}%")
    print(f"Final Bankroll: ${results['final_bankroll']:.2f}")
    print(f"Max Bankroll: ${results['max_bankroll']:.2f}")
    print(f"Min Bankroll: ${results['min_bankroll']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Skipped Games: {len(results['skipped_games'])}")
    print(f"Skipped Percentage: {results['skipped_percentage']:.2f}%")
    
    # Save bet history to CSV
    results['bet_history'].to_csv('kelly_bet_history.csv', index=False)

    # Save skipped games to JSON
    with open('skipped_games.json', 'w') as f:
        json.dump(results['skipped_games'], f, indent=4) 