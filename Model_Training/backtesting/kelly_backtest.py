import pandas as pd
import numpy as np
import joblib
from feature_engineering import engineer_features
from datetime import datetime

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

def backtest_kelly(model_path, odds_data_path, stats_data_path, max_bet=20):
    """Run backtesting using Kelly Criterion for bankroll management."""
    # Load model and data
    model = joblib.load(model_path)
    odds_df = pd.read_csv(odds_data_path)
    stats_df = pd.read_csv(stats_data_path)
    
    # Convert date columns to datetime
    odds_df['date'] = pd.to_datetime(odds_df['date'])
    stats_df['date'] = pd.to_datetime(stats_df['date'])
    
    # Initialize tracking variables
    bankroll = 1000  # Starting bankroll
    max_bankroll = bankroll
    min_bankroll = bankroll
    total_bets = 0
    winning_bets = 0
    total_profit = 0
    bet_history = []
    
    # Sort data by date
    odds_df = odds_df.sort_values('date')
    stats_df = stats_df.sort_values('date')
    
    # Process each game
    for _, game in odds_df.iterrows():
        # Find corresponding stats for this game
        game_stats = stats_df[
            (stats_df['date'] == game['date']) & 
            (stats_df['team_id_home'] == game['team']) & 
            (stats_df['team_id_away'] == game['opponent'])
        ]
        
        if game_stats.empty:
            continue
            
        # Prepare features for prediction
        features = engineer_features(game_stats.copy())
        features = features.drop(columns=['target', 'game_id', 'date', 'team_id_home', 'team_id_away', 'season', 'wl_home'])
        
        # Get model prediction
        win_prob = model.predict_proba(features)[0][1]
        
        # Convert odds to decimal
        decimal_odds = convert_american_to_decimal(game['moneyLine'])
        
        # Calculate Kelly fraction
        kelly_fraction = kelly_criterion(win_prob, decimal_odds)
        
        # Calculate bet size (capped at max_bet)
        bet_size = min(bankroll * kelly_fraction, max_bet)
        
        # Skip if bet size is too small
        if bet_size < 1:
            continue
            
        # Determine if bet won
        won = (game['score'] > game['opponentScore'])
        
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
            'win_prob': win_prob,
            'odds': game['moneyLine'],
            'decimal_odds': decimal_odds,
            'kelly_fraction': kelly_fraction,
            'bet_size': bet_size,
            'won': won,
            'profit': profit,
            'bankroll': bankroll
        })
    
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
        'bet_history': results_df
    }

if __name__ == "__main__":
    # Run backtest
    results = backtest_kelly(
        model_path='best_model.joblib',
        odds_data_path='oddsData.csv',
        stats_data_path='team_game_stats_season.csv',
        max_bet=20
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
    
    # Save bet history to CSV
    results['bet_history'].to_csv('kelly_bet_history.csv', index=False) 