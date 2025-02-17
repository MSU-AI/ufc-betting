import pandas as pd
import numpy as np

def create_rolling_features(df, window_sizes=[3, 5, 10]):
    """Create rolling window features for recent performance."""
    # Sort by date to ensure correct rolling calculations
    df['date'] = pd.to_datetime(df['date'])
    
    # Base stats to create rolling features for
    base_stats = {
        'pts': ['home_avg_pts', 'away_avg_pts'],
        'reb': ['home_avg_reb', 'away_avg_reb'],
        'ast': ['home_avg_ast', 'away_avg_ast'],
        'fg_pct': ['home_avg_fg_pct', 'away_avg_fg_pct'],
        'fg3_pct': ['home_avg_fg3_pct', 'away_avg_fg3_pct']
    }
    
    rolling_features = []
    
    for window in window_sizes:
        for stat, (home_col, away_col) in base_stats.items():
            # Create home team rolling features
            home_roll_col = f'home_{stat}_last_{window}'
            df[home_roll_col] = df.groupby(['team_id_home', 'season'])[home_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
            # Create away team rolling features
            away_roll_col = f'away_{stat}_last_{window}'
            df[away_roll_col] = df.groupby(['team_id_away', 'season'])[away_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
            # Create differential features
            diff_col = f'{stat}_diff_last_{window}'
            df[diff_col] = df[home_roll_col] - df[away_roll_col]
            
            rolling_features.extend([home_roll_col, away_roll_col, diff_col])
            
        # Create rolling win percentage features
        df['home_winpct_last_{}'.format(window)] = df.groupby(['team_id_home', 'season'])['target'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        
        df['away_winpct_last_{}'.format(window)] = df.groupby(['team_id_away', 'season'])['target'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        
        rolling_features.extend([
            'home_winpct_last_{}'.format(window),
            'away_winpct_last_{}'.format(window)
        ])
        
        # Create momentum features (trend in performance)
        df['home_momentum_{}'.format(window)] = df.groupby(['team_id_home', 'season'])['home_avg_pts'].transform(
            lambda x: x.rolling(window=window, min_periods=1).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            )
        )
        
        df['away_momentum_{}'.format(window)] = df.groupby(['team_id_away', 'season'])['away_avg_pts'].transform(
            lambda x: x.rolling(window=window, min_periods=1).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            )
        )
        
        rolling_features.extend([
            'home_momentum_{}'.format(window),
            'away_momentum_{}'.format(window)
        ])
    
    return rolling_features

def create_offensive_ratings(df):
    """Calculate offensive ratings for home and away teams."""
    df['home_off_rating'] = df['home_avg_pts'] / (df['home_avg_fg_pct'] * 100)
    df['away_off_rating'] = df['away_avg_pts'] / (df['away_avg_fg_pct'] * 100)
    return ['home_off_rating', 'away_off_rating']

def create_efficiency_differences(df):
    """Calculate shooting efficiency differentials."""
    df['fg_pct_diff'] = df['home_avg_fg_pct'] - df['away_avg_fg_pct']
    df['fg3_pct_diff'] = df['home_avg_fg3_pct'] - df['away_avg_fg3_pct']
    df['ft_pct_diff'] = df['home_avg_ft_pct'] - df['away_avg_ft_pct']
    return ['fg_pct_diff', 'fg3_pct_diff', 'ft_pct_diff']

def create_basic_differentials(df):
    """Calculate basic stat differentials."""
    df['pts_diff'] = df['home_avg_pts'] - df['away_avg_pts']
    df['reb_diff'] = df['home_avg_reb'] - df['away_avg_reb']
    df['ast_diff'] = df['home_avg_ast'] - df['away_avg_ast']
    return ['pts_diff', 'reb_diff', 'ast_diff']

def create_defensive_metrics(df):
    """Calculate defensive impact metrics."""
    df['def_impact_home'] = df['home_avg_stl'] + df['home_avg_blk']
    df['def_impact_away'] = df['away_avg_stl'] + df['away_avg_blk']
    df['def_impact_diff'] = df['def_impact_home'] - df['def_impact_away']
    return ['def_impact_home', 'def_impact_away', 'def_impact_diff']

def create_shooting_efficiency(df):
    """Calculate composite shooting efficiency metrics."""
    df['home_shooting_eff'] = (df['home_avg_fg_pct'] * 2 + 
                              df['home_avg_fg3_pct'] * 3 + 
                              df['home_avg_ft_pct']) / 6
    df['away_shooting_eff'] = (df['away_avg_fg_pct'] * 2 + 
                              df['away_avg_fg3_pct'] * 3 + 
                              df['away_avg_ft_pct']) / 6
    df['shooting_eff_diff'] = df['home_shooting_eff'] - df['away_shooting_eff']
    return ['home_shooting_eff', 'away_shooting_eff', 'shooting_eff_diff']

def create_performance_metrics(df):
    """Calculate overall performance metrics."""
    df['home_performance'] = (df['home_avg_pts'] * 1.0 + 
                            df['home_avg_reb'] * 0.4 + 
                            df['home_avg_ast'] * 0.3 + 
                            df['def_impact_home'] * 0.3)
    
    df['away_performance'] = (df['away_avg_pts'] * 1.0 + 
                            df['away_avg_reb'] * 0.4 + 
                            df['away_avg_ast'] * 0.3 + 
                            df['def_impact_away'] * 0.3)
    
    df['performance_diff'] = df['home_performance'] - df['away_performance']
    return ['home_performance', 'away_performance', 'performance_diff']

def engineer_features(df):
    """
    Engineer additional features for team performance analysis and ensure all stats are floats
    Returns both the modified DataFrame and list of engineered feature columns
    """
    engineered_features = []
    
    # First ensure all numeric columns are float
    numeric_columns = [
        'home_avg_pts', 'away_avg_pts',
        'home_avg_reb', 'away_avg_reb',
        'home_avg_ast', 'away_avg_ast',
        'home_avg_stl', 'away_avg_stl',
        'home_avg_blk', 'away_avg_blk',
        'home_avg_fg_pct', 'away_avg_fg_pct',
        'home_avg_fg3_pct', 'away_avg_fg3_pct',
        'home_avg_ft_pct', 'away_avg_ft_pct'
    ]
    engineered_features.extend(numeric_columns)
    
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
    
    # Create rolling window features (last N games)
    windows = [3, 5, 10]
    
    for window in windows:
        # Rolling averages for each team
        win_rate_col = f'home_win_rate_{window}g'
        df[win_rate_col] = df.groupby('team_id_home')['wl_home'].transform(
            lambda x: pd.to_numeric(x == 'W', errors='coerce').rolling(window, min_periods=1).mean()
        ).astype(float)
        engineered_features.append(win_rate_col)
        
        # Offensive features
        home_eff_col = f'home_scoring_efficiency_{window}g'
        away_eff_col = f'away_scoring_efficiency_{window}g'
        
        df[home_eff_col] = (
            df.groupby('team_id_home')['home_avg_pts'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            ) / df.groupby('team_id_home')['home_avg_fg_pct'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
        ).astype(float)
        
        df[away_eff_col] = (
            df.groupby('team_id_away')['away_avg_pts'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            ) / df.groupby('team_id_away')['away_avg_fg_pct'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
        ).astype(float)
        
        engineered_features.extend([home_eff_col, away_eff_col])
        
        # Defensive features
        home_def_col = f'home_defensive_rating_{window}g'
        away_def_col = f'away_defensive_rating_{window}g'
        
        df[home_def_col] = (
            df.groupby('team_id_home')['home_avg_stl'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            ) + df.groupby('team_id_home')['home_avg_blk'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
        ).astype(float)
        
        df[away_def_col] = (
            df.groupby('team_id_away')['away_avg_stl'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            ) + df.groupby('team_id_away')['away_avg_blk'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
        ).astype(float)
        
        engineered_features.extend([home_def_col, away_def_col])
    
    # Head-to-head features
    df['h2h_wins'] = df.groupby(['team_id_home', 'team_id_away'])['wl_home'].transform(
        lambda x: pd.to_numeric(x == 'W', errors='coerce').expanding().mean()
    ).astype(float)
    engineered_features.append('h2h_wins')
    
    # Home/Away performance
    df['home_advantage'] = df.groupby('team_id_home')['wl_home'].transform(
        lambda x: pd.to_numeric(x == 'W', errors='coerce').expanding().mean()
    ).astype(float)
    
    df['away_disadvantage'] = df.groupby('team_id_away')['wl_home'].transform(
        lambda x: pd.to_numeric(x == 'L', errors='coerce').expanding().mean()
    ).astype(float)
    
    engineered_features.extend(['home_advantage', 'away_disadvantage'])
    
    # Momentum features
    df['home_momentum'] = df.groupby('team_id_home')['wl_home'].transform(
        lambda x: pd.to_numeric(x == 'W', errors='coerce').rolling(3, min_periods=1).sum()
    ).astype(float)
    
    df['away_momentum'] = df.groupby('team_id_away')['wl_home'].transform(
        lambda x: pd.to_numeric(x == 'L', errors='coerce').rolling(3, min_periods=1).sum()
    ).astype(float)
    
    engineered_features.extend(['home_momentum', 'away_momentum'])
    
    # Shooting efficiency differential
    shooting_diffs = ['fg_pct_diff', 'fg3_pct_diff', 'ft_pct_diff']
    df['fg_pct_diff'] = (df['home_avg_fg_pct'] - df['away_avg_fg_pct']).astype(float)
    df['fg3_pct_diff'] = (df['home_avg_fg3_pct'] - df['away_avg_fg3_pct']).astype(float)
    df['ft_pct_diff'] = (df['home_avg_ft_pct'] - df['away_avg_ft_pct']).astype(float)
    engineered_features.extend(shooting_diffs)
    
    # Overall team efficiency
    df['home_efficiency'] = (
        (df['home_avg_pts'] + df['home_avg_ast'] + df['home_avg_reb']) /
        (df['home_avg_fg_pct'] + df['home_avg_fg3_pct'] + df['home_avg_ft_pct'])
    ).astype(float)
    
    df['away_efficiency'] = (
        (df['away_avg_pts'] + df['away_avg_ast'] + df['away_avg_reb']) /
        (df['away_avg_fg_pct'] + df['away_avg_fg3_pct'] + df['away_avg_ft_pct'])
    ).astype(float)
    
    engineered_features.extend(['home_efficiency', 'away_efficiency'])
    
    # Fill any NaN values with 0
    df = df.fillna(0.0)
    
    # Final check to ensure all numeric columns are float
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols] = df[numeric_cols].astype(float)
    
    return engineered_features 