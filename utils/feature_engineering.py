import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def create_rolling_features(df, window_sizes=[3, 5, 10]):
    """Create rolling window features for recent performance."""
    # Sort by date to ensure correct rolling calculations
    df['date'] = pd.to_datetime(df['date'])

    
    rolling_features = []
    
    for window in window_sizes:
        # Create rolling win percentage features using shifted target
        df['home_winpct_last_{}'.format(window)] = df.groupby(['team_id_home', 'season'])['target'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        
        df['away_winpct_last_{}'.format(window)] = df.groupby(['team_id_away', 'season'])['target'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        
        rolling_features.extend([
            'home_winpct_last_{}'.format(window),
            'away_winpct_last_{}'.format(window)
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

def create_efficiency_metrics(df):
    """Calculate efficiency-based metrics."""
    # Offensive efficiency (points per shooting percentage)
    df['home_off_eff'] = df['home_avg_pts'] / (df['home_avg_fg_pct'] * 100)
    df['away_off_eff'] = df['away_avg_pts'] / (df['away_avg_fg_pct'] * 100)
    df['off_eff_diff'] = df['home_off_eff'] - df['away_off_eff']
    
    # True shooting percentage
    df['home_ts_pct'] = df['home_avg_pts'] / (2 * (df['home_avg_fg_pct'] * 100 + 0.44 * df['home_avg_ft_pct'] * 100))
    df['away_ts_pct'] = df['away_avg_pts'] / (2 * (df['away_avg_fg_pct'] * 100 + 0.44 * df['away_avg_ft_pct'] * 100))
    df['ts_pct_diff'] = df['home_ts_pct'] - df['away_ts_pct']
    
    return ['home_off_eff', 'away_off_eff', 'off_eff_diff',
            'home_ts_pct', 'away_ts_pct', 'ts_pct_diff']

def create_performance_differentials(df):
    """Calculate differentials between home and away team stats."""
    # Basic stat differentials
    df['pts_diff'] = df['home_avg_pts'] - df['away_avg_pts']
    df['reb_diff'] = df['home_avg_reb'] - df['away_avg_reb']
    df['ast_diff'] = df['home_avg_ast'] - df['away_avg_ast']
    df['stl_diff'] = df['home_avg_stl'] - df['away_avg_stl']
    df['blk_diff'] = df['home_avg_blk'] - df['away_avg_blk']
    
    # Shooting percentage differentials
    df['fg_pct_diff'] = df['home_avg_fg_pct'] - df['away_avg_fg_pct']
    df['fg3_pct_diff'] = df['home_avg_fg3_pct'] - df['away_avg_fg3_pct']
    df['ft_pct_diff'] = df['home_avg_ft_pct'] - df['away_avg_ft_pct']
    
    return ['pts_diff', 'reb_diff', 'ast_diff', 'stl_diff', 'blk_diff',
            'fg_pct_diff', 'fg3_pct_diff', 'ft_pct_diff']

def create_composite_metrics(df):
    """Create composite performance metrics."""
    # Defensive rating (blocks + steals)
    df['home_def_rating'] = df['home_avg_blk'] + df['home_avg_stl']
    df['away_def_rating'] = df['away_avg_blk'] + df['away_avg_stl']
    df['def_rating_diff'] = df['home_def_rating'] - df['away_def_rating']
    
    # Overall performance rating
    df['home_performance'] = (
        df['home_avg_pts'] * 1.0 +
        df['home_avg_reb'] * 0.4 +
        df['home_avg_ast'] * 0.3 +
        df['home_def_rating'] * 0.3
    )
    
    df['away_performance'] = (
        df['away_avg_pts'] * 1.0 +
        df['away_avg_reb'] * 0.4 +
        df['away_avg_ast'] * 0.3 +
        df['away_def_rating'] * 0.3
    )
    
    df['performance_diff'] = df['home_performance'] - df['away_performance']
    
    return ['home_def_rating', 'away_def_rating', 'def_rating_diff',
            'home_performance', 'away_performance', 'performance_diff']

def scale_features(df):
    """Scale numerical features using StandardScaler."""
    # Identify columns to scale (exclude non-numeric and target columns)
    cols_to_scale = df.select_dtypes(include=['float64', 'int64']).columns
    cols_to_scale = [col for col in cols_to_scale if col != 'target']
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Scale features
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    return df

def engineer_features(df):
    """Main function to engineer all features."""
    create_rolling_features(df)
    create_offensive_ratings(df)
    create_efficiency_differences(df)
    create_basic_differentials(df)
    create_defensive_metrics(df)
    create_shooting_efficiency(df)
    create_performance_metrics(df)
    create_efficiency_metrics(df)
    create_performance_differentials(df)
    create_composite_metrics(df)
    
    # Fill NA with 0 before scaling
    df = df.fillna(0)
    
    # Scale features as final step
    df = scale_features(df)
    
    return df