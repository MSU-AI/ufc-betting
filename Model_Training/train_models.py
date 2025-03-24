import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve, log_loss
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib
from feature_engineering import engineer_features
import json
from datetime import datetime
import os
import mlflow
import mlflow.sklearn
from itertools import combinations 
from pprint import pprint

def load_data(filepaths, basic_features_only=False):
    """
    Load and combine multiple datasets horizontally
    
    Args:
        filepaths (list): List of paths to CSV files
        basic_features_only (bool): Whether to use only basic features
    """
    # Load and process each dataset
    dataframes = []
    for filepath in filepaths:
        df = pd.read_csv(filepath)
        
        # Convert win/loss to binary
        df['target'] = (df['wl_home'] == 'W').astype(int)
        
        # Basic features (season averages)
        base_features = [
            'home_avg_pts', 'home_avg_reb', 'home_avg_ast', 'home_avg_stl', 
            'home_avg_blk', 'home_avg_fg_pct', 'home_avg_fg3_pct', 'home_avg_ft_pct',
            'away_avg_pts', 'away_avg_reb', 'away_avg_ast', 'away_avg_stl',
            'away_avg_blk', 'away_avg_fg_pct', 'away_avg_fg3_pct', 'away_avg_ft_pct'
        ]
        
        if not basic_features_only:
            # Engineer new features
            df = engineer_features(df)
            
        dataframes.append(df)
    
    # Go through dataframes and append suffix based on filename
    renamed_dfs = []
    if len(dataframes) > 1:
        for i, df in enumerate(dataframes):
            filepath = filepaths[i]
            
        # Determine suffix from filename
        if '3game' in filepath:
            suffix = '_3g'
        elif '5game' in filepath:
            suffix = '_5g'
        elif '10game' in filepath:
            suffix = '_10g'
        elif 'whole' in filepath:
            suffix = '_whole'
        else:
            raise ValueError(f"Unable to determine time window from filename: {filepath}")
        
        # Rename all columns except target (which we'll handle separately)
        rename_cols = {col: f"{col}{suffix}" for col in df.columns if col != 'target'}
        df = df.rename(columns=rename_cols)
        renamed_dfs.append(df)
    
    # Keep target column only from the first dataframe
    target = renamed_dfs[0]['target']
    
    # Drop target from all dataframes before combining
    feature_dfs = [df.drop(columns=['target']) for df in renamed_dfs]
    
    # Combine all dataframes horizontally
    combined_df = pd.concat(feature_dfs, axis=1)
    
    # Get indices of rows that have no missing values
    valid_indices = combined_df.dropna().index
    
    # Drop rows with missing values from features
    combined_df = combined_df.dropna()
    
    # Drop the same rows from target using the valid indices
    target = target.loc[valid_indices]
    
    # Create feature matrix and target vector
    columns_to_drop = []
    for col in combined_df.columns:
        if any(word in col for word in ['target', 'game_id', 'date', 'team_id', 'season', 'wl']):
            columns_to_drop.append(col)
    
    X = combined_df.drop(columns=columns_to_drop)
    y = target  # Now target will have the same rows as X
    
    print(f"\nUsing {'basic' if basic_features_only else 'all'} features:")
    print(f"Number of features: {len(X.columns)}")
    print("Features used:", X.columns.tolist())
    print(f"Total number of samples: {len(X)}")
    
    return X, y

def train_evaluate_models(X, y, selected_windows):
    """Train and evaluate models without MLflow tracking"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'XGBoost': None
    }
    
    # XGBoost hyperparameter tuning
    xgb_params = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200],
        'min_child_weight': [1, 3],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }
    
    # Tune XGBoost
    base_xgb = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    grid_search_xgb = GridSearchCV(
        estimator=base_xgb,
        param_grid=xgb_params,
        cv=5,
        scoring='neg_log_loss',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search_xgb.fit(X_train_scaled, y_train)
    print("\nBest XGBoost parameters:", grid_search_xgb.best_params_)
    models['XGBoost'] = grid_search_xgb.best_estimator_
    
    calibrated_models = {}
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train and calibrate model
        model.fit(X_train_scaled, y_train)
        calibrated = CalibratedClassifierCV(model)
        calibrated.fit(X_train_scaled, y_train)
        calibrated_models[name] = calibrated
        
        # Get predictions
        y_pred_proba = calibrated.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        brier = brier_score_loss(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        logloss = log_loss(y_test, y_pred_proba)
        
        results[name] = {
            'brier_score': brier,
            'auc_score': auc,
            'log_loss': logloss,
            'y_test': y_test,
            'y_pred_proba': y_pred_proba
        }
    
    
    return calibrated_models, results

def plot_calibration_curves(results):
    plt.figure(figsize=(10, 6))
    
    for name, result in results.items():
        prob_true, prob_pred = calibration_curve(result['y_test'], 
                                               result['y_pred_proba'], 
                                               n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=f'{name} (Brier: {result["brier_score"]:.3f})')
    
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('True probability')
    plt.title('Calibration Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('calibration_curves.png')
    plt.close()

def plot_roc_curves(results):
    plt.figure(figsize=(10, 6))
    
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(result['y_test'], result['y_pred_proba'])
        plt.plot(fpr, tpr, label=f'{name} (AUC: {result["auc_score"]})')
    
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('roc_curves.png')
    plt.close()

def plot_log_loss(results):
    plt.figure(figsize=(10, 6))
    
    for name, result in results.items():
        log_loss_score = log_loss(result['y_test'], result['y_pred_proba'])
        plt.bar(name, log_loss_score)
        plt.text(name, log_loss_score, f'{log_loss_score:.3f}', 
                ha='center', va='bottom')
    
    plt.ylabel('Log Loss')
    plt.title('Model Log Loss Comparison')
    plt.grid(True, axis='y')
    plt.savefig('log_loss.png')
    plt.close()

def save_model_metrics(best_model_name, best_brier_score):
    """Save best model type and Brier score"""
    metrics = {
        'model_type': best_model_name,
        'brier_score': float(best_brier_score),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('model_performance.json', 'w') as w:
        json.dump(metrics, w, indent=4)

def save_model(calibrated_model, feature_names):
    """Save calibrated model and feature names using joblib"""
    model_data = {
        'model': calibrated_model,
        'feature_names': feature_names
    }
    joblib.dump(model_data, 'calibrated_model.joblib')

def try_window_combinations():
    """Try all possible combinations of window sizes and return results"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    available_files = {
        '3game': os.path.join(base_dir, 'team_game_stats_3game.csv'),
        '5game': os.path.join(base_dir, 'team_game_stats_5game.csv'),
        '10game': os.path.join(base_dir, 'team_game_stats_10game.csv'),
        'whole': os.path.join(base_dir, 'team_game_stats_whole.csv')
    }
    
    windows = ['3game', '5game', '10game', 'whole']
    all_results = {}
    
    # Try combinations of all sizes from 2 to 4
    for size in range(1, len(windows) + 1):
        for combo in combinations(windows, size):
            print(f"\nTrying combination: {combo}")
            selected_files = [available_files[window] for window in combo]
            X, y = load_data(selected_files, basic_features_only=False)
            calibrated_models, results = train_evaluate_models(X, y, combo)
            all_results[combo] = [results['Logistic Regression']['brier_score'], results['XGBoost']['brier_score']]
    
    print(all_results)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    #train model using only whole season data
    X, y = load_data([os.path.join(base_dir, 'team_game_stats_whole.csv')], basic_features_only=False)
    calibrated_models, results = train_evaluate_models(X, y, ['whole'])
    plot_calibration_curves(results)
    plot_roc_curves(results)
    plot_log_loss(results)
    pprint(results)

if __name__ == "__main__":
    main() 