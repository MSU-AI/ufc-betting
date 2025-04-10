import sys
import os
from pathlib import Path

# Add parent directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve, log_loss
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import joblib
from utils.feature_engineering import engineer_features
import json
from datetime import datetime
import optuna
from optuna.integration import XGBoostPruningCallback
from temperature_scaling import ModelWithTemperature
from sklearn.feature_selection import RFE

def load_data(filepaths, basic_features_only=False):
    """
    Load and combine data from multiple window lengths
    
    Parameters:
    -----------
    filepaths : dict
        Dictionary mapping window names to file paths
        e.g. {'season': 'path/to/season.csv', '5_game': 'path/to/5game.csv'}
    basic_features_only : bool
        Whether to use only basic features or engineer additional features
    """
    dfs = {}
    
    # Load and engineer features for each window's data separately
    for window_name, filepath in filepaths.items():
        df = pd.read_csv(filepath)
        
        if not basic_features_only:
            # Engineer features for each window separately
            df = engineer_features(df)
        
        # Add window prefix to all columns except those that should remain unchanged
        unchanged_cols = ['target', 'team_abbreviation_home', 'team_abbreviation_away', 
                         'game_id', 'date', 'team_id_home', 'team_id_away', 
                         'season', 'wl_home']
        
        rename_cols = {col: f'{window_name}_{col}' 
                      for col in df.columns 
                      if col not in unchanged_cols}
        
        df = df.rename(columns=rename_cols)
        dfs[window_name] = df

    # Start with the first dataframe
    first_window = list(filepaths.keys())[0]
    df = dfs[first_window].copy()
    
    # Merge with remaining windows
    merge_cols = ['game_id', 'date', 'team_id_home', 'team_id_away', 
                 'team_abbreviation_home', 'team_abbreviation_away', 
                 'season', 'wl_home']
    
    remaining_windows = list(filepaths.keys())[1:]
    for window_name in remaining_windows:
        df = df.merge(dfs[window_name], on=merge_cols, suffixes=('', f'_{window_name}'))

    # Convert win/loss to binary
    df['target'] = (df['wl_home'] == 'W').astype(int)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Split into train/val/test datasets
    train_data = df[df['date'] < '2020-12-22']
    val_data = df[(df['date'] >= '2020-12-22') & (df['date'] < '2021-10-01')]
    test_data = df[df['date'] >= '2021-10-01']
    
    # Drop rows with missing values
    train_data = train_data.dropna(subset=train_data.columns.tolist())
    val_data = val_data.dropna(subset=val_data.columns.tolist())
    test_data = test_data.dropna(subset=test_data.columns.tolist())

    # Create feature matrix and target vector
    drop_cols = ['target', 'team_abbreviation_home', 'team_abbreviation_away', 
                'game_id', 'date', 'team_id_home', 'team_id_away', 'season', 'wl_home']
    X_train = train_data.drop(columns=drop_cols)
    y_train = train_data['target']
    X_val = val_data.drop(columns=drop_cols)
    y_val = val_data['target']
    X_test = test_data.drop(columns=drop_cols)
    y_test = test_data['target']

    print(f"\nUsing {'basic' if basic_features_only else 'all'} features:")
    print(f"Number of features: {len(X_train.columns)}")
    print("Features used:", X_train.columns.tolist())
    
    return X_train, y_train, X_val, y_val, X_test, y_test, test_data

class FFNClassifier(nn.Module):
    def __init__(self, input_dim, architecture):
        super(FFNClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in architecture:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        # Output layer just produces logits
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x, return_probas=False):
        logits = self.network(x)
        # Only apply softmax when explicitly requested (for predictions)
        if return_probas:
            return torch.softmax(logits, dim=1)
        return logits
    
    def predict_proba(self, X):
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        self.eval()
        with torch.no_grad():
            # Explicitly request probabilities here
            return self.forward(X, return_probas=True).numpy()

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, input_dim),
            nn.LayerNorm(input_dim)
        )
        
    def forward(self, x):
        return x + self.block(x)

class ResFFNClassifier(nn.Module):
    def __init__(self, input_dim, architecture):
        super(ResFFNClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        
        # Input projection if needed
        if prev_dim != architecture[0]:
            self.input_proj = nn.Linear(prev_dim, architecture[0])
            prev_dim = architecture[0]
        else:
            self.input_proj = nn.Identity()
            
        # Create residual blocks
        for hidden_dim in architecture:
            # Projection layer if dimensions don't match
            if prev_dim != hidden_dim:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.ReLU())
            
            # Add residual block
            layers.append(ResidualBlock(hidden_dim, hidden_dim))
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        # Output layer just produces logits
        self.output = nn.Linear(prev_dim, 2)
    
    def forward(self, x, return_probas=False):
        x = self.input_proj(x)
        x = self.network(x)
        logits = self.output(x)
        # Only apply softmax when explicitly requested
        if return_probas:
            return torch.softmax(logits, dim=1)
        return logits
    
    def predict_proba(self, X):
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        self.eval()
        with torch.no_grad():
            # Explicitly request probabilities here
            return self.forward(X, return_probas=True).numpy()


def train_neural_net(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    # Convert DataFrames to numpy arrays first, then to PyTorch tensors
    X_train = torch.FloatTensor(X_train.to_numpy())
    y_train = torch.LongTensor(y_train.values)
    X_val = torch.FloatTensor(X_val.to_numpy())
    y_val = torch.LongTensor(y_val.values)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    valid_dataset = TensorDataset(X_val, y_val)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Training loop
    best_val_loss = float('inf')
    best_state = None
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            # Get logits (not probabilities) for training
            logits = model(batch_X)
            # CrossEntropyLoss expects logits
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            # Get logits (not probabilities) for validation
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model state
    model.load_state_dict(best_state)
    
    valid_dataset = TensorDataset(X_val, y_val)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    #model = ModelWithTemperature(model)
    #model.set_temperature(valid_loader)
    
    return model

def objective(trial, X_train, y_train, X_val, y_val):
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }
    
    # Create and train model using sklearn API
    model = xgb.XGBClassifier(**param)
    model.fit(
        X_train, 
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Get validation score
    val_pred = model.predict_proba(X_val)[:, 1]
    val_score = log_loss(y_val, val_pred)
    
    return val_score

def train_evaluate_models(X_train, y_train, X_val, y_val, X_test, y_test):
    # Remove scaling since features are already scaled
    X_train_scaled = X_train
    X_val_scaled = X_val
    X_test_scaled = X_test
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        #'XGBoost': None,  # Will be set after tuning
        'FFN_Small': FFNClassifier(X_train.shape[1], [64, 32]),
        'FFN_Medium': FFNClassifier(X_train.shape[1], [128, 64, 32]),
        'FFN_Large': FFNClassifier(X_train.shape[1], [256, 128, 64, 32]),
    }
    
    '''
    
    # Optimize XGBoost hyperparameters using Optuna
    print("\nOptimizing XGBoost hyperparameters...")
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    
    study.optimize(
        lambda trial: objective(trial, X_train_scaled, y_train, X_val_scaled, y_val),
        n_trials=50,
        timeout=3600
    )
    
    # Get best parameters and train final model
    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['eval_metric'] = 'logloss'
    
    print("\nBest XGBoost parameters:", best_params)
    print("Best validation log loss:", study.best_value)
    
    # Train final XGBoost model with best parameters
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
    dval = xgb.DMatrix(X_val_scaled, label=y_val)
    dtest = xgb.DMatrix(X_test_scaled, label=y_test)
    
    # Remove n_estimators from params and use as num_boost_round
    num_boost_round = best_params.pop('n_estimators')
    
    # Train XGBoost model using train method
    best_model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dval, 'validation')],
        verbose_eval=False
    )
    
    models['XGBoost'] = best_model
    '''
    # Train and calibrate models
    calibrated_models = {}
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        if isinstance(model, FFNClassifier) or isinstance(model, ResFFNClassifier):
            # Train neural network
            model = train_neural_net(model, X_train_scaled, y_train, X_val_scaled, y_val)
            # Add neural network to calibrated_models
            calibrated_models[name] = model
            # Get predictions
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test_scaled.to_numpy())
                # Take only the probability for class 1 (index 1)
                y_pred_proba = model(X_test_tensor, return_probas=True).numpy()[:, 1]
        else:
            # Handle XGBoost model differently than other models
            if name == 'XGBoost':
                # XGBoost model is already trained
                y_pred_proba = model.predict(dtest)
            else:
                # Train and calibrate traditional models
                model.fit(X_train_scaled, y_train)
                calibrated = CalibratedClassifierCV(model)
                calibrated.fit(X_train_scaled, y_train)
                calibrated_models[name] = calibrated
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
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), height_ratios=[1, 2])
    fig.suptitle('Model Calibration Analysis', fontsize=12)
    
    # Plot prediction distribution histogram
    for name, result in results.items():
        ax1.hist(result['y_pred_proba'], 
                bins=50, 
                density=True, 
                alpha=0.3, 
                label=name)
    ax1.set_xlabel('Predicted probability')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution of Predictions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot calibration curves
    for name, result in results.items():
        prob_true, prob_pred = calibration_curve(
            result['y_test'],
            result['y_pred_proba'],
            n_bins=10,
            strategy='quantile'  # Use quantile binning for more reliable curves
        )
        ax2.plot(prob_pred, 
                prob_true, 
                marker='o', 
                linewidth=2,
                label=f'{name} (Brier: {result["brier_score"]:.3f})')
    
    # Add reference line
    ax2.plot([0, 1], [0, 1], 
            linestyle='--', 
            color='gray', 
            label='Perfectly calibrated')
    
    ax2.set_xlabel('Mean predicted probability')
    ax2.set_ylabel('True probability')
    ax2.set_title('Calibration Curves')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
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

def save_model_metrics(best_model_name, best_auc_score):
    """Save best model type and AUC score"""
    metrics = {
        'model_type': best_model_name,
        'auc_score': float(best_auc_score),
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

def main():
    # Load data
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = {
        #'season': os.path.join(base_path, 'team_game_stats_season.csv'),
        #'3_game': os.path.join(base_path, 'team_game_stats_3game.csv'),
        #'5_game': os.path.join(base_path, 'team_game_stats_5game.csv'),
        '10_game': os.path.join(base_path, 'team_game_stats_10game.csv')
    }
    X_train, y_train, X_val, y_val, X_test, y_test, test_data = load_data(data_path, basic_features_only=False)
    
    # Train and evaluate models
    calibrated_models, results = train_evaluate_models(X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Print results
    print("\nModel Performance:")
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"Brier Score: {result['brier_score']:.3f}")
        print(f"AUC Score: {result['auc_score']:.2f}")
        print(f"Log Loss: {result['log_loss']:.3f}")
    
    # Plot performance curves
    plot_calibration_curves(results)
    plot_roc_curves(results)
    plot_log_loss(results)
    
    # Select best model based on AUC score
    best_model_name = max(results.items(), key=lambda x: x[1]['auc_score'])[0]
    best_auc_score = results[best_model_name]['auc_score']
    
    # Save metrics
    save_model_metrics(best_model_name, best_auc_score)
    
    # Save best model
    best_model = calibrated_models[best_model_name]
    joblib.dump(best_model, 'best_model.joblib')
    print(f"\nSaved best model ({best_model_name}) to best_model.joblib")
    
    #save test data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_data.to_csv(os.path.join(current_dir, 'test_data.csv'), index=False)

if __name__ == "__main__":
    main() 