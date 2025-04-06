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
from feature_engineering import engineer_features
import json
from datetime import datetime
import os
import optuna
from optuna.integration import XGBoostPruningCallback

def load_data(filepath, basic_features_only=False):
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
    
    #convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # defining train dataset - use data before 2021 season start (Dec 22, 2020)
    train_data = df[df['date'] < '2020-12-22']
    
    # defining val dataset - 2021 season (Dec 22, 2020 - end of season)
    val_data = df[(df['date'] >= '2020-12-22') & (df['date'] < '2021-10-01')]
    
    # defining test dataset - after 2021 season
    test_data = df[df['date'] >= '2021-10-01']

    if not basic_features_only:
        # Engineer new features for each dataset separately
        train_data = engineer_features(train_data)
        val_data = engineer_features(val_data)
        test_data = engineer_features(test_data)
    
    # Drop rows with missing values
    train_data = train_data.dropna(subset=train_data.columns.tolist())
    val_data = val_data.dropna(subset=val_data.columns.tolist())
    test_data = test_data.dropna(subset=test_data.columns.tolist())

    # Create feature matrix and target vector
    X_train = train_data.drop(columns=['target', 'team_abbreviation_home', 'team_abbreviation_away', 'game_id', 'date', 'team_id_home', 'team_id_away', 'season', 'wl_home'])
    y_train = train_data['target']
    X_val = val_data.drop(columns=['target', 'team_abbreviation_home', 'team_abbreviation_away', 'game_id', 'date', 'team_id_home', 'team_id_away', 'season', 'wl_home'])
    y_val = val_data['target']
    X_test = test_data.drop(columns=['target', 'team_abbreviation_home', 'team_abbreviation_away', 'game_id', 'date', 'team_id_home', 'team_id_away', 'season', 'wl_home'])
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
                nn.Dropout(0.5)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

    def predict_proba(self, X):
        """
        Predict class probabilities for input X
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input samples
            
        Returns:
        --------
        array-like of shape (n_samples, 2)
            Returns predicted probabilities for both classes [P(y=0), P(y=1)]
        """
        #convert X to numpy array
        X = X.to_numpy()
        
        # Convert input to tensor if not already
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
            
        # Set model to evaluation mode
        self.eval()
        
        # Get predictions
        with torch.no_grad():
            probas = self.forward(X).numpy()
        
        # Return probabilities for both classes [P(y=0), P(y=1)]
        return np.column_stack([1 - probas, probas])

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
        self.output = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.network(x)
        return self.output(x)
    
    def predict_proba(self, X):
        X = X.to_numpy()
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        self.eval()
        with torch.no_grad():
            probas = self.forward(X).numpy()
        return np.column_stack([1 - probas, probas])

def train_neural_net(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    # Convert DataFrames to numpy arrays first, then to PyTorch tensors
    X_train = torch.FloatTensor(X_train.to_numpy())
    y_train = torch.FloatTensor(y_train.values).reshape(-1, 1)
    X_val = torch.FloatTensor(X_val.to_numpy())
    y_val = torch.FloatTensor(y_val.values).reshape(-1, 1)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    best_val_loss = float('inf')
    best_state = None
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            
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
        'eval_metric': 'logloss'
    }
    
    # Create pruning callback
    pruning_callback = XGBoostPruningCallback(trial, 'validation_0-logloss')
    
    # Create DMatrix objects for XGBoost training
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Use train method instead of fit
    model = xgb.train(
        param,
        dtrain,
        num_boost_round=param['n_estimators'],
        evals=[(dval, 'validation_0')],
        callbacks=[pruning_callback],
        verbose_eval=False
    )
    
    # Evaluate model
    y_val_pred = model.predict(dval)
    score = log_loss(y_val, y_val_pred)
    
    return score

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
        'ResFFN_Small': ResFFNClassifier(X_train.shape[1], [64, 32]),
        'ResFFN_Medium': ResFFNClassifier(X_train.shape[1], [128, 64, 32]),
        'ResFFN_Large': ResFFNClassifier(X_train.shape[1], [256, 128, 64, 32])
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
                y_pred_proba = model(X_test_tensor).numpy().flatten()
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
    data_path = os.path.join(base_path, 'team_game_stats_season.csv')
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