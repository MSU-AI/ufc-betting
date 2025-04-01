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
    
    if not basic_features_only:
        # Engineer new features
        df = engineer_features(df)
    
    # Drop rows with missing values
    df = df.dropna(subset=df.columns.tolist())

    
    # defining train dataset
    train_data = df[df['season'] < 22021]
    
    # defining val dataset
    val_data = df[df['season'] == 22021]
    
    # defining test dataset
    test_data = df[df['season'] > 22021]
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
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def train_neural_net(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train.values).reshape(-1, 1)
    X_val = torch.FloatTensor(X_val)
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

def train_evaluate_models(X_train, y_train, X_val, y_val, X_test, y_test):
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'XGBoost': None,  # Will be set after tuning
        'FFN_Small': FFNClassifier(X_train.shape[1], [64, 32]),
        'FFN_Medium': FFNClassifier(X_train.shape[1], [128, 64, 32]),
        'FFN_Large': FFNClassifier(X_train.shape[1], [256, 128, 64, 32])
    }
    
    # Define parameter grid
    xgb_params = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200],
        'min_child_weight': [1, 3],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }

    # Initialize variables to store the best parameters and score
    best_params = None
    best_score = float('inf')

    # Iterate over all combinations of parameters
    for max_depth in xgb_params['max_depth']:
        for learning_rate in xgb_params['learning_rate']:
            for n_estimators in xgb_params['n_estimators']:
                for min_child_weight in xgb_params['min_child_weight']:
                    for subsample in xgb_params['subsample']:
                        for colsample_bytree in xgb_params['colsample_bytree']:
                            # Set parameters
                            params = {
                                'max_depth': max_depth,
                                'learning_rate': learning_rate,
                                'n_estimators': n_estimators,
                                'min_child_weight': min_child_weight,
                                'subsample': subsample,
                                'colsample_bytree': colsample_bytree,
                                'random_state': 42,
                                'eval_metric': 'logloss'
                            }
                            
                            # Initialize and train model
                            model = xgb.XGBClassifier(**params)
                            model.fit(
                                X_train_scaled, y_train,
                                eval_set=[(X_val_scaled, y_val)],
                                verbose=False
                            )
                            
                            # Evaluate model
                            y_val_pred = model.predict_proba(X_val_scaled)[:, 1]
                            score = log_loss(y_val, y_val_pred)
                            
                            # Update best parameters if current score is better
                            if score < best_score:
                                best_score = score
                                best_params = params

    print("\nBest XGBoost parameters:", best_params)
    print("Best validation log loss:", best_score)

    # Train the final model with the best parameters
    best_model = xgb.XGBClassifier(**best_params)
    models['XGBoost'] = best_model
    
    # Train and calibrate models
    calibrated_models = {}
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        if isinstance(model, FFNClassifier):
            # Train neural network
            model = train_neural_net(model, X_train_scaled, y_train, X_val_scaled, y_val)
            # Add neural network to calibrated_models
            calibrated_models[name] = model
            # Get predictions
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test_scaled)
                y_pred_proba = model(X_test_tensor).numpy().flatten()
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
    
    # Select best model based on Brier score
    best_model_name = min(results.items(), key=lambda x: x[1]['brier_score'])[0]
    best_brier_score = results[best_model_name]['brier_score']
    
    # Save metrics
    save_model_metrics(best_model_name, best_brier_score)
    
    # Save best model
    best_model = calibrated_models[best_model_name]
    joblib.dump(best_model, 'best_model.joblib')
    print(f"\nSaved best model ({best_model_name}) to best_model.joblib")
    
    #save test data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_data.to_csv(os.path.join(current_dir, 'test_data.csv'), index=False)

if __name__ == "__main__":
    main() 