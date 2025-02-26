import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve, log_loss
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import joblib
from feature_engineering import engineer_features

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
    
    if basic_features_only:
        feature_cols = base_features
    else:
        # Engineer new features
        engineered_features = engineer_features(df)
        feature_cols = base_features + engineered_features
    
    # Drop rows with missing values
    df = df.dropna(subset=feature_cols + ['target'])
    
    # Create feature matrix and target vector
    X = df[feature_cols]
    y = df['target']
    
    print(f"\nUsing {'basic' if basic_features_only else 'all'} features:")
    print(f"Number of features: {len(feature_cols)}")
    print("Features used:", feature_cols)
    
    return X, y

def train_evaluate_models(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'XGBoost': None,  # Will be set after tuning
        'LightGBM': None  # Will be set after tuning
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
    
    # LightGBM hyperparameter tuning
    lgb_params = {
        'num_leaves': [15, 31],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200],
        'min_child_samples': [20, 50],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'reg_alpha': [0.0, 0.1],
        'reg_lambda': [0.0, 0.1]
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
    
    # Tune LightGBM
    base_lgb = lgb.LGBMClassifier(random_state=42, verbose=-1)
    grid_search_lgb = GridSearchCV(
        estimator=base_lgb,
        param_grid=lgb_params,
        cv=5,
        scoring='neg_log_loss',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search_lgb.fit(X_train_scaled, y_train)
    print("\nBest LightGBM parameters:", grid_search_lgb.best_params_)
    models['LightGBM'] = grid_search_lgb.best_estimator_
    
    # Train and calibrate models
    calibrated_models = {}
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train base model
        model.fit(X_train_scaled, y_train)
        
        # Calibrate model using Platt Scaling
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
        plt.plot(fpr, tpr, label=f'{name} (AUC: {result["auc_score"]:.2f})')
    
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

def main():
    # Load data with only basic features
    X, y = load_data('team_game_stats.csv', basic_features_only=True)
    
    # Train and evaluate models
    calibrated_models, results = train_evaluate_models(X, y)
    
    # Print results
    print("\nModel Performance (Basic Features Only):")
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"Brier Score: {result['brier_score']:.3f}")
        print(f"AUC Score: {result['auc_score']:.2f}")
        print(f"Log Loss: {result['log_loss']:.3f}")
    
    # Plot all curves
    plot_calibration_curves(results)
    plot_roc_curves(results)
    plot_log_loss(results)
    
    # Select best model based on Brier score
    best_model_name = min(results.items(), key=lambda x: x[1]['brier_score'])[0]
    print(f"\nBest Model (based on Brier score): {best_model_name}")
    
    # Save best model and feature names
    best_model = calibrated_models[best_model_name]
    feature_names = X.columns.tolist()
    joblib.dump({
        'model': best_model,
        'feature_names': feature_names
    }, 'basic_features_model.pkl')

if __name__ == "__main__":
    main() 