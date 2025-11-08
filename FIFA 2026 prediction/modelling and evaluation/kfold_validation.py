"""K-fold Cross Validation analysis for FIFA tournament prediction models

This script performs k-fold cross validation on both RandomForest and Linear Regression models
to evaluate their prediction performance and stability.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def load_data():
    """Load and prepare the training data"""
    # Load team names and features
    team_names = pd.read_excel('team names.xlsx').iloc[:, 0].dropna().astype(str).tolist()
    features = pd.read_csv('features_final.csv')
    print(f"Loaded {len(team_names)} teams")
    
    train_matches = []
    train_labels = []
    
    print("Preparing match pairs...")
    for ta in team_names:
        for tb in team_names:
            if ta == tb:
                continue
            try:
                # Get team features
                fa = features[features['team name'] == ta].iloc[0]
                fb = features[features['team name'] == tb].iloc[0]
                
                # Create match features
                match_features = [
                    fa.get('points', 0),
                    fa.get('win_rate_total', fa.get('win_rate', 0)),
                    fa.get('goals_scored_total', fa.get('goals_scored', 0)),
                    fa.get('goals_conceded_total', fa.get('goals_conceded', 0)),
                    fb.get('points', 0),
                    fb.get('win_rate_total', fb.get('win_rate', 0)),
                    fb.get('goals_scored_total', fb.get('goals_scored', 0)),
                    fb.get('goals_conceded_total', fb.get('goals_conceded', 0))
                ]
                
                # Create label (1 if team A wins, 0 if team B wins)
                score = (fa.get('points', 0) - fb.get('points', 0)) + \
                        0.01 * (fa.get('goals_scored_total', 0) - fb.get('goals_scored_total', 0))
                label = 1 if score > 0 else 0
                
                train_matches.append(match_features)
                train_labels.append(label)
                
            except (KeyError, IndexError):
                continue
    
    print(f"Created {len(train_matches)} match pairs")
    return np.array(train_matches), np.array(train_labels)

def perform_kfold_validation(X, y, n_splits=5):
    """Perform k-fold cross validation with both models"""
    # Initialize models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Initialize K-fold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Lists to store scores for each fold
    rf_scores = []
    fold_indices = []
    
    print(f"\nPerforming {n_splits}-fold cross validation...")
    
    # Manual K-fold to get detailed per-fold information
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train and evaluate RandomForest
        rf_model.fit(X_train, y_train)
        rf_score = accuracy_score(y_val, rf_model.predict(X_val))
        rf_scores.append(rf_score)
        
        print(f"\nFold {fold} Results:")
        print(f"RandomForest Accuracy: {rf_score:.3f}")
        print(f"Training size: {len(X_train)}, Validation size: {len(X_val)}")
        
        fold_indices.append(fold)
    
    # Calculate overall statistics
    print("\nOverall Results:")
    print(f"RandomForest - Average Accuracy: {np.mean(rf_scores):.3f} (+/- {np.std(rf_scores) * 2:.3f})")
    
    # Also perform scikit-learn's cross_val_score for verification
    print("\nCross Validation Score Verification:")
    cv_scores = cross_val_score(rf_model, X, y, cv=n_splits)
    print(f"RandomForest CV Scores: {cv_scores}")
    print(f"RandomForest CV Mean: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.plot(fold_indices, rf_scores, 'bo-', label='RandomForest')
    plt.axhline(y=np.mean(rf_scores), color='b', linestyle='--', alpha=0.3)
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title(f'{n_splits}-Fold Cross Validation Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('kfold_validation_results.png')
    plt.close()

if __name__ == '__main__':
    # Load data
    X, y = load_data()
    
    # Perform k-fold validation
    perform_kfold_validation(X, y, n_splits=5)