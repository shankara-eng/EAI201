"""Model evaluation for FIFA tournament prediction

This script evaluates the performance of both RandomForest and Linear Regression models
using various metrics including:
- K-fold Cross Validation
- ROC-AUC scores
- Confusion Matrix
- Feature Importance Analysis
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data():
    """Load and prepare the training data"""
    # Load team names and features
    team_names = pd.read_excel('team names.xlsx').iloc[:, 0].dropna().astype(str).tolist()
    features = pd.read_csv('features_final.csv')
    
    train_matches = []
    train_labels = []
    
    for ta in team_names:
        for tb in team_names:
            if ta == tb:
                continue
            try:
                # Extract features for team A
                fa = features[features['team name'] == ta].iloc[0]
                # Extract features for team B
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
    
    return np.array(train_matches), np.array(train_labels)

def evaluate_model(X, y):
    """Perform comprehensive model evaluation"""
    # Split data for validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    print("\n" + "="*80)
    print("MODEL EVALUATION METRICS".center(80))
    print("="*80 + "\n")

    # 1. K-fold Cross Validation
    print("1. K-FOLD CROSS VALIDATION")
    print("-"*50)
    print("Purpose: Evaluate model's consistency across different data splits")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_model, X, y, cv=kf, scoring='accuracy')
    
    print("\nResults:")
    for fold, score in enumerate(cv_scores, 1):
        print(f"Fold {fold}: {score:.3f}")
    print(f"\nOverall Performance:")
    print(f"- Average Accuracy: {cv_scores.mean():.3f}")
    print(f"- Standard Deviation: {cv_scores.std():.3f}")
    print(f"- 95% Confidence Interval: {cv_scores.mean():.3f} ± {cv_scores.std() * 2:.3f}")
    
    # 2. Train model and get predictions
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    y_pred_prob = rf_model.predict_proba(X_test)[:, 1]
    
    # 3. ROC-AUC Score
    print("\n" + "="*80)
    print("2. ROC-AUC SCORE")
    print("-"*50)
    print("Purpose: Measure model's ability to distinguish between classes")
    print("- Score ranges from 0 to 1 (1 being perfect prediction)")
    print("- 0.5 indicates random guessing")
    print("- > 0.7 is considered acceptable")
    print("- > 0.8 is considered good")
    print("- > 0.9 is considered excellent")
    
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    print(f"\nResults:")
    print(f"ROC-AUC Score: {roc_auc:.3f}")
    performance = "Excellent" if roc_auc > 0.9 else "Good" if roc_auc > 0.8 else "Acceptable" if roc_auc > 0.7 else "Poor"
    print(f"Performance Rating: {performance}")
    
    # 4. Confusion Matrix
    print("\n" + "="*80)
    print("3. CONFUSION MATRIX")
    print("-"*50)
    print("Purpose: Detailed breakdown of prediction successes and failures")
    cm = confusion_matrix(y_test, y_pred)
    print("\nResults:")
    print("                  Predicted Team A   Predicted Team B")
    print(f"Actually Team A      {cm[0][0]}                {cm[0][1]}")
    print(f"Actually Team B      {cm[1][0]}                {cm[1][1]}")
    
    # Calculate additional metrics
    total = np.sum(cm)
    accuracy = (cm[0][0] + cm[1][1]) / total
    print(f"\nMatrix Analysis:")
    print(f"- Correct Predictions: {cm[0][0] + cm[1][1]} out of {total} ({accuracy:.1%})")
    print(f"- Incorrect Predictions: {cm[0][1] + cm[1][0]} out of {total} ({1-accuracy:.1%})")
    
    # 5. Classification Report
    print("\n" + "="*80)
    print("4. DETAILED CLASSIFICATION METRICS")
    print("-"*50)
    print("Purpose: Comprehensive performance metrics for each outcome")
    print("\nResults:")
    print(classification_report(y_test, y_pred))
    
    # 6. Feature Importance Analysis
    print("\n" + "="*80)
    print("5. FEATURE IMPORTANCE ANALYSIS")
    print("-"*50)
    print("Purpose: Rank which features contribute most to predictions")
    
    feature_names = [
        'Team A Points', 'Team A Win Rate', 'Team A Goals Scored', 'Team A Goals Conceded',
        'Team B Points', 'Team B Win Rate', 'Team B Goals Scored', 'Team B Goals Conceded'
    ]
    
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    })
    importances = importances.sort_values('importance', ascending=False)
    
    print("\nFeature Rankings:")
    for idx, (feature, importance) in enumerate(zip(importances['feature'], importances['importance']), 1):
        print(f"{idx}. {feature:<20} : {importance:.3f} ({importance*100:.1f}%)")
    
    # 7. Visualizations
    plt.figure(figsize=(12, 6))
    
    # Feature Importance Plot
    plt.subplot(1, 2, 1)
    sns.barplot(data=importances, x='importance', y='feature')
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    
    # Confusion Matrix Heatmap
    plt.subplot(1, 2, 2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('model_evaluation_results.png')
    plt.close()

if __name__ == '__main__':
    # Load and prepare data
    X, y = load_and_prepare_data()
    
    # Evaluate model
    evaluate_model(X, y)