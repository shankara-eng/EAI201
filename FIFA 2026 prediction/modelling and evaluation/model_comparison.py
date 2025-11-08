"""
Model Comparison: Random Forest vs Linear Regression
Generates and compares:
1. ROC curves for both models
2. Confusion matrices for both models
3. Performance metrics comparison
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression  # Better for binary classification than LinearRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Load and prepare the match data"""
    # Load team names and features
    team_names = pd.read_excel('team names.xlsx').iloc[:, 0].dropna().astype(str).tolist()
    features = pd.read_csv('features_final.csv')
    
    train_matches = []
    train_labels = []
    
    print("Preparing match data...")
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
                
                # Create binary label (1 if team A wins, 0 if team B wins)
                score = (fa.get('points', 0) - fb.get('points', 0)) + \
                        0.01 * (fa.get('goals_scored_total', 0) - fb.get('goals_scored_total', 0))
                label = 1 if score > 0 else 0
                
                train_matches.append(match_features)
                train_labels.append(label)
                
            except (KeyError, IndexError):
                continue
    
    return np.array(train_matches), np.array(train_labels)

def plot_roc_curves(models, X_test, y_test, model_names):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(10, 8))
    
    for model, name in zip(models, model_names):
        # Get predictions
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curves_comparison.png')
    plt.close()

def plot_confusion_matrices(models, X_test, y_test, model_names):
    """Plot confusion matrices for all models"""
    fig, axes = plt.subplots(1, len(models), figsize=(15, 5))
    
    for ax, model, name in zip(axes, models, model_names):
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{name}\nConfusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices_comparison.png')
    plt.close()

def print_model_metrics(models, X_test, y_test, model_names):
    """Print detailed metrics for each model"""
    for model, name in zip(models, model_names):
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        cm = confusion_matrix(y_test, y_pred)
        accuracy = (cm[0,0] + cm[1,1]) / np.sum(cm)
        precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
        recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n{'='*50}")
        print(f"{name} Metrics:")
        print(f"{'='*50}")
        print(f"Accuracy:  {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1 Score:  {f1:.3f}")
        print("\nConfusion Matrix:")
        print("                  Predicted Team A   Predicted Team B")
        print(f"Actually Team A      {cm[0][0]}                {cm[0][1]}")
        print(f"Actually Team B      {cm[1][0]}                {cm[1][1]}")

def main():
    # Load data
    X, y = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    log_reg_model = LogisticRegression(random_state=42)
    
    models = [rf_model, log_reg_model]
    model_names = ['Random Forest', 'Logistic Regression']
    
    # Train models
    print("\nTraining models...")
    for model, name in zip(models, model_names):
        print(f"Training {name}...")
        model.fit(X_train, y_train)
    
    # Generate plots
    print("\nGenerating ROC curves...")
    plot_roc_curves(models, X_test, y_test, model_names)
    
    print("Generating confusion matrices...")
    plot_confusion_matrices(models, X_test, y_test, model_names)
    
    # Print metrics
    print("\nCalculating performance metrics...")
    print_model_metrics(models, X_test, y_test, model_names)
    
    print("\nAnalysis complete!")
    print("- ROC curves saved as 'roc_curves_comparison.png'")
    print("- Confusion matrices saved as 'confusion_matrices_comparison.png'")

if __name__ == '__main__':
    main()