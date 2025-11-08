"""
Feature importance analysis for Task 4
- Trains RandomForestClassifier and LogisticRegression on the synthetic match dataset
- Computes and plots:
  - Random forest feature importances
  - Logistic regression coefficients (absolute, after standardization)
- Saves plots and prints ranked features
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

FEATURE_NAMES = [
    'A_points', 'A_winrate', 'A_gf', 'A_ga',
    'B_points', 'B_winrate', 'B_gf', 'B_ga'
]


def load_matches():
    team_names = pd.read_excel('team names.xlsx').iloc[:, 0].dropna().astype(str).tolist()
    features = pd.read_csv('features_final.csv')
    rows = []
    for ta in team_names:
        for tb in team_names:
            if ta == tb:
                continue
            try:
                fa = features[features['team name'] == ta].iloc[0]
                fb = features[features['team name'] == tb].iloc[0]
                A_points = fa.get('points', 0)
                B_points = fb.get('points', 0)
                A_goals = fa.get('goals_scored_total', fa.get('goals_scored', 0))
                B_goals = fb.get('goals_scored_total', fb.get('goals_scored', 0))
                A_win = fa.get('win_rate_total', fa.get('win_rate', 0))
                B_win = fb.get('win_rate_total', fb.get('win_rate', 0))
                feat = [A_points, A_win, A_goals, fa.get('goals_conceded_total', 0),
                        B_points, B_win, B_goals, fb.get('goals_conceded_total', 0)]
                score = (A_points - B_points) + 0.01 * (A_goals - B_goals)
                label = 1 if score > 0 else 0
                rows.append((*feat, label))
            except Exception:
                continue
    df = pd.DataFrame(rows, columns=FEATURE_NAMES + ['label'])
    X = df[FEATURE_NAMES].values
    y = df['label'].values
    return df, X, y


def analyze():
    df, X, y = load_matches()
    print(f"Rows: {len(df)}, class balance: {df['label'].value_counts().to_dict()}")

    # train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    rf_imp = rf.feature_importances_

    # Logistic Regression (standardize first)
    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_test = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(Xs_train, y_train)
    lr_coef = lr.coef_[0]

    # Prepare DataFrame for plotting
    df_imp = pd.DataFrame({
        'feature': FEATURE_NAMES,
        'rf_importance': rf_imp,
        'lr_coef': lr_coef
    })
    # For LR, show absolute coefficient magnitude for ranking
    df_imp['lr_abscoef'] = np.abs(df_imp['lr_coef'])

    # Normalize to percent
    df_imp['rf_pct'] = df_imp['rf_importance'] / df_imp['rf_importance'].sum() * 100
    df_imp['lr_pct'] = df_imp['lr_abscoef'] / df_imp['lr_abscoef'].sum() * 100

    # Sort by RF importance for plotting
    df_plot = df_imp.sort_values('rf_importance', ascending=False)

    sns.set(style='whitegrid')
    plt.figure(figsize=(10,6))
    sns.barplot(x='rf_pct', y='feature', data=df_plot, color='C0')
    plt.xlabel('Random Forest importance (%)')
    plt.title('Random Forest feature importances (%)')
    plt.tight_layout()
    plt.savefig('feature_importance_rf.png')
    plt.close()

    # Logistic regression coefficients (show direction and magnitude)
    df_plot_lr = df_imp.sort_values('lr_abscoef', ascending=False)
    plt.figure(figsize=(10,6))
    sns.barplot(x='lr_abscoef', y='feature', data=df_plot_lr, palette='viridis')
    for i, v in enumerate(df_plot_lr['lr_coef']):
        plt.text(df_plot_lr['lr_abscoef'].values[i] + 0.01, i, f"{v:.3f}", va='center')
    plt.xlabel('Absolute coefficient (|coef|)')
    plt.title('Logistic Regression coefficients (absolute, standardized features)')
    plt.tight_layout()
    plt.savefig('feature_coefficients_lr.png')
    plt.close()

    # Combined table output
    print('\nFeature ranking (RandomForest importance desc):')
    print(df_imp.sort_values('rf_importance', ascending=False)[['feature','rf_pct']].to_string(index=False))

    print('\nFeature ranking (LogisticRegression |coef| desc):')
    print(df_imp.sort_values('lr_abscoef', ascending=False)[['feature','lr_abscoef','lr_coef']].to_string(index=False))

    # Save a CSV of importances
    df_imp.to_csv('feature_importances_summary.csv', index=False)
    print('\nSaved: feature_importance_rf.png, feature_coefficients_lr.png, feature_importances_summary.csv')

if __name__ == '__main__':
    analyze()
