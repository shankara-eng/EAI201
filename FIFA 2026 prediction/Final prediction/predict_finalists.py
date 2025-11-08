"""
Predict 2026 finalists using Logistic Regression + Monte-Carlo simulation

Usage: python predict_finalists.py [--teams N] [--runs R]

- trains a LogisticRegression on the synthetic match dataset used earlier
- selects the top N teams from `team names.xlsx` (or all available teams if N exceeds list)
- runs R tournament simulations by shuffling teams and simulating group stage + knockouts
- outputs CSV with frequency of reaching final and winning
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random
import json

FEATURE_NAMES = [
    'A_points', 'A_winrate', 'A_gf', 'A_ga',
    'B_points', 'B_winrate', 'B_gf', 'B_ga'
]


def build_dataset():
    teams = pd.read_excel('team names.xlsx').iloc[:, 0].dropna().astype(str).tolist()
    features = pd.read_csv('features_final.csv')
    rows = []
    for ta in teams:
        for tb in teams:
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
                rows.append((ta, tb, feat, label))
            except Exception:
                continue
    df = pd.DataFrame(rows, columns=['A','B','X','label'])
    X = np.vstack(df['X'].values)
    y = df['label'].values
    return df, X, y


def build_match_features_for_pair(features_df, teamA, teamB):
    fa = features_df[features_df['team name'] == teamA].iloc[0]
    fb = features_df[features_df['team name'] == teamB].iloc[0]
    A_points = fa.get('points', 0)
    B_points = fb.get('points', 0)
    A_goals = fa.get('goals_scored_total', fa.get('goals_scored', 0))
    B_goals = fb.get('goals_scored_total', fb.get('goals_scored', 0))
    A_win = fa.get('win_rate_total', fa.get('win_rate', 0))
    B_win = fb.get('win_rate_total', fb.get('win_rate', 0))
    feat = [A_points, A_win, A_goals, fa.get('goals_conceded_total', 0),
            B_points, B_win, B_goals, fb.get('goals_conceded_total', 0)]
    return np.array(feat)


def train_model(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(Xs, y)
    return model, scaler


def predict_match_winner_prob(model, scaler, features_df, a, b):
    X = build_match_features_for_pair(features_df, a, b).reshape(1, -1)
    Xs = scaler.transform(X)
    p = model.predict_proba(Xs)[0,1]
    return p


def simulate_tournament_once(model, scaler, features_df, teams, rng):
    # shuffle and take teams length
    order = list(teams)
    rng.shuffle(order)
    # group into groups of 4
    groups = [order[i*4:(i+1)*4] for i in range(len(order)//4)] if len(order) >=4 else [order]

    # group stage
    points_table = {t:0 for t in order}
    for group in groups:
        for i in range(len(group)):
            for j in range(i+1, len(group)):
                a, b = group[i], group[j]
                p = predict_match_winner_prob(model, scaler, features_df, a, b)
                # assign winner by threshold 0.5, ties split by prob closeness
                if p > 0.5:
                    winner = a
                else:
                    winner = b
                points_table[winner] += 3
    # top2 from each group
    group_winners = []
    for group in groups:
        sorted_group = sorted(group, key=lambda t: points_table[t], reverse=True)
        group_winners.extend(sorted_group[:2])

    # knockout
    cur = list(group_winners)
    while len(cur) > 1:
        nxt = []
        # pair sequentially
        for i in range(0, len(cur), 2):
            a, b = cur[i], cur[i+1]
            p = predict_match_winner_prob(model, scaler, features_df, a, b)
            winner = a if p > 0.5 else b
            nxt.append(winner)
        cur = nxt
    champion = cur[0]

    # The finalists are the two in the final round: retrace bracket to get finalists
    # For simplicity, rerun knockout but capture last pair
    cur = list(group_winners)
    last_pair = None
    while len(cur) > 1:
        nxt = []
        for i in range(0, len(cur), 2):
            a, b = cur[i], cur[i+1]
            p = predict_match_winner_prob(model, scaler, features_df, a, b)
            winner = a if p > 0.5 else b
            nxt.append(winner)
            last_pair = (a,b)
        cur = nxt
    finalists = last_pair if last_pair is not None else (None, None)

    return finalists, champion


def run_simulations(args):
    df, X, y = build_dataset()
    features_df = pd.read_csv('features_final.csv')
    team_list = pd.read_excel('team names.xlsx').iloc[:,0].dropna().astype(str).tolist()
    available = [t for t in team_list if not features_df[features_df['team name']==t].empty]
    sim_teams = min(args.teams, len(available))
    teams = available[:sim_teams]

    print(f"Using {len(teams)} teams for simulation. Running {args.runs} simulations...")

    model, scaler = train_model(X, y)

    rng = random.Random(args.seed)

    finalists_counts = {}
    winners_counts = {}
    pair_counts = {}

    for run in range(args.runs):
        finals, champion = simulate_tournament_once(model, scaler, features_df, teams, rng)
        if finals[0] is None:
            continue
        a, b = finals
        # increment finalists
        finalists_counts[a] = finalists_counts.get(a,0) + 1
        finalists_counts[b] = finalists_counts.get(b,0) + 1
        # increment pair frequency (sorted tuple)
        pair = tuple(sorted([a,b]))
        pair_counts[pair] = pair_counts.get(pair,0) + 1
        # winner
        winners_counts[champion] = winners_counts.get(champion,0) + 1

    # Convert to DataFrame
    df_finalists = pd.DataFrame.from_dict(finalists_counts, orient='index', columns=['finals_count'])
    df_finalists['finals_pct'] = df_finalists['finals_count'] / args.runs
    df_finalists = df_finalists.sort_values('finals_count', ascending=False)

    df_winners = pd.DataFrame.from_dict(winners_counts, orient='index', columns=['wins_count'])
    df_winners['wins_pct'] = df_winners['wins_count'] / args.runs
    df_winners = df_winners.sort_values('wins_count', ascending=False)

    df_pairs = pd.DataFrame([(a,b,c) for (a,b),c in pair_counts.items()], columns=['teamA','teamB','pair_count'])
    if not df_pairs.empty:
        df_pairs['pair_pct'] = df_pairs['pair_count'] / args.runs
        df_pairs = df_pairs.sort_values('pair_count', ascending=False)

    df_finalists.to_csv('predicted_finalists_frequency.csv')
    df_winners.to_csv('predicted_winners_frequency.csv')
    df_pairs.to_csv('predicted_final_pairs_frequency.csv', index=False)

    print('Saved: predicted_finalists_frequency.csv, predicted_winners_frequency.csv, predicted_final_pairs_frequency.csv')
    # Print top 10 finalists and winners
    print('\nTop 10 finalists by frequency:')
    print(df_finalists.head(10))
    print('\nTop 10 winners by frequency:')
    print(df_winners.head(10))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--teams', type=int, default=16, help='Number of teams to use in simulation (must be multiple of 4 ideally)')
    p.add_argument('--runs', type=int, default=500, help='Number of Monte-Carlo tournament simulations')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    args = p.parse_args()
    run_simulations(args)
