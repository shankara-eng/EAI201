import pandas as pd
import numpy as np

# Load cleaned datasets
fifa_rankings = pd.read_csv("cleaned_FIFA_rankings.csv")
matches = pd.read_csv("cleaned_WorldCupMatches.csv")
players = pd.read_csv("cleaned_WorldCupPlayers.csv")
goals = pd.read_csv("cleaned_fifa_world_cup_all_goals.csv")
group_stats = pd.read_csv("cleaned_group_stats.csv")
intl_results = pd.read_csv("cleaned_international_football_results.csv")

# FIFA rankings features
fifa_rankings['normalized_rank'] = fifa_rankings['Rank'] / fifa_rankings['Rank'].max()
fifa_rankings['points_normalized'] = fifa_rankings['Points'] / fifa_rankings['Points'].max()

# WorldCupMatches features
matches['Home Win'] = (matches['Home Team Goals'] > matches['Away Team Goals']).astype(int)
matches['Away Win'] = (matches['Home Team Goals'] < matches['Away Team Goals']).astype(int)
matches['Draw'] = (matches['Home Team Goals'] == matches['Away Team Goals']).astype(int)

home_stats = matches.groupby('Home Team Name').agg(
    matches_played_home=('MatchID', 'count'),
    goals_scored_home=('Home Team Goals', 'sum'),
    goals_conceded_home=('Away Team Goals', 'sum'),
    wins_home=('Home Win', 'sum'),
    draws_home=('Draw', 'sum'),
    losses_home=('Away Win', 'sum'),
    win_rate=('Home Win', 'mean')
)

away_stats = matches.groupby('Away Team Name').agg(
    matches_played_away=('MatchID', 'count'),
    goals_scored_away=('Away Team Goals', 'sum'),
    goals_conceded_away=('Home Team Goals', 'sum'),
    wins_away=('Away Win', 'sum'),
    draws_away=('Draw', 'sum'),
    losses_away=('Home Win', 'sum'),
    win_rate=('Away Win', 'mean')
)

team_stats = home_stats.join(away_stats, how='outer', lsuffix='_home', rsuffix='_away').fillna(0)
team_stats['matches_played_total'] = team_stats['matches_played_home'] + team_stats['matches_played_away']
team_stats['wins_total'] = team_stats['wins_home'] + team_stats['wins_away']
team_stats['draws_total'] = team_stats['draws_home'] + team_stats['draws_away']
team_stats['losses_total'] = team_stats['losses_home'] + team_stats['losses_away']
team_stats['goals_scored_total'] = team_stats['goals_scored_home'] + team_stats['goals_scored_away']
team_stats['goals_conceded_total'] = team_stats['goals_conceded_home'] + team_stats['goals_conceded_away']
team_stats['win_rate_total'] = team_stats['wins_total'] / team_stats['matches_played_total']

# WorldCupPlayers features
player_event_counts = players.groupby(['Team Initials', 'Player Name'])['Event'].count().reset_index(name='events_count')
player_counts_per_team = players.groupby('Team Initials')['Player Name'].nunique()
avg_events_per_player = player_event_counts.groupby('Team Initials')['events_count'].mean()

player_features = pd.DataFrame({
    'player_count': player_counts_per_team,
    'avg_events_per_player': avg_events_per_player
})

# fifa world cup all goals features
goals['early_goal'] = goals['minute_regulation'] <= 30
goals['mid_goal'] = (goals['minute_regulation'] > 30) & (goals['minute_regulation'] <= 75)
goals['late_goal'] = goals['minute_regulation'] > 75

goal_period_counts = goals.groupby('team_name').agg(
    early_goals=('early_goal', 'sum'),
    mid_goals=('mid_goal', 'sum'),
    late_goals=('late_goal', 'sum'),
    penalty_goals=('penalty', 'sum'),
    own_goals=('own_goal', 'sum')
)

# group_stats features
group_stats_features = group_stats.set_index('team')[[
    'expected_goal_scored', 'exp_goal_conceded',
    'exp_goal_difference', 'points', 'wins', 'losses'
]]

# international football results features
intl_results['win'] = (intl_results['team'] == intl_results['home_team']) & (intl_results['home_team'] != intl_results['away_team'])

historic_performance = intl_results.groupby('team').agg(
    matches_played=('date', 'count'),
    win_count=('win', 'sum')
)
historic_performance['win_rate'] = historic_performance['win_count'] / historic_performance['matches_played']

# Join all features with suffixes to handle overlapping columns
all_features = team_stats.join(player_features, how='left', rsuffix='_player') \
                         .join(goal_period_counts, how='left', rsuffix='_goals') \
                         .join(group_stats_features, how='left', rsuffix='_group') \
                         .join(historic_performance, how='left', rsuffix='_historic')

# Fill missing values
all_features = all_features.fillna(0)

# Save the final engineered features DataFrame
all_features.to_csv("engineered_features.csv")

print(all_features.head())
