import pandas as pd

# Define a team name standardization function
def standardize_team_names(df, col):
    mapping = {
        "USA": "United States",
        "Korea Republic": "South Korea",
        # Add other mappings as necessary
    }
    df[col] = df[col].replace(mapping)
    return df

# 1. Clean FIFA top 100 teams rankings.csv
fifa_rankings = pd.read_csv("FIFA top 100 teams rankings.csv", encoding='latin1')
fifa_rankings = fifa_rankings.drop_duplicates()
fifa_rankings = fifa_rankings.dropna()
fifa_rankings = standardize_team_names(fifa_rankings, 'Team')

# 2. Clean WorldCupMatches.csv
matches = pd.read_csv("WorldCupMatches.csv", encoding='latin1')
matches['Datetime'] = pd.to_datetime(matches['Datetime'], errors='coerce')
matches = matches.dropna(subset=['Home Team Name', 'Away Team Name'])
matches = matches.drop_duplicates()
matches = standardize_team_names(matches, 'Home Team Name')
matches = standardize_team_names(matches, 'Away Team Name')

# 3. Clean WorldCupPlayers.csv
players = pd.read_csv("WorldCupPlayers.csv", encoding='latin1')
players = players.drop_duplicates()
players = players.dropna(subset=['Player Name'])
players = standardize_team_names(players, 'Team Initials')

# 4. Clean fifa world cup all goals.csv
goals = pd.read_csv("fifa world cup all goals.csv", encoding='latin1')
goals = goals.drop_duplicates()
goals = goals.dropna(subset=['goal_id', 'team_name'])
goals = standardize_team_names(goals, 'team_name')

# 5. Clean group_stats.csv
group_stats = pd.read_csv("group_stats.csv", encoding='latin1')
group_stats = group_stats.drop(columns=['Unnamed: 0'], errors='ignore')
group_stats = group_stats.drop_duplicates()
group_stats = standardize_team_names(group_stats, 'team')

# 6. Clean internaltional footbal results.csv
intl_results = pd.read_csv("internaltional footbal results.csv", encoding='latin1')
intl_results['date'] = pd.to_datetime(intl_results['date'], errors='coerce', dayfirst=True)

intl_results = intl_results.drop_duplicates()
intl_results = intl_results.dropna(subset=['home_team', 'away_team'])
intl_results = standardize_team_names(intl_results, 'home_team')
intl_results = standardize_team_names(intl_results, 'away_team')

# Save cleaned data if needed
fifa_rankings.to_csv("cleaned_FIFA_rankings.csv", index=False)
matches.to_csv("cleaned_WorldCupMatches.csv", index=False)
players.to_csv("cleaned_WorldCupPlayers.csv", index=False)
goals.to_csv("cleaned_fifa_world_cup_all_goals.csv", index=False)
group_stats.to_csv("cleaned_group_stats.csv", index=False)
intl_results.to_csv("cleaned_international_football_results.csv", index=False)
