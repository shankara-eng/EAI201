#python .\linreg.py --interactive

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import random
import argparse
import json
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, scrolledtext
except Exception:
    tk = None
    ttk = None
    messagebox = None
    scrolledtext = None


def pick(s: pd.Series, *candidates):
    for c in candidates:
        if c in s.index:
            return s[c]
    return 0


# Load team names and preview
try:
    team_names = pd.read_excel('team names.xlsx').iloc[:, 0].dropna().astype(str).tolist()
except Exception as e:
    raise SystemExit("Failed to read 'team names.xlsx': " + str(e))

print(f"Loaded {len(team_names)} team names")

try:
    features = pd.read_csv('features_final.csv')
except Exception as e:
    raise SystemExit("Failed to read 'features_final.csv': " + str(e))

print('Features preview:\n', features.head())


def build_match_features(teamA, teamB):
    # Find feature rows
    fa = features[features['team name'] == teamA]
    fb = features[features['team name'] == teamB]
    if fa.empty or fb.empty:
        raise KeyError(f"Team not found in features: {teamA if fa.empty else teamB}")
    fa = fa.iloc[0]
    fb = fb.iloc[0]
    # choose multiple candidate columns for each base measure
    A_points = pick(fa, 'points', 'rating', 'fifa_points')
    B_points = pick(fb, 'points', 'rating', 'fifa_points')
    A_winrate = pick(fa, 'win_rate', 'winrate', 'win_rate_total', 'win_rate_pct')
    B_winrate = pick(fb, 'win_rate', 'winrate', 'win_rate_total', 'win_rate_pct')
    A_gf = pick(fa, 'goals_scored_total', 'goals_scored', 'totalgoalsscored', 'gf')
    B_gf = pick(fb, 'goals_scored_total', 'goals_scored', 'totalgoalsscored', 'gf')
    A_ga = pick(fa, 'goals_conceded_total', 'goals_conceded', 'totalgoalsconceded', 'ga')
    B_ga = pick(fb, 'goals_conceded_total', 'goals_conceded', 'totalgoalsconceded', 'ga')

    # expected goals / xG related
    A_xg = pick(fa, 'expected_goal_scored', 'exp_goal_scored', 'exp_goal')
    B_xg = pick(fb, 'expected_goal_scored', 'exp_goal_scored', 'exp_goal')
    A_xg_conc = pick(fa, 'exp_goal_conceded', 'exp_goal_conceded', 'exp_goal_conceded')
    B_xg_conc = pick(fb, 'exp_goal_conceded', 'exp_goal_conceded', 'exp_goal_conceded')
    A_xg_diff = pick(fa, 'exp_goal_difference', 'exp_goal_difference')
    B_xg_diff = pick(fb, 'exp_goal_difference', 'exp_goal_difference')

    # other useful columns
    A_matches = pick(fa, 'matches_played_home', 'matches_played', 'matches_played_total', 'matches')
    B_matches = pick(fb, 'matches_played_home', 'matches_played', 'matches_played_total', 'matches')
    A_win_count = pick(fa, 'win_count', 'wins_total', 'wins')
    B_win_count = pick(fb, 'win_count', 'wins_total', 'wins')

    # derived features
    A_goal_diff = (0 if A_gf is None else A_gf) - (0 if A_ga is None else A_ga)
    B_goal_diff = (0 if B_gf is None else B_gf) - (0 if B_ga is None else B_ga)
    # avoid division by zero
    A_goal_ratio = (A_gf / (A_ga + 1)) if (A_ga is not None and A_ga != 0) else float(A_gf)
    B_goal_ratio = (B_gf / (B_ga + 1)) if (B_ga is not None and B_ga != 0) else float(B_gf)

    vals = [
        A_points, A_winrate, A_gf, A_ga, A_matches, A_win_count, A_xg, A_xg_conc, A_xg_diff, A_goal_diff, A_goal_ratio,
        B_points, B_winrate, B_gf, B_ga, B_matches, B_win_count, B_xg, B_xg_conc, B_xg_diff, B_goal_diff, B_goal_ratio
    ]
    # convert to numeric (float)
    vals = [float(0 if v is None or (isinstance(v, float) and np.isnan(v)) else v) for v in vals]
    return np.array(vals, dtype=float)


def parse_args():
    p = argparse.ArgumentParser(description='Linear-regression tournament simulator (interactive)')
    p.add_argument('--interactive', action='store_true', help='Run interactive GUI for manual knockout selection')
    p.add_argument('--teams', type=int, default=16, help='Number of teams to simulate (must be even)')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--shuffle-runs', type=int, default=0, help='If >0 run Monte-Carlo by shuffling groups this many times and report win frequencies')
    return p.parse_args()


def interactive_knockout_gui(all_teams, regressor, args):
    if tk is None:
        print("Tkinter not available. Install or run without --interactive.")
        return

    root = tk.Tk()
    root.title('LinReg Knockout Manager')

    mainframe = ttk.Frame(root, padding=8)
    mainframe.grid(row=0, column=0, sticky='nsew')

    ttk.Label(mainframe, text='Choose action:').grid(row=0, column=0, sticky='w')

    # Monte Carlo controls
    mc_frame = ttk.LabelFrame(mainframe, text='Monte Carlo Simulation', padding=(5,5,5,5))
    mc_frame.grid(row=1, column=0, sticky='ew', pady=(5,5))
    ttk.Label(mc_frame, text='Number of runs:').grid(row=0, column=0, sticky='w')
    mc_runs = ttk.Entry(mc_frame, width=10)
    mc_runs.insert(0, '100')  # default value
    mc_runs.grid(row=0, column=1, padx=(5,5))
    ttk.Button(mc_frame, text='Run Monte Carlo', command=lambda: monte_carlo_runs(mc_runs.get())).grid(row=0, column=2, padx=(5,0))

    # text area for game-flow
    if scrolledtext is not None:
        txt = scrolledtext.ScrolledText(mainframe, width=80, height=24, wrap=tk.WORD)
        txt.grid(row=0, column=1, rowspan=10, padx=(8,0))
    else:
        txt = None

    def gui_log(msg: str):
        if txt is not None:
            txt.insert(tk.END, msg + '\n')
            txt.see(tk.END)
        else:
            print(msg)

    # keep last groups so user can shuffle and then auto-run the same groups
    last_groups = {'groups': None, 'teams': None}

    def predict_and_winner(a, b):
        X = build_match_features(a, b).reshape(1, -1)
        pred = float(regressor.predict(X)[0])
        return pred, (a if pred > 0 else b)

    def on_auto():
        # use last shuffled groups if present, otherwise create new random groups
        if last_groups['groups'] is not None:
            groups = last_groups['groups']
            teams = last_groups['teams']
        else:
            sim_teams = min(args.teams, len(all_teams))
            # use numpy RNG seeded for reproducibility when auto-run without prior shuffle
            np.random.seed(args.seed)
            teams = list(all_teams)
            np.random.shuffle(teams)
            teams = teams[:sim_teams]
            groups = [teams[i*4:(i+1)*4] for i in range(len(teams)//4)] if len(teams) >=4 else [teams]
        points_table = {t:0 for t in teams}
        for group in groups:
            gui_log(f"\nGroup: {group}")
            for i in range(len(group)):
                for j in range(i+1, len(group)):
                    a, b = group[i], group[j]
                    pred, winner = predict_and_winner(a, b)
                    points_table[winner] += 3
                    gui_log(f"{a} vs {b} -> {winner} (pred diff {pred:.2f})")
        # choose top2
        group_winners = []
        for group in groups:
            sorted_group = sorted(group, key=lambda t: points_table[t], reverse=True)
            group_winners.extend(sorted_group[:2])
        gui_log(f"\nAdvancing to Knockout: {group_winners}")
        # knockout simulate
        cur = list(group_winners)
        rnd = 1
        while len(cur) > 1:
            gui_log(f"\nKnockout Round {rnd}:")
            nxt = []
            for i in range(0, len(cur), 2):
                a, b = cur[i], cur[i+1]
                pred, winner = predict_and_winner(a, b)
                gui_log(f"{a} vs {b} -> {winner} advances (pred {pred:.2f})")
                nxt.append(winner)
            cur = nxt
            rnd += 1
        gui_log(f"\nTournament Winner: {cur[0]}")
        messagebox.showinfo('Auto Result', f'Tournament Winner: {cur[0]}')

    def shuffle_groups():
        # non-deterministic shuffle: pick a random unique selection and partition into groups
        sim_teams = min(args.teams, len(all_teams))
        order = list(all_teams)
        random.shuffle(order)
        teams = order[:sim_teams]
        groups = [teams[i*4:(i+1)*4] for i in range(len(teams)//4)] if len(teams) >=4 else [teams]
        last_groups['groups'] = groups
        last_groups['teams'] = teams
        gui_log(f"Shuffled groups: {groups}")
        # After shuffling, immediately run the auto group-stage -> knockout simulation
        # so the user sees progression to the next rounds without needing to click Auto.
        gui_log('Running auto-predict for shuffled groups...')
        try:
            on_auto()
        except Exception as e:
            gui_log(f"Auto-run after shuffle failed: {e}")

    def monte_carlo_runs(n_runs: int):
        # Run n_runs shuffles and simulate tournaments, counting wins per team.
        try:
            n = int(n_runs)
            if n <= 0:
                raise ValueError("Number of runs must be positive")
        except ValueError as e:
            gui_log(f"Invalid number of runs: {e}")
            return
        except Exception as e:
            gui_log(f"Error parsing runs: {e}")
            return

        # use last known RNG seed if set, otherwise use time-based seed
        rng = random.Random(args.seed)
        available = [t for t in all_teams if not features[features['team name'] == t].empty]
        if not available:
            gui_log('No teams with features available for Monte-Carlo')
            return

        sim_teams = min(args.teams, len(available))
        win_counts = {t: 0 for t in available}
        gui_log(f"\nRunning {n} Monte-Carlo simulations with {sim_teams} teams...")

        for i in range(n):
            order = list(available)
            rng.shuffle(order)
            teams = order[:sim_teams]
            groups = [teams[i*4:(i+1)*4] for i in range(len(teams)//4)] if len(teams) >=4 else [teams]

            # simulate one tournament
            points_table = {t: 0 for t in teams}
            for group in groups:
                for i in range(len(group)):
                    for j in range(i+1, len(group)):
                        ta, tb = group[i], group[j]
                        pred, winner = predict_and_winner(ta, tb)
                        points_table[winner] += 3

            group_winners = []
            for group in groups:
                sorted_group = sorted(group, key=lambda t: points_table[t], reverse=True)
                group_winners.extend(sorted_group[:2])

            cur = list(group_winners)
            while len(cur) > 1:
                nxt = []
                for i in range(0, len(cur), 2):
                    ta, tb = cur[i], cur[i+1]
                    pred, winner = predict_and_winner(ta, tb)
                    nxt.append(winner)
                cur = nxt

            win_counts[cur[0]] += 1
            if (i + 1) % 10 == 0:
                gui_log(f"Completed {i + 1} simulations...")

        sorted_counts = sorted(win_counts.items(), key=lambda kv: kv[1], reverse=True)
        gui_log(f"\nMonte-Carlo results (wins out of {n}):")
        for team, cnt in sorted_counts:
            if cnt > 0:
                gui_log(f"{team}: {cnt} ({cnt/n:.3f})")
            gui_log('Invalid runs value')
            return
        if n <= 0:
            gui_log('Runs must be > 0')
            return

        # only simulate with teams that have feature rows present
        available = [t for t in all_teams if not features[features['team name'] == t].empty]
        if not available:
            gui_log('No teams with features available for Monte-Carlo')
            return
        sim_teams = min(args.teams, len(available))
        rng = random.Random(args.seed)
        win_counts = {t: 0 for t in available}
        gui_log(f"Starting Monte-Carlo: {n} runs (seed={args.seed})...")
        for run in range(n):
            order = list(available)
            rng.shuffle(order)
            teams = order[:sim_teams]
            groups = [teams[i*4:(i+1)*4] for i in range(len(teams)//4)] if len(teams) >=4 else [teams]

            # simulate one tournament without messageboxes
            points_table = {t: 0 for t in teams}
            for group in groups:
                for i in range(len(group)):
                    for j in range(i+1, len(group)):
                        a, b = group[i], group[j]
                        pred, winner = predict_and_winner(a, b)
                        points_table[winner] += 3

            group_winners = []
            for group in groups:
                sorted_group = sorted(group, key=lambda t: points_table[t], reverse=True)
                group_winners.extend(sorted_group[:2])

            cur = list(group_winners)
            while len(cur) > 1:
                nxt = []
                for i in range(0, len(cur), 2):
                    a, b = cur[i], cur[i+1]
                    pred, winner = predict_and_winner(a, b)
                    nxt.append(winner)
                cur = nxt

            win_counts[cur[0]] += 1

        # prepare sorted results
        sorted_counts = sorted(win_counts.items(), key=lambda kv: kv[1], reverse=True)
        gui_log('\nMonte-Carlo results (wins out of %d):' % n)
        for team, cnt in sorted_counts:
            if cnt > 0:
                gui_log(f"{team}: {cnt} ({cnt/n:.3f})")
        # also show top 10 in a messagebox
        top10 = '\n'.join([f"{t}: {c} ({c/n:.3f})" for t, c in sorted_counts[:10]])
        try:
            messagebox.showinfo('Monte-Carlo Results', f'Top winners:\n{top10}')
        except Exception:
            pass

    def on_manual():
        start_win = tk.Toplevel(root)
        start_win.title('Manual Start Round')
        ttk.Label(start_win, text='Start at round:').grid(row=0, column=0, sticky='w')
        start_var = tk.IntVar(value=1)
        ttk.Radiobutton(start_win, text='Round 1 (16 teams)', variable=start_var, value=1).grid(row=1, column=0, sticky='w')
        ttk.Radiobutton(start_win, text='Round 2 (8 teams)', variable=start_var, value=2).grid(row=2, column=0, sticky='w')
        ttk.Radiobutton(start_win, text='Round 3 (4 teams)', variable=start_var, value=3).grid(row=3, column=0, sticky='w')

        def start_sel():
            start_win.destroy()
            r = start_var.get()
            if r == 1:
                needed = 16
            elif r == 2:
                needed = 8
            else:
                needed = 4
            sel = tk.Toplevel(root)
            sel.title(f'Select {needed} teams')
            vars = []
            for i in range(needed):
                ttk.Label(sel, text=f'Slot {i+1}').grid(row=i, column=0, sticky='w')
                v = tk.StringVar(value=all_teams[i])
                om = ttk.OptionMenu(sel, v, v.get(), *all_teams)
                om.grid(row=i, column=1, sticky='ew')
                vars.append(v)

            def randomize_slots():
                # fill slots with a random (unique) selection of teams
                try:
                    order = list(all_teams)
                    np.random.shuffle(order)
                except Exception:
                    order = all_teams[:]
                sel_teams = order[:needed]
                for vv, team in zip(vars, sel_teams):
                    vv.set(team)

            ttk.Button(sel, text='Randomize slots', command=randomize_slots).grid(row=needed+0, column=2)

            def submit_slots():
                chosen = [v.get() for v in vars]
                if len(set(chosen)) != len(chosen):
                    messagebox.showerror('Error','Please pick unique teams for each slot')
                    return
                sel.destroy()
                gui_log(f"\nManual start teams: {chosen}")
                current = chosen
                while len(current) > 1:
                    match_win = tk.Toplevel(root)
                    match_win.title(f'Round with {len(current)} teams')
                    pair_vars = []
                    pairs = []
                    for i in range(0, len(current), 2):
                        a, b = current[i], current[i+1]
                        ttk.Label(match_win, text=f'{a} vs {b}').grid(row=i//2, column=0, sticky='w')
                        pred, winner = predict_and_winner(a, b)
                        pv = tk.StringVar(value=winner)
                        om = ttk.OptionMenu(match_win, pv, pv.get(), a, b)
                        om.grid(row=i//2, column=1, sticky='ew')
                        pair_vars.append(pv)
                        pairs.append((a,b))

                    def submit_winners():
                        winners = [pv.get() for pv in pair_vars]
                        for (a,b), w in zip(pairs, winners):
                            gui_log(f"{a} vs {b} -> {w}")
                        match_win.destroy()
                        nonlocal current
                        current = winners

                    ttk.Button(match_win, text='Confirm winners', command=submit_winners).grid(row=99, column=0, columnspan=2)
                    match_win.grab_set()
                    root.wait_window(match_win)
                gui_log(f"\nTournament Winner: {current[0]}")
                messagebox.showinfo('Manual Result', f'Tournament Winner: {current[0]}')

            ttk.Button(sel, text='Submit teams', command=submit_slots).grid(row=needed+1, column=0, columnspan=2)
            sel.grab_set()

        ttk.Button(start_win, text='Start', command=start_sel).grid(row=4, column=0)

    ttk.Button(mainframe, text='Shuffle groups', command=shuffle_groups).grid(row=1, column=0, sticky='ew')
    # Monte-Carlo runs entry + button
    ttk.Label(mainframe, text='Runs:').grid(row=1, column=2, sticky='w')
    runs_var = tk.StringVar(value='100')
    runs_entry = ttk.Entry(mainframe, textvariable=runs_var, width=8)
    runs_entry.grid(row=1, column=3, sticky='w')
    def on_mc():
        try:
            nr = int(runs_var.get())
        except Exception:
            nr = 100
        monte_carlo_runs(nr)
    ttk.Button(mainframe, text='Shuffle N times', command=on_mc).grid(row=1, column=4, sticky='ew')
    ttk.Button(mainframe, text='Auto predict from group stage', command=on_auto).grid(row=2, column=0, sticky='ew')
    ttk.Button(mainframe, text='Manual knockout (choose teams)', command=on_manual).grid(row=3, column=0, sticky='ew')
    root.mainloop()


# Build training data: synthetic labels = (A_points - B_points)
train_matches = []
train_scores = []
for ta in team_names:
    for tb in team_names:
        if ta == tb:
            continue
        try:
            feat = build_match_features(ta, tb)
        except KeyError:
            continue
        train_matches.append(feat)
        # label: point difference (fallback to 0 if missing)
        fa = features[features['team name'] == ta].iloc[0]
        fb = features[features['team name'] == tb].iloc[0]
        A_points = pick(fa, 'points', 'rating', 'fifa_points') or 0
        B_points = pick(fb, 'points', 'rating', 'fifa_points') or 0
        train_scores.append(float(A_points - B_points))

if not train_matches:
    raise SystemExit('No training data created — check feature names and team names')

X_train = np.vstack(train_matches)
y_train = np.array(train_scores, dtype=float)

# Train Linear Regression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


def simulate_and_print(regressor, args):
    # Tournament simulation (uses args.teams and args.seed)
    np.random.seed(args.seed)
    sim_teams = min(args.teams, len(team_names))
    tournament_teams = team_names[:sim_teams]
    np.random.shuffle(tournament_teams)

    groups = [tournament_teams[i*4:(i+1)*4] for i in range(len(tournament_teams)//4)]
    points_table = {team: 0 for team in tournament_teams}
    print("\nGroup Stage:")
    for i, group in enumerate(groups):
        group_points = {team: 0 for team in group}
        print(f"\nGroup {chr(ord('A')+i)}: {group}")
        for m1 in range(len(group)):
            for m2 in range(m1+1, len(group)):
                ta, tb = group[m1], group[m2]
                X = build_match_features(ta, tb).reshape(1, -1)
                pred_score_diff = float(regressor.predict(X)[0])
                # small threshold to consider draw
                eps = 1e-6
                if pred_score_diff > eps:
                    winner = ta
                    group_points[winner] += 3
                    print(f"{ta} defeats {tb} (3 points for {ta}) -> score diff {pred_score_diff:.2f}")
                elif pred_score_diff < -eps:
                    winner = tb
                    group_points[winner] += 3
                    print(f"{tb} defeats {ta} (3 points for {tb}) -> score diff {pred_score_diff:.2f}")
                else:
                    group_points[ta] += 1
                    group_points[tb] += 1
                    print(f"{ta} draws with {tb} (1 point each) -> score diff {pred_score_diff:.2f}")
        print("Points Table:", group_points)
        for team in group:
            points_table[team] += group_points[team]

    # Take top 2 from each group
    group_winners = []
    for group in groups:
        sorted_group = sorted(group, key=lambda t: points_table[t], reverse=True)
        group_winners.extend(sorted_group[:2])
    print("\nAdvancing to Knockout:", group_winners)

    # Knockout rounds
    knockout = group_winners
    round_no = 1
    while len(knockout) > 1:
        print(f"\nKnockout Round {round_no}:")
        winners = []
        for i in range(0, len(knockout), 2):
            ta, tb = knockout[i], knockout[i+1]
            X = build_match_features(ta, tb).reshape(1, -1)
            pred_score_diff = float(regressor.predict(X)[0])
            winner = ta if pred_score_diff > 0 else tb
            print(f"{ta} vs {tb} -> {winner} advances (pred diff {pred_score_diff:.2f})")
            winners.append(winner)
        knockout = winners
        round_no += 1

    print(f"\nTournament Winner: {knockout[0]}")


def perform_monte_carlo_cli(regressor, args, runs: int):
    # Monte-Carlo shuffling for CLI mode. Uses args.seed for reproducibility.
    # only simulate with teams that have feature rows present
    available = [t for t in team_names if not features[features['team name'] == t].empty]
    if not available:
        print('No teams with features available for Monte-Carlo')
        return
    rng = random.Random(args.seed)
    sim_teams = min(args.teams, len(available))
    win_counts = {t: 0 for t in available}
    for _ in range(runs):
        order = list(available)
        rng.shuffle(order)
        teams = order[:sim_teams]
        groups = [teams[i*4:(i+1)*4] for i in range(len(teams)//4)] if len(teams) >=4 else [teams]

        # simulate one tournament
        points_table = {t: 0 for t in teams}
        for group in groups:
            for i in range(len(group)):
                for j in range(i+1, len(group)):
                    ta, tb = group[i], group[j]
                    X = build_match_features(ta, tb).reshape(1, -1)
                    pred_score_diff = float(regressor.predict(X)[0])
                    if pred_score_diff > 0:
                        points_table[ta] += 3
                    elif pred_score_diff < 0:
                        points_table[tb] += 3

        group_winners = []
        for group in groups:
            sorted_group = sorted(group, key=lambda t: points_table[t], reverse=True)
            group_winners.extend(sorted_group[:2])

        cur = list(group_winners)
        while len(cur) > 1:
            nxt = []
            for i in range(0, len(cur), 2):
                ta, tb = cur[i], cur[i+1]
                X = build_match_features(ta, tb).reshape(1, -1)
                pred_score_diff = float(regressor.predict(X)[0])
                winner = ta if pred_score_diff > 0 else tb
                nxt.append(winner)
            cur = nxt

        win_counts[cur[0]] += 1

    sorted_counts = sorted(win_counts.items(), key=lambda kv: kv[1], reverse=True)
    print(f"\nMonte-Carlo results (wins out of {runs}):")
    for team, cnt in sorted_counts:
        if cnt > 0:
            print(f"{team}: {cnt} ({cnt/runs:.3f})")


if __name__ == '__main__':
    args = parse_args()
    # If interactive requested, show GUI which will call predictions using regressor
    if args.interactive:
        interactive_knockout_gui(team_names, regressor, args)
    else:
        # If user requested Monte-Carlo shuffle runs via CLI, run that and exit
        if getattr(args, 'shuffle_runs', 0) and args.shuffle_runs > 0:
            perform_monte_carlo_cli(regressor, args, args.shuffle_runs)
        else:
            simulate_and_print(regressor, args)
