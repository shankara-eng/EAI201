"""Random forest tournament simulator (improved)

Improvements made:
- Normalize team names from `team names.xlsx` to match `features_final.csv`.
- Report and map missing/variant names (e.g. 'Türkiye' -> 'Turkey').
- Slightly improved label creation (points + small tie-breaker on goals).
- Use consistent numeric arrays for model training and predict with .to_numpy() to avoid sklearn warnings.
- Add robust error handling and a main guard.
"""
#python .\randomforest.py --interactivey

import unicodedata
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


def normalize(name: str) -> str:
    if not isinstance(name, str):
        return ""
    # remove accents, lowercase, remove punctuation
    s = unicodedata.normalize('NFKD', name)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    for ch in "._-,/()":
        s = s.replace(ch, ' ')
    s = ' '.join(s.split())
    return s


def parse_args():
    p = argparse.ArgumentParser(description='RandomForest tournament simulator')
    p.add_argument('-t', '--teams', type=int, default=16, help='Number of teams to simulate (must be even)')
    p.add_argument('--seed', type=int, default=42, help='Random seed for reproducible draws')
    p.add_argument('--features', type=str, default='features_final.csv', help='Path to features CSV')
    p.add_argument('--teams-file', type=str, default='team names.xlsx', help='Excel with team list')
    p.add_argument('--save', action='store_true', help='Save tournament results to JSON file')
    p.add_argument('-o', '--output', type=str, default='tournament.json', help='Output file when --save used')
    p.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    p.add_argument('--interactive', action='store_true', help='Run interactive GUI for manual knockout selection')
    p.add_argument('--shuffle-runs', type=int, default=0, help='If >0 run Monte-Carlo by shuffling groups this many times and report win frequencies')
    return p.parse_args()


def main(args):
    # load files with clear errors if missing
    try:
        raw_team_names = pd.read_excel(args.teams_file).iloc[:, 0].dropna().astype(str).tolist()
    except Exception as e:
        print(f"Error: cannot read '{args.teams_file}' —", e)
        return

    try:
        features = pd.read_csv(args.features)
    except Exception as e:
        print(f"Error: cannot read '{args.features}' —", e)
        return

    print(f"Loaded {len(raw_team_names)} team names from Excel.")
    print("Features preview:\n", features.head())

    # build normalized lookup for features
    norm_to_feature = {normalize(t): t for t in features['team name'].tolist()}

    # manual normalization overrides for known variants
    manual_map = {
        'turkiye': 'turkey',
        'cote d ivoire': 'ivory coast',
        'cabo verde': 'cape verde',
        'united states of america': 'united states',
        'usa': 'united states'
    }

    mapped_team_names = []
    missing = []
    for t in raw_team_names:
        n = normalize(t)
        # direct match
        if n in norm_to_feature:
            mapped_team_names.append(norm_to_feature[n])
            continue
        # apply manual mapping
        if n in manual_map:
            mapped = manual_map[n]
            if mapped in norm_to_feature:
                mapped_team_names.append(norm_to_feature[mapped])
                continue
        # try manual_map target normalized lookup
        for key, val in manual_map.items():
            if n == key and val in norm_to_feature:
                mapped_team_names.append(norm_to_feature[val])
                break
        else:
            # attempt fuzzy-ish match: find closest by prefix
            candidates = [v for k, v in norm_to_feature.items() if n in k or k in n]
            if candidates:
                mapped_team_names.append(candidates[0])
            else:
                missing.append(t)

    if missing:
        print("\nWarning: the following team names from Excel could not be mapped to features_final.csv:")
        for t in missing:
            print(' -', t)
    if not mapped_team_names:
        print("No teams mapped — aborting")
        return

    # deduplicate while preserving order
    seen = set()
    team_names = []
    for t in mapped_team_names:
        if t not in seen:
            seen.add(t)
            team_names.append(t)

    print(f"Proceeding with {len(team_names)} mapped teams (duplicates removed).")

    # validate requested team count
    if args.teams % 2 != 0 or args.teams <= 0:
        print(f"--teams must be a positive even number, got {args.teams}. Using 16.")
        sim_teams = 16
    else:
        sim_teams = args.teams
    if sim_teams > len(team_names):
        print(f"Requested {sim_teams} teams but only {len(team_names)} available. Reducing to available count.")
        sim_teams = len(team_names) - (len(team_names) % 2)


    # helper to safely pick columns
    def pick(s: pd.Series, *candidates):
        for c in candidates:
            if c in s.index:
                return s[c]
        return np.nan

    def build_match_features(teamA, teamB):
        fa = features[features['team name'] == teamA].iloc[0]
        fb = features[features['team name'] == teamB].iloc[0]
        A_points = pick(fa, 'points')
        B_points = pick(fb, 'points')
        A_goals = pick(fa, 'goals_scored_total', 'goals_scored', 'totalgoalsscored')
        B_goals = pick(fb, 'goals_scored_total', 'goals_scored', 'totalgoalsscored')
        A_winrate = pick(fa, 'win_rate_total', 'win_rate', 'winrate')
        B_winrate = pick(fb, 'win_rate_total', 'win_rate', 'winrate')
        X = pd.DataFrame({
            'A_points': [A_points],
            'A_winrate': [A_winrate],
            'A_gf': [A_goals],
            'A_ga': [pick(fa, 'goals_conceded_total', 'goals_conceded', 'totalgoalsconceded')],
            'B_points': [B_points],
            'B_winrate': [B_winrate],
            'B_gf': [B_goals],
            'B_ga': [pick(fb, 'goals_conceded_total', 'goals_conceded', 'totalgoalsconceded')]
        })
        return X

    def predict_match_winner(teamA, teamB, clf_model):
        try:
            X = build_match_features(teamA, teamB)
            pred = float(clf_model.predict(X.to_numpy())[0])
            # If model is a regressor, treat >0.5 as class 1; keep compatibility with classifiers
            return teamA if pred > 0.5 else teamB
        except Exception:
            # fallback: choose by points from features
            fa = features[features['team name'] == teamA].iloc[0]
            fb = features[features['team name'] == teamB].iloc[0]
            pa = pick(fa, 'points') or 0
            pb = pick(fb, 'points') or 0
            return teamA if pa >= pb else teamB

    # Interactive GUI helper functions
    def run_knockout_sequence(initial_teams, clf_model):
        # runs an interactive sequence where user chooses winners for each match
        rounds = []
        current = list(initial_teams)
        while len(current) > 1:
            pairs = [(current[i], current[i+1]) for i in range(0, len(current), 2)]
            rounds.append(pairs)
            # for interactive mode we will let the caller present choices; here we just compute predicted winners
            winners = [predict_match_winner(a, b, clf_model) for a, b in pairs]
            current = winners
        return rounds, current[0]

    def interactive_knockout_gui(all_teams, clf_model, args):
        if tk is None:
            print("Tkinter not available in this Python. Interactive mode disabled.")
            return

        root = tk.Tk()
        root.title('Knockout Manager')

        mainframe = ttk.Frame(root, padding=10)
        mainframe.grid(row=0, column=0, sticky='nsew')

        ttk.Label(mainframe, text='Choose action:').grid(row=0, column=0, sticky='w')

        # scrolling text area to show game-flow
        if scrolledtext is not None:
            txt = scrolledtext.ScrolledText(mainframe, width=80, height=20, wrap=tk.WORD)
            txt.grid(row=0, column=1, rowspan=10, padx=(10,0))
        else:
            txt = None

        def gui_log(msg: str):
            if txt is not None:
                txt.insert(tk.END, msg + "\n")
                txt.see(tk.END)
            else:
                print(msg)

        def on_auto():
            # auto simulate using current trained classifier and default team selection
            sim_teams = min(args.teams, len(all_teams))
            teams = all_teams[:sim_teams]
            np.random.seed(args.seed)
            np.random.shuffle(teams)
            # pair and auto simulate full tournament using existing logic
            groups = [teams[i*4:(i+1)*4] for i in range(len(teams)//4)] if len(teams) >= 4 else [teams]
            points_table = {team: 0 for team in teams}
            for group in groups:
                gui_log(f"\nGroup matches for group: {group}")
                for m1 in range(len(group)):
                    for m2 in range(m1+1, len(group)):
                        ta, tb = group[m1], group[m2]
                        winner = predict_match_winner(ta, tb, clf_model)
                        points_table[winner] += 3
                        gui_log(f"{ta} vs {tb} -> {winner}")
            # choose top 2 per group
            group_winners = []
            for group in groups:
                sorted_group = sorted(group, key=lambda t: points_table[t], reverse=True)
                group_winners.extend(sorted_group[:2])
            gui_log(f"\nAdvancing to Knockout: {group_winners}")
            # simulate knockout rounds with logging
            cur = list(group_winners)
            round_no = 1
            while len(cur) > 1:
                gui_log(f"\nKnockout Round {round_no}:")
                next_round = []
                for i in range(0, len(cur), 2):
                    a, b = cur[i], cur[i+1]
                    w = predict_match_winner(a, b, clf_model)
                    gui_log(f"{a} vs {b} -> {w} advances")
                    next_round.append(w)
                cur = next_round
                round_no += 1
            final = cur[0]
            gui_log(f"\nTournament Winner: {final}")
            if messagebox:
                messagebox.showinfo('Auto Result', f'Tournament Winner: {final}')

        def on_manual():
            # allow user to select starting round and then choose participants
            start_win = tk.Toplevel(root)
            start_win.title('Manual Knockout Start')
            ttk.Label(start_win, text='Start at round:').grid(row=0, column=0, sticky='w')
            start_var = tk.IntVar(value=1)
            ttk.Radiobutton(start_win, text='Round 1 (16 teams)', variable=start_var, value=1).grid(row=1, column=0, sticky='w')
            ttk.Radiobutton(start_win, text='Round 2 (8 teams)', variable=start_var, value=2).grid(row=2, column=0, sticky='w')
            ttk.Radiobutton(start_win, text='Round 3 (4 teams)', variable=start_var, value=3).grid(row=3, column=0, sticky='w')

            def start_selection():
                start_win.destroy()
                r = start_var.get()
                if r == 1:
                    needed = 16
                elif r == 2:
                    needed = 8
                else:
                    needed = 4
                select_win = tk.Toplevel(root)
                select_win.title(f'Select {needed} teams')
                vars = []
                for i in range(needed):
                    ttk.Label(select_win, text=f'Slot {i+1}').grid(row=i, column=0, sticky='w')
                    v = tk.StringVar(value=all_teams[i])
                    opt = ttk.OptionMenu(select_win, v, v.get(), *all_teams)
                    opt.grid(row=i, column=1, sticky='ew')
                    vars.append(v)

                def submit_slots():
                    chosen = [v.get() for v in vars]
                    # basic uniqueness check
                    if len(set(chosen)) != len(chosen):
                        messagebox.showerror('Error', 'Please pick unique teams for each slot.')
                        return
                    select_win.destroy()
                    # proceed round-by-round, letting user pick winners
                    current = chosen
                    gui_log(f"\nManual start teams: {current}")
                    round_no = r
                    while len(current) > 1:
                        # present match chooser
                        match_win = tk.Toplevel(root)
                        match_win.title(f'Round with {len(current)} teams')
                        pair_vars = []
                        pairs = []
                        for i in range(0, len(current), 2):
                            a, b = current[i], current[i+1]
                            ttk.Label(match_win, text=f'{a} vs {b}').grid(row=i//2, column=0, sticky='w')
                            pv = tk.StringVar(value=predict_match_winner(a, b, clf_model))
                            om = ttk.OptionMenu(match_win, pv, pv.get(), a, b)
                            om.grid(row=i//2, column=1, sticky='ew')
                            pair_vars.append(pv)
                            pairs.append((a, b))

                        def submit_winners():
                            winners = [pv.get() for pv in pair_vars]
                            # log each match choice
                            for (a, b), winner in zip(pairs, winners):
                                gui_log(f"{a} vs {b} -> {winner}")
                            if len(set(winners)) != len(winners):
                                # allow duplicates if they come from different matches; do not reject
                                pass
                            match_win.destroy()
                            nonlocal current
                            current = winners
                        ttk.Button(match_win, text='Confirm winners', command=submit_winners).grid(row=99, column=0, columnspan=2)
                        match_win.grab_set()
                        root.wait_window(match_win)
                        round_no += 1
                    # final winner
                    messagebox.showinfo('Manual Tournament', f'Tournament Winner: {current[0]}')

                ttk.Button(select_win, text='Submit teams', command=submit_slots).grid(row=needed+1, column=0, columnspan=2)
                select_win.grab_set()

            ttk.Button(start_win, text='Start', command=start_selection).grid(row=4, column=0)

        ttk.Button(mainframe, text='Auto predict from group stage', command=on_auto).grid(row=1, column=0, sticky='ew')
        # Monte-Carlo runs entry + button (allows running many shuffled tournaments)
        ttk.Label(mainframe, text='Runs:').grid(row=1, column=2, sticky='w')
        runs_var = tk.StringVar(value='100')
        runs_entry = ttk.Entry(mainframe, textvariable=runs_var, width=8)
        runs_entry.grid(row=1, column=3, sticky='w')

        def monte_carlo_runs(n_runs: int):
            try:
                n = int(n_runs)
            except Exception:
                gui_log('Invalid runs value')
                return
            if n <= 0:
                gui_log('Runs must be > 0')
                return

            # only use the provided all_teams list (already mapped)
            available = list(all_teams)
            if not available:
                gui_log('No teams available for Monte-Carlo')
                return
            sim_n = min(args.teams, len(available))
            rng = random.Random(args.seed)
            win_counts = {t: 0 for t in available}
            gui_log(f"Starting Monte-Carlo: {n} runs (seed={args.seed})...")
            for run in range(n):
                order = list(available)
                rng.shuffle(order)
                teams = order[:sim_n]
                groups = [teams[i*4:(i+1)*4] for i in range(len(teams)//4)] if len(teams) >=4 else [teams]

                points_table = {t: 0 for t in teams}
                for group in groups:
                    for i in range(len(group)):
                        for j in range(i+1, len(group)):
                            ta, tb = group[i], group[j]
                            winner = predict_match_winner(ta, tb, clf_model)
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
                        w = predict_match_winner(a, b, clf_model)
                        nxt.append(w)
                    cur = nxt

                win_counts[cur[0]] += 1

            sorted_counts = sorted(win_counts.items(), key=lambda kv: kv[1], reverse=True)
            gui_log('\nMonte-Carlo results (wins out of %d):' % n)
            for team, cnt in sorted_counts:
                if cnt > 0:
                    gui_log(f"{team}: {cnt} ({cnt/n:.3f})")
            top10 = '\n'.join([f"{t}: {c} ({c/n:.3f})" for t, c in sorted_counts[:10]])
            try:
                messagebox.showinfo('Monte-Carlo Results', f'Top winners:\n{top10}')
            except Exception:
                pass

        def on_mc():
            try:
                nr = int(runs_var.get())
            except Exception:
                nr = 100
            monte_carlo_runs(nr)

        ttk.Button(mainframe, text='Shuffle N times', command=on_mc).grid(row=1, column=4, sticky='ew')
        ttk.Button(mainframe, text='Manual knockout (choose teams)', command=on_manual).grid(row=2, column=0, sticky='ew')

        root.mainloop()

    # Create training data
    train_matches = []
    train_labels = []
    for ta in team_names:
        for tb in team_names:
            if ta == tb:
                continue
            Xvec = build_match_features(ta, tb).values[0]
            train_matches.append(Xvec)
            # improved pseudo-label: points difference, tie-break by goals
            fa = features[features['team name'] == ta].iloc[0]
            fb = features[features['team name'] == tb].iloc[0]
            A_points = pick(fa, 'points') or 0
            B_points = pick(fb, 'points') or 0
            A_goals = pick(fa, 'goals_scored_total', 'goals_scored', 'totalgoalsscored') or 0
            B_goals = pick(fb, 'goals_scored_total', 'goals_scored', 'totalgoalsscored') or 0
            score = (A_points - B_points) + 0.01 * (A_goals - B_goals)
            label = 1 if score > 0 else 0
            train_labels.append(label)

    X_train = np.array(train_matches)
    y_train = np.array(train_labels)

    # Train Linear Regression (we use regression output and threshold at 0.5 for winner selection)
    clf = LinearRegression()
    clf.fit(X_train, y_train)

    # If interactive mode requested, run GUI and skip automatic simulation
    if args.interactive:
        try:
            interactive_knockout_gui(team_names, clf, args)
        except Exception as e:
            print('Interactive GUI failed to start:', e)
        return

    # If user requested Monte-Carlo shuffle runs via CLI, run that and exit
    if getattr(args, 'shuffle_runs', 0) and args.shuffle_runs > 0:
        def perform_monte_carlo_cli(runs: int):
            # only use mapped team_names (they are already normalized/mapped)
            available = team_names
            if not available:
                print('No available teams for Monte-Carlo')
                return
            rng = random.Random(args.seed)
            sim_teams = min(sim_teams if 'sim_teams' in locals() else args.teams, len(available))
            win_counts = {t: 0 for t in available}
            print(f"Starting Monte-Carlo: {runs} runs (seed={args.seed})...")
            for _ in range(runs):
                order = list(available)
                rng.shuffle(order)
                teams = order[:sim_teams]
                groups = [teams[i*4:(i+1)*4] for i in range(len(teams)//4)] if len(teams) >=4 else [teams]

                points_table = {t: 0 for t in teams}
                for group in groups:
                    for i in range(len(group)):
                        for j in range(i+1, len(group)):
                            ta, tb = group[i], group[j]
                            winner = predict_match_winner(ta, tb, clf)
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
                        w = predict_match_winner(a, b, clf)
                        nxt.append(w)
                    cur = nxt

                win_counts[cur[0]] += 1

            sorted_counts = sorted(win_counts.items(), key=lambda kv: kv[1], reverse=True)
            print(f"\nMonte-Carlo results (wins out of {runs}):")
            for team, cnt in sorted_counts:
                if cnt > 0:
                    print(f"{team}: {cnt} ({cnt/runs:.3f})")

        perform_monte_carlo_cli(args.shuffle_runs)
        return

    # Tournament simulation
    np.random.seed(args.seed)
    tournament_teams = team_names[:sim_teams]
    np.random.shuffle(tournament_teams)

    # Split into four groups
    groups = [tournament_teams[i*4:(i+1)*4] for i in range(4)]
    points_table = {team: 0 for team in tournament_teams}
    print("\nGroup Stage:")
    for i, group in enumerate(groups):
        group_points = {team: 0 for team in group}
        print(f"\nGroup {chr(ord('A')+i)}: {group}")
        for m1 in range(4):
            for m2 in range(m1+1, 4):
                ta, tb = group[m1], group[m2]
                X = build_match_features(ta, tb)
                pred = float(clf.predict(X.to_numpy())[0])
                # for regression outputs, treat >0.5 as class 1 (teamA wins)
                if pred > 0.5:
                    winner = ta
                    group_points[winner] += 3
                    print(f"{ta} defeats {tb} (3 points for {ta})")
                else:
                    winner = tb
                    group_points[winner] += 3
                    print(f"{tb} defeats {ta} (3 points for {tb})")
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
            X = build_match_features(ta, tb)
            pred = float(clf.predict(X.to_numpy())[0])
            winner = ta if pred > 0.5 else tb
            print(f"{ta} vs {tb} -> {winner} advances")
            winners.append(winner)
        knockout = winners
        round_no += 1

    print(f"\nTournament Winner: {knockout[0]}")

    # Optionally save results
    if args.save:
        results = {
            'teams': tournament_teams,
            'groups': groups,
            'points_table': points_table,
            'advancing': group_winners,
            'winner': knockout[0]
        }
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Saved tournament results to {args.output}")
        except Exception as e:
            print("Failed to save results:", e)


if __name__ == '__main__':
    args = parse_args()
    main(args)
