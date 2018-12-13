import plotly as py
import plotly.graph_objs as go
import pandas as pd
py.offline.init_notebook_mode(connected=True)

# Analysis of past 5 years of football events in the top 5 European Leagues
# By William Vagharfard and Sakib Md Bin Malek
def analysis():
    data = pd.read_csv('../football_events/merged.csv')

    # main_files by William Vagharfard
    print('-----------------------------------')
    print('             Data Info             ')
    print('-----------------------------------')
    print(data.info())

    # getting the total number of games for top, mid, and bottom teams
    games_home_high = data[(data['side'] == 1) & (data['ht_rank'] == 'high')]['id_odsp']
    games_away_high = data[(data['side'] == 2) & (data['at_rank'] == 'high')]['id_odsp']

    games_home_med = data[(data['side'] == 1) & (data['ht_rank'] == 'med')]['id_odsp']
    games_away_med = data[(data['side'] == 2) & (data['at_rank'] == 'med')]['id_odsp']

    games_home_low = data[(data['side'] == 1) & (data['ht_rank'] == 'low')]['id_odsp']
    games_away_low = data[(data['side'] == 2) & (data['at_rank'] == 'low')]['id_odsp']

    total_games_top = games_home_high.nunique() + games_away_high.nunique()
    total_games_med = games_home_med.nunique() + games_away_med.nunique()
    total_games_low = games_home_low.nunique() + games_away_low.nunique()


    print("\n William's Analysis\n")
    print()
    print('-----------------------------------')
    print('             Total Games           ')
    print('-----------------------------------')
    print(f'Top teams: {total_games_top}')
    print(f'Mid teams: {total_games_top}')
    print(f'Low teams: {total_games_top}')


    # getting all goals for top, mid, and bottom teams
    goal_home_top = data[(data['is_goal'] == 1) & (data['side'] == 1) & (data['ht_rank'] == 'high')]['event_type'].count()
    goal_away_top = data[(data['is_goal'] == 1) & (data['side'] == 2) & (data['at_rank'] == 'high')]['event_type'].count()

    goal_home_med = data[(data['is_goal'] == 1) & (data['side'] == 1) & (data['ht_rank'] == 'med')]['event_type'].count()
    goal_away_med = data[(data['is_goal'] == 1) & (data['side'] == 2) & (data['at_rank'] == 'med')]['event_type'].count()

    goal_home_low = data[(data['is_goal'] == 1) & (data['side'] == 1) & (data['ht_rank'] == 'low')]['event_type'].count()
    goal_away_low = data[(data['is_goal'] == 1) & (data['side'] == 2) & (data['at_rank'] == 'low')]['event_type'].count()


    top_total_goals = goal_home_top + goal_away_top
    med_total_goals = goal_home_med + goal_away_med
    low_total_goals = goal_home_low + goal_away_low

    print()
    print('-----------------------------------')
    print('             Total Goals           ')
    print('-----------------------------------')
    print(f'Top teams: {top_total_goals}')
    print(f'Mid teams: {med_total_goals}')
    print(f'Low teams: {low_total_goals}')

    # getting percentage of goals per game for top, mid, and bottom teams
    top_home_percent = round(goal_home_top/top_total_goals * 100, 2)
    med_home_percent = round(goal_home_med/med_total_goals * 100, 2)
    low_home_percent = round(goal_home_low/low_total_goals * 100, 2)

    top_away_percent = round(goal_away_top/top_total_goals * 100, 2)
    med_away_percent = round(goal_away_med/med_total_goals * 100, 2)
    low_away_percent = round(goal_away_low/low_total_goals * 100, 2)


    home_goals_percents = [f'{top_home_percent}%', f'{med_home_percent}%', f'{low_home_percent}%']
    away_goals_percents = [f'{top_away_percent}%', f'{med_away_percent}%', f'{low_away_percent}%']

    # plotting total home/away goals (and their percentages) for top, mid, bottom teams
    goals_data_home = go.Bar(
                    x = ['Top teams', 'Mid teams', 'Bottom teams'],
                    y = [goal_home_top, goal_home_med, goal_home_low],
                    text= home_goals_percents,
                    textposition='auto',
                    textfont=dict(size=40),
                    marker=dict(color='rgb(55, 66, 250)'),
                    hoverinfo = ['y', 'y', 'y'],
                    hoverlabel=dict(font=dict(size=30)),
                    name='Home')


    goals_data_away = go.Bar(
                    x = ['Top teams', 'Mid teams', 'Bottom teams'],
                    y = [goal_away_top, goal_away_med, goal_away_low],
                    text=away_goals_percents,
                    textposition='auto',
                    textfont=dict(size=40),
                    marker=dict(color='rgb(255, 71, 87)'),
                    hoverinfo= ['y', 'y', 'y'],
                    hoverlabel=dict(font=dict(size=30)),
                    name='Away')

    goals_data = [goals_data_home, goals_data_away]

    layout = go.Layout(
            barmode='group',
            font=dict(family='Avenir Medium', size=35),
            legend=dict(x=0.9, y=0.9, font=dict(size=50)),
            hovermode='closest',
            title='Total Home and Away Goals',
            xaxis=dict(tickfont=dict(size=40)))

    fig = go.Figure(data=goals_data, layout=layout)
    py.offline.plot(fig, auto_open=True, filename='../graphs/home_away_goals.html')

    # total goals from corners
    gct_h = data[(data['is_goal'] == 1) & (data['situation'] == 3) & (data['side'] == 1) &
                 (data['ht_rank'] == 'high')]['event_type'].count()
    gct_a = data[(data['is_goal'] == 1) & (data['situation'] == 3) & (data['side'] == 2) &
                 (data['at_rank'] == 'high')]['event_type'].count()

    gcm_h = data[(data['is_goal'] == 1) & (data['situation'] == 3) & (data['side'] == 1) &
                 (data['ht_rank'] == 'med')]['event_type'].count()
    gcm_a = data[(data['is_goal'] == 1) & (data['situation'] == 3) & (data['side'] == 2) &
                 (data['at_rank'] == 'med')]['event_type'].count()

    gcl_h = data[(data['is_goal'] == 1) & (data['situation'] == 3) & (data['side'] == 1) &
                 (data['ht_rank'] == 'low')]['event_type'].count()
    gcl_a = data[(data['is_goal'] == 1) & (data['situation'] == 3) & (data['side'] == 2) &
                 (data['at_rank'] == 'low')]['event_type'].count()

    goal_corner_top = gct_h + gct_a
    goal_corner_med = gcm_h + gcm_a
    goal_corner_low = gcl_h + gcl_a

    print()
    print('-----------------------------------')
    print('        Total Corner Goals         ')
    print('-----------------------------------')
    print(f'Top teams: {goal_corner_top}')
    print(f'Mid teams: {goal_corner_med}')
    print(f'Low teams: {goal_corner_low}')

    # total goals from open play
    to_h = data[(data['is_goal'] == 1) & (data['situation'] == 1) & (data['side'] == 1) &
                 (data['ht_rank'] == 'high')]['event_type'].count()
    to_a = data[(data['is_goal'] == 1) & (data['situation'] == 1) & (data['side'] == 2) &
                 (data['at_rank'] == 'high')]['event_type'].count()

    mo_h = data[(data['is_goal'] == 1) & (data['situation'] == 1) & (data['side'] == 1) &
                 (data['ht_rank'] == 'med')]['event_type'].count()
    mo_a = data[(data['is_goal'] == 1) & (data['situation'] == 1) & (data['side'] == 2) &
                 (data['at_rank'] == 'med')]['event_type'].count()

    lo_h = data[(data['is_goal'] == 1) & (data['situation'] == 1) & (data['side'] == 1) &
                 (data['ht_rank'] == 'low')]['event_type'].count()
    lo_a = data[(data['is_goal'] == 1) & (data['situation'] == 1) & (data['side'] == 2) &
                 (data['at_rank'] == 'low')]['event_type'].count()


    goal_openplay_top = to_h + to_a
    goal_openplay_med = mo_h + mo_a
    goal_openplay_low = lo_h + lo_a

    print()
    print('-----------------------------------')
    print('       Total Open Play Goals       ')
    print('-----------------------------------')
    print(f'Top teams: {goal_openplay_top}')
    print(f'Mid teams: {goal_openplay_med}')
    print(f'Low teams: {goal_openplay_low}')

    # total goals from set pieces
    tsp_h = data[(data['is_goal'] == 1) & (data['situation'] == 2) & (data['side'] == 1) &
                 (data['ht_rank'] == 'high')]['event_type'].count()
    tsp_a = data[(data['is_goal'] == 1) & (data['situation'] == 2) & (data['side'] == 2) &
                 (data['at_rank'] == 'high')]['event_type'].count()

    msp_h = data[(data['is_goal'] == 1) & (data['situation'] == 2) & (data['side'] == 1) &
                 (data['ht_rank'] == 'med')]['event_type'].count()
    msp_a = data[(data['is_goal'] == 1) & (data['situation'] == 2) & (data['side'] == 2) &
                 (data['at_rank'] == 'med')]['event_type'].count()

    lsp_h = data[(data['is_goal'] == 1) & (data['situation'] == 2) & (data['side'] == 1) &
                 (data['ht_rank'] == 'low')]['event_type'].count()
    lsp_a = data[(data['is_goal'] == 1) & (data['situation'] == 2) & (data['side'] == 2) &
                 (data['at_rank'] == 'low')]['event_type'].count()


    goal_setpiece_top = tsp_h + tsp_a
    goal_setpiece_med = msp_h + msp_a
    goal_setpiece_low = lsp_h + lsp_a

    print()
    print('-----------------------------------')
    print('           Set Piece Goals         ')
    print('-----------------------------------')
    print('set piece goals are from free kicks where there is not a direct goal')
    print('e.g.: a freekick is headed in')
    print(f'Top teams: {goal_setpiece_top}')
    print(f'Mid teams: {goal_setpiece_med}')
    print(f'Low teams: {goal_setpiece_low}')

    # total goals from freekicks
    tfk_h = data[(data['is_goal'] == 1) & (data['situation'] == 4) & (data['side'] == 1) &
                 (data['ht_rank'] == 'high')]['event_type'].count()
    tfk_a = data[(data['is_goal'] == 1) & (data['situation'] == 4) & (data['side'] == 2) &
                 (data['at_rank'] == 'high')]['event_type'].count()

    mfk_h = data[(data['is_goal'] == 1) & (data['situation'] == 4) & (data['side'] == 1) &
                 (data['ht_rank'] == 'med')]['event_type'].count()
    mfk_a = data[(data['is_goal'] == 1) & (data['situation'] == 4) & (data['side'] == 2) &
                 (data['at_rank'] == 'med')]['event_type'].count()

    lfk_h = data[(data['is_goal'] == 1) & (data['situation'] == 4) & (data['side'] == 1) &
                 (data['ht_rank'] == 'low')]['event_type'].count()
    lfk_a = data[(data['is_goal'] == 1) & (data['situation'] == 4) & (data['side'] == 2) &
                 (data['at_rank'] == 'low')]['event_type'].count()


    goal_freekick_top = tfk_h + tfk_a
    goal_freekick_med = mfk_h + mfk_a
    goal_freekick_low = lfk_h + lfk_a

    print()
    print('-----------------------------------')
    print('           Free Kick Goals         ')
    print('-----------------------------------')
    print('free kick goals are goals scored directly from a free kick (no assists)')
    print(f'Top teams: {goal_freekick_top}')
    print(f'Mid teams: {goal_freekick_med}')
    print(f'Low teams: {goal_freekick_low}')

    # plotting types of goals for high, mid, and bottom teams
    top_lab = ['Open Play','Set Piece','Corner Kick','Free Kick']
    top_val = [goal_openplay_top, goal_setpiece_top, goal_corner_top, goal_freekick_top]
    colorsss = ['rgb(6, 82, 221)', 'rgb(250, 152, 58)',
                'rgb(11, 232, 129)',
                'rgb(245, 59, 87)']

    med_lab = ['Open Play','Set Piece','Corner Kick','Free Kick']
    med_val = [goal_openplay_med, goal_setpiece_med, goal_corner_med, goal_freekick_med]

    low_lab = ['Open Play','Set Piece','Corner Kick','Free Kick']
    low_val = [goal_openplay_low, goal_setpiece_low, goal_corner_low, goal_freekick_low]

    x = go.Pie(
        labels=top_lab,
        values=top_val,
        marker=dict(colors=colorsss),
        textinfo='percent',
        title='Top Teams',
        titleposition='bottom center',
        domain={'x': [0.10, 0.35]})

    y = go.Pie(
        labels=med_lab,
        values=med_val,
        textinfo='percent',
        title='Mid Teams',
        titleposition='bottom center',
        domain={'x': [0.40, 0.65]})

    z = go.Pie(
        labels=low_lab,
        values=low_val,
        textinfo='percent',
        title='Bottom Teams',
        titleposition='bottom center',
        domain={'x': [0.70, 0.95]})

    layout = go.Layout(
        font=dict(family='Avenir Heavy', size=40),
        legend=dict(x=0.49, y=0.9, font=dict(size=40)),
        title='Goal types by team level'
    )

    fig=go.Figure(data=[x, y, z], layout=layout)
    py.offline.plot(fig, filename='../graphs/team_goal_types.html')


    # getting own goals by team level
    own_goals_highA = data[(data['event_type2'] == 15) & (data['side'] == 2) &
                         (data['ht_rank'] == 'high')]['event_type'].count()
    own_goals_highB = data[(data['event_type2'] == 15) & (data['side'] == 1) &
                         (data['at_rank'] == 'high')]['event_type'].count()


    own_goals_medA = data[(data['event_type2'] == 15) & (data['side'] == 2) &
                         (data['ht_rank'] == 'med')]['event_type'].count()
    own_goals_medB = data[(data['event_type2'] == 15) & (data['side'] == 1) &
                         (data['at_rank'] == 'med')]['event_type'].count()

    own_goals_lowA = data[(data['event_type2'] == 15) & (data['side'] == 2) &
                         (data['ht_rank'] == 'low')]['event_type'].count()
    own_goals_lowB = data[(data['event_type2'] == 15) & (data['side'] == 1) &
                         (data['at_rank'] == 'low')]['event_type'].count()

    own_goals_high = own_goals_highA + own_goals_highB
    own_goals_med = own_goals_medA + own_goals_medB
    own_goals_low = own_goals_lowA + own_goals_lowB

    owng_gameT = own_goals_high/total_games_top
    owng_gameM = own_goals_med/total_games_med
    owng_gameL = own_goals_low/total_games_low

    print()
    print('-----------------------------------')
    print('        Own Goals per Game         ')
    print('-----------------------------------')
    print(f'Top teams: {round(owng_gameT, 3)}')
    print(f'Mid teams: {round(owng_gameM, 3)}')
    print(f'Low teams: {round(owng_gameL, 3)}')


    # plotting own goals by team level
    colors = ['rgb(112, 161, 255)', 'rgb(123, 237, 159)', 'rgb(255, 165, 2)']
    own_goals = go.Bar(
                    x = ['Top teams', 'Mid teams', 'Bottom teams'],
                    y = [owng_gameT, owng_gameM, owng_gameL],
                    text= [round(owng_gameT, 3), round(owng_gameM, 3), round(owng_gameL, 3)],
                    textposition='outside',
                    textfont=dict(size=40),
                    marker=dict(color=colors),
                    hoverinfo = ['y', 'y', 'y'],
                    hoverlabel=dict(font=dict(size=30)),
                    name='Home')

    goals_data = [own_goals]

    layout = go.Layout(
            font=dict(family='Avenir Medium', size=35),
            legend=dict(x=0.9, y=0.9, font=dict(size=50)),
            hovermode='closest',
            title='Own Goals per game by team level',
            xaxis=dict(tickfont=dict(size=40)))

    fig = go.Figure(data=goals_data, layout=layout)

    py.offline.plot(fig, auto_open=True, filename='../graphs/own_goals.html')

    # getting offsides per game by team level
    off_lowA = data[(data['event_type'] == 9) & (data['side'] == 1) & (data['ht_rank'] == 'low')]['event_type'].count()
    off_lowB = data[(data['event_type'] == 9) & (data['side'] == 2) & (data['at_rank'] == 'low')]['event_type'].count()
    offsides_low = off_lowA + off_lowB

    off_medA = data[(data['event_type'] == 9) & (data['side'] == 1) & (data['ht_rank'] == 'med')]['event_type'].count()
    off_medB = data[(data['event_type'] == 9) & (data['side'] == 2) & (data['at_rank'] == 'med')]['event_type'].count()
    offsides_med = off_medA + off_medB

    off_highA = data[(data['event_type'] == 9) & (data['side'] == 1) & (data['ht_rank'] == 'high')]['event_type'].count()
    off_highB = data[(data['event_type'] == 9) & (data['side'] == 2) & (data['at_rank'] == 'high')]['event_type'].count()
    offsides_high = off_highA + off_highB

    offs_gameT = offsides_high/total_games_top
    offs_gameM = offsides_med/total_games_med
    offs_gameL = offsides_low/total_games_low

    print()
    print('-----------------------------------')
    print('         Offsides per Game         ')
    print('-----------------------------------')
    print(f'Top teams: {round(offs_gameT, 3)}')
    print(f'Mid teams: {round(offs_gameM, 3)}')
    print(f'Low teams: {round(offs_gameL, 3)}')


    # plotting offsides per game by team level
    colors = ['rgb(112, 161, 255)', 'rgb(123, 237, 159)', 'rgb(255, 165, 2)']

    offsides = go.Bar(
                    x = ['Top teams', 'Mid teams', 'Bottom teams'],
                    y = [offs_gameT, offs_gameM, offs_gameL],
                    text= [round(offs_gameT, 3), round(offs_gameM, 3), round(offs_gameL, 3)],
                    textposition='outside',
                    textfont=dict(size=40),
                    marker=dict(color=colors),
                    hoverinfo = ['y', 'y', 'y'],
                    hoverlabel=dict(font=dict(size=30)),
                    name='Home')

    off_data = [offsides]

    layout = go.Layout(
            font=dict(family='Avenir Medium', size=35),
            legend=dict(x=0.9, y=0.9, font=dict(size=50)),
            hovermode='closest',
            title='Offsides per game by team level',
            xaxis=dict(tickfont=dict(size=40)))


    fig = go.Figure(data=off_data, layout=layout)

    py.offline.plot(fig, auto_open=True, filename='../graphs/offsides.html')

    # getting total shots off target and total shot attempts by team level
    # we get the off target shots because it is easier to find the total shots ON target by subtracting off-target from the total attempts
    # this is because this data-set splits on-target shots into 3 different categories
    off_tar_highH = data[(data['shot_outcome'] == 2) & (data['side'] == 1) & (data['ht_rank'] == 'high')]['event_type'].count()
    off_tar_highA = data[(data['shot_outcome'] == 2) & (data['side'] == 2) & (data['at_rank'] == 'high')]['event_type'].count()
    off_tar_medH = data[(data['shot_outcome'] == 2) & (data['side'] == 1) & (data['ht_rank'] == 'med')]['event_type'].count()
    off_tar_medA = data[(data['shot_outcome'] == 2) & (data['side'] == 2) & (data['at_rank'] == 'med')]['event_type'].count()
    off_tar_lowH = data[(data['shot_outcome'] == 2) & (data['side'] == 1) & (data['ht_rank'] == 'low')]['event_type'].count()
    off_tar_lowA = data[(data['shot_outcome'] == 2) & (data['side'] == 2) & (data['at_rank'] == 'low')]['event_type'].count()

    attempts_highH = data[(data['event_type'] == 1) & (data['side'] == 1) & (data['ht_rank'] == 'high')]['event_type'].count()
    attempts_highA = data[(data['event_type'] == 1) & (data['side'] == 2) & (data['at_rank'] == 'high')]['event_type'].count()
    attempts_medH = data[(data['event_type'] == 1) & (data['side'] == 1) & (data['ht_rank'] == 'med')]['event_type'].count()
    attempts_medA = data[(data['event_type'] == 1) & (data['side'] == 2) & (data['at_rank'] == 'med')]['event_type'].count()
    attempts_lowH = data[(data['event_type'] == 1) & (data['side'] == 1) & (data['ht_rank'] == 'low')]['event_type'].count()
    attempts_lowA = data[(data['event_type'] == 1) & (data['side'] == 2) & (data['at_rank'] == 'low')]['event_type'].count()

    off_tar_high = off_tar_highH + off_tar_highA
    off_tar_med = off_tar_medH + off_tar_medA
    off_tar_low = off_tar_lowH + off_tar_lowA

    attempts_high = attempts_highH + attempts_highA
    attempts_med = attempts_medH + attempts_medA
    attempts_low = attempts_lowH + attempts_lowA


    # plotting total percentage of shots that are on target
    # again this is done by taking the off-target percent and subtracting it from 100 %
    offtarget_high_perc = round((off_tar_high / attempts_high) * 100, 2)
    offtarget_med_perc = round((off_tar_med / attempts_med) * 100, 2)
    offtarget_low_perc = round((off_tar_low / attempts_low) * 100, 2)

    print()
    print('-----------------------------------')
    print('    Percent of shots on target     ')
    print('-----------------------------------')
    print(f'Top teams: {100 - offtarget_high_perc}')
    print(f'Mid teams: {100 - offtarget_med_perc}')
    print(f'Low teams: {100 - offtarget_low_perc}')

    colors = ['rgb(112, 161, 255)', 'rgb(123, 237, 159)', 'rgb(255, 165, 2)']

    on_target = go.Bar(
        x=['Top teams', 'Mid teams', 'Bottom teams'],
        y=[(100 - offtarget_high_perc), (100 - offtarget_med_perc), (100 - offtarget_low_perc)],
        text=[f'{(100 - offtarget_high_perc)}%', f'{(100 - offtarget_med_perc)}%', f'{(100 - offtarget_low_perc)}%'],
        textposition='outside',
        textfont=dict(size=40),
        marker=dict(color=colors),
        hoverinfo=['y', 'y', 'y'],
        hoverlabel=dict(font=dict(size=30)),
        name='On Target')

    layout = go.Layout(
        font=dict(family='Avenir Medium', size=35),
        legend=dict(x=0.9, y=0.9, font=dict(size=50)),
        hovermode='closest',
        title='Percent of shots on target',
        xaxis=dict(tickfont=dict(size=40)))

    on_tar_data = [on_target]

    fig = go.Figure(data=on_tar_data, layout=layout)

    py.offline.plot(fig, auto_open=True, filename='../graphs/on_target_goals.html')


    # card and foul main_files by team levels
    # by Sakib Md Bin Malek

    count_of_games = {}
    count_of_foul = {}
    count_of_cards = {}
    count_of_foul_home_team = {}
    count_of_cards_home_team = {}
    team_rank = ['high', 'med', 'low']

    condition_for_card = ((data['event_type'] >= 4) & (data['event_type'] <= 6))
    condition_for_home_team_action = (data['side'] == 1)
    condition_for_foul = (data['event_type'] == 3)
    print("\nSakib's Analysis\n")
    for rank_home in team_rank:
        for rank_away in team_rank:
            condition = (data['ht_rank'] == rank_home) & (data['at_rank'] == rank_away)
            count_of_games[(rank_home+'-'+rank_away)] = data[condition]['id_odsp'].nunique()
            count_of_foul[(rank_home+'-'+rank_away)] = len(data[condition & condition_for_foul])
            count_of_cards[(rank_home+'-'+rank_away)] = len(data[condition & condition_for_card])
            count_of_foul_home_team[(rank_home+'-'+rank_away)] = len(data[condition & condition_for_foul & condition_for_home_team_action])
            count_of_cards_home_team[(rank_home+'-'+rank_away)] = len(data[condition & condition_for_card & condition_for_home_team_action])

    for rank_home in team_rank:
        for rank_away in team_rank:
            games = count_of_games[(rank_home + '-' + rank_away)]
            fouls = count_of_foul[(rank_home + '-' + rank_away)]
            fouls_by_home_team = count_of_foul_home_team[(rank_home + '-' + rank_away)]
            cards = count_of_cards[(rank_home + '-' + rank_away)]
            cards_by_home_team = count_of_cards_home_team[(rank_home + '-' + rank_away)]

            print()
            print(rank_home, ' - ', rank_away, 'games: ', games)
            print('fouls: ', fouls, '\nfouls/game: ', round(fouls/games, 3))
            print('    home team: ', round(fouls_by_home_team/games, 3), '\n    away team: ', round((fouls - fouls_by_home_team) / games, 3))
            print('cards: ', cards, '\ncards/game: ', round(cards / games, 3))
            print('    home team: ', round(cards_by_home_team / games, 3), '\n    away team: ', round((cards - cards_by_home_team) / games, 3))


if __name__ == '__main__':
    analysis()