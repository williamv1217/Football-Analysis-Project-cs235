from datasets.commons import rankToRank
#from sets import Set
import pickle

# eventType
ATTEMPT = 1
CORNER = 2
FOUL = 3
YELLOW = 4
YELLOW2 = 5
RED = 6
SUBSTITUTION = 7
FREE_KICK = 8
OFFSIDE = 9
HAND_BALL = 10
PENALTY = 11

# eventType2
KEY_PASS = 1
FAILED_THROUGH_PASS = 2

# shotOutcome
ON_TARGET = 1
OFF_TARGET = 2
BLOCKED = 3
HIT_THE_BAR = 4


def dataset(dataset):
    game_ids = dataset["id_odsp"].unique()

    set_x = []
    set_y = []
    game_count = 0
    print(len(game_ids))
    for game in game_ids:
        # isolating the games
        game_entries = dataset[dataset["id_odsp"].str.match(game)]
        game_count += 1
        for i in [1, 2]:
            condition_for_i_team_action = (game_entries['side'] == i)
            game_entries_for_i_team = game_entries[condition_for_i_team_action]
            # game_entries_for_away = game_entries[~condition_for_home_team_action]

            ##conditions for event types
            condition_for_attempt = (game_entries_for_i_team['event_type'] == ATTEMPT)
            condition_for_corner = (game_entries_for_i_team['event_type'] == CORNER)
            condition_for_foul = (game_entries_for_i_team['event_type'] == FOUL)
            condition_for_yellow = (game_entries_for_i_team['event_type'] == YELLOW)
            condition_for_yellow2 = (game_entries_for_i_team['event_type'] == YELLOW2)
            condition_for_red = (game_entries_for_i_team['event_type'] == RED)
            condition_for_freekick = (game_entries_for_i_team['event_type'] == FREE_KICK)
            condition_for_offside = (game_entries_for_i_team['event_type'] == OFFSIDE)
            condition_for_handball = (game_entries_for_i_team['event_type'] == HAND_BALL)
            condition_for_penalty = (game_entries_for_i_team['event_type'] == PENALTY)

            ##conditions for event types 2
            condition_for_keypass = (game_entries_for_i_team['event_type2'] == KEY_PASS)
            condition_for_failedthroughpass = (game_entries_for_i_team['event_type2'] == FAILED_THROUGH_PASS)

            ##conditions for shot outcome
            condition_for_ontarget = (game_entries_for_i_team['shot_outcome'] == ON_TARGET)
            condition_for_offtarget = (game_entries_for_i_team['shot_outcome'] == OFF_TARGET)
            condition_for_blocked = (game_entries_for_i_team['shot_outcome'] == BLOCKED)
            condition_for_goalpost = (game_entries_for_i_team['shot_outcome'] == HIT_THE_BAR)

            htgoal = game_entries['fthg'].values[0]
            awgoal = game_entries['ftag'].values[0]
            ht = game_entries['ht'].values[0]
            at = game_entries['at'].values[0]

            if i == 1:
                pres_team = ht
            else:
                pres_team = at

            count_of_attempt = len(game_entries_for_i_team[condition_for_attempt])
            count_of_corner = len(game_entries_for_i_team[condition_for_corner])
            count_of_foul = len(game_entries_for_i_team[condition_for_foul])
            count_of_yellow = len(game_entries_for_i_team[condition_for_yellow])
            count_of_yellow2 = len(game_entries_for_i_team[condition_for_yellow2])
            count_of_red = len(game_entries_for_i_team[condition_for_red])
            count_of_freekick = len(game_entries_for_i_team[condition_for_freekick])
            count_of_offside = len(game_entries_for_i_team[condition_for_offside])
            count_of_handball = len(game_entries_for_i_team[condition_for_handball])
            count_of_penalty = len(game_entries_for_i_team[condition_for_penalty])

            count_of_keypass = len(game_entries_for_i_team[condition_for_keypass])
            count_of_failedthroughpass = len(game_entries_for_i_team[condition_for_failedthroughpass])

            count_of_ontarget = len(game_entries_for_i_team[condition_for_ontarget])
            count_of_offtarget = len(game_entries_for_i_team[condition_for_offtarget])
            count_of_blocked = len(game_entries_for_i_team[condition_for_blocked])
            count_of_goalpost = len(game_entries_for_i_team[condition_for_goalpost])

            htrank = game_entries['ht_rank'].values[0]
            awrank = game_entries['at_rank'].values[0]

            home_team_rank = rankToRank(htrank)
            away_team_rank = rankToRank(awrank)

            pres_x = [i, home_team_rank, away_team_rank, count_of_attempt, count_of_corner, count_of_foul,
                      count_of_yellow, count_of_yellow2, count_of_red, count_of_freekick, count_of_offside,
                      count_of_handball, count_of_penalty, count_of_keypass, count_of_failedthroughpass,
                      count_of_ontarget, count_of_offtarget, count_of_blocked, count_of_goalpost]
            if i == 1:
                pres_y = [htgoal]
            else:
                pres_y = [awgoal]

            set_x.append(pres_x)
            set_y.append(pres_y)
        print(game_count)

    output_x = open('test_set_x.pkl', 'wb')
    output_y = open('test_set_y.pkl', 'wb')

    pickle.dump(set_x, output_x)
    pickle.dump(set_y, output_y)
    return set_x, set_y
