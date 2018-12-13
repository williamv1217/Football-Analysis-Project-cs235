import numpy as np

# returns a prediction feature list for a team.
# the input 'ret' can be set to either 'avg', 'last', or None
# the input is two 2d lists which we have to get from our "merged.csv" dataset
# the output is two arrays, one for team1 and one for team2
# the output is in the following format:
#   the entire list                                 (ret=None)
#   the entire list with each variable averaged     (ret="avg")
#   only the final game in the list                 (ret="last")
def prediction_feature_list(team1_home_list, team1_away_list, ret=None):

    team1_home = np.array(team1_home_list)
    team1_away = np.array(team1_away_list)

    team1 = []
    team2 = []

    for row in range(len(team1_home)):
        if team1_home[row][0] == 1:
            team1.append(team1_home[row])
        else:
            team2.append(team1_home[row])

    for row in range(len(team1_away)):
        if team1_away[row][0] == 1:
            team2.append(team1_away[row])
        else:
            team1.append(team1_away[row])

    team1 = np.array(team1)
    team2 = np.array(team2)

    team1_avg = np.average(team1, axis=0)
    team2_avg = np.average(team2, axis=0)

    if ret == 'avg':
        return team1_avg, team2_avg
    elif ret == 'last':
        return team1[len(team1) - 1], team2[len(team2) - 1]
    else:
        return team1, team2





if __name__ == '__main__':
    team1_home = [[1, 1, 2, 13, 5, 15, 1, 0, 0, 19, 2, 0, 0, 0, 0, 6, 3, 4, 0]
        , [2, 1, 2, 10, 4, 19, 3, 0, 1, 15, 2, 0, 0, 0, 0, 4, 3, 3, 0]
        , [1, 1, 1, 7, 3, 13, 3, 0, 0, 10, 2, 0, 0, 0, 0, 2, 1, 3, 1]
        , [2, 1, 1, 15, 7, 10, 1, 0, 0, 13, 3, 0, 0, 0, 0, 2, 7, 5, 1]
        , [1, 2, 1, 16, 10, 6, 0, 0, 0, 8, 1, 0, 0, 0, 0, 5, 6, 5, 0]
        , [2, 2, 1, 17, 10, 8, 3, 0, 0, 6, 1, 0, 0, 0, 0, 7, 6, 4, 0]
        , [1, 1, 1, 14, 5, 6, 3, 0, 0, 17, 2, 1, 0, 0, 0, 6, 3, 5, 0]
        , [2, 1, 1, 16, 7, 17, 2, 0, 0, 6, 3, 0, 0, 0, 0, 5, 6, 5, 0]]

    team1_away = [[1, 1, 1, 19, 4, 12, 3, 0, 0, 13, 0, 0, 0, 0, 0, 7, 10, 2, 0]
        , [2, 1, 1, 9, 7, 13, 6, 1, 0, 12, 2, 0, 0, 0, 0, 4, 3, 2, 0]
        , [1, 1, 2, 12, 5, 16, 4, 0, 0, 6, 2, 0, 0, 0, 0, 2, 3, 5, 2]
        , [2, 1, 2, 8, 3, 5, 2, 0, 0, 16, 0, 0, 0, 0, 0, 3, 5, 0, 0]]

    print(len(team1_home))

    a, b = prediction_feature_list(team1_home, team1_away)
    print(a)
    print(b)

    a, b = prediction_feature_list(team1_home, team1_away, ret='avg')
    print(a)
    print(b)

    a, b = prediction_feature_list(team1_home, team1_away, ret='last')
    print(a)
    print(b)