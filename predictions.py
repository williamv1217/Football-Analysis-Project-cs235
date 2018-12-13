from knn import knn
from linear_regression import linear_regression
from feature_sets import prediction_feature_list
from neural_network import neural_network

def prediction():
    # test set features for chelsea vs man city (home games for chelsea)
    x = [[1, 1, 1, 12, 3, 11, 2, 0, 0, 14, 4, 0, 0, 0, 0, 4, 6, 1, 1],
         [2, 1, 1, 15, 6, 15, 3, 0, 0, 11, 2, 0, 0, 0, 0, 6, 6, 3, 0],
         [1, 1, 1, 3, 1, 12, 0, 0, 0, 13, 2, 0, 0, 0, 0, 2, 1, 0, 0],
         [2, 1, 1, 10, 8, 14, 2, 0, 0, 12, 3, 0, 0, 0, 0, 5, 4, 1, 0],
         [1, 2, 1, 11, 11, 9, 2, 0, 1, 14, 5, 1, 0, 0, 0, 3, 3, 5, 0],
         [2, 2, 1, 11, 3, 15, 3, 0, 0, 10, 1, 0, 1, 0, 0, 7, 1, 3, 0]]

    # test set features for chelsea vs man city (away game for chelsea)
    y = [[1, 1, 1, 25, 12, 11, 3, 0, 0, 11, 0, 0, 0, 0, 0, 3, 17, 5, 0],
         [2, 1, 1, 17, 6, 12, 3, 0, 0, 11, 4, 0, 0, 0, 0, 6, 5, 4, 2],
         [1, 1, 1, 16, 13, 16, 4, 1, 0, 12, 1, 0, 0, 0, 0, 4, 8, 4, 0],
         [2, 1, 1, 6, 2, 12, 4, 0, 0, 14, 1, 0, 0, 0, 0, 2, 1, 2, 1],
         [1, 1, 2, 18, 5, 19, 4, 0, 0, 13, 1, 0, 0, 0, 0, 8, 6, 4, 0],
         [2, 1, 2, 10, 1, 13, 2, 0, 0, 18, 3, 0, 0, 0, 0, 3, 3, 3, 1],
         [1, 1, 1, 15, 9, 13, 2, 0, 2, 8, 2, 1, 0, 0, 0, 5, 6, 3, 1],
         [2, 1, 1, 10, 2, 8, 3, 0, 0, 13, 1, 1, 0, 0, 0, 4, 5, 1, 0]]

    # change ret= to 'avg' in order to get the avg for each element instead of the last rows
    # this will allow you to see the predicion if we take all games from past 5 years into consideration
    team1, team2 = prediction_feature_list(x, y, ret='last')


    knn(team1, team2, 'chelsea', 'man city')
    linear_regression(team1, team2, 'chelsea', 'man city')
    neural_network()


if __name__ == '__main__':
    prediction()