import pickle
import numpy as np
from sklearn.metrics import r2_score, explained_variance_score
from math_functions import root_mean_squares
from feature_sets import prediction_feature_list

class knn_regression():
    def __init__(self, k, weights=None):
        self.k = k
        self.weights = weights

    # fitting the knn classifier
    def fit(self, x, y):
        np_x = np.array(x)
        np_y = np.array(y)
        self.training_features = np_x
        self.training_label = np_y

    # get neighbors for points
    def get_neighbors(self, train_set, test_set, k):
        np_train_set = np.array(train_set)
        np_test_set = np.array(test_set)
        euc_distance = np.sqrt(np.sum((np_train_set - np_test_set)**2, axis=1))
        return np.argsort(euc_distance)[0:k]

    # make prediction
    def prediction(self, test_instance):
        np_test_instance = np.array(test_instance)
        nearest_point = self.get_neighbors(self.training_features, np_test_instance, self.k)
        total = 0.0
        for i in nearest_point:
            total += self.training_label[i][0]
        return total/self.k

def knn(team1_input, team2_input, team1_name, team2_name):

    train_set_x = pickle.load(open('train_set_x.pkl', 'rb'))
    train_set_y = pickle.load(open('train_set_y.pkl', 'rb'))
    test_set_x = pickle.load(open('test_set_x.pkl', 'rb'))
    test_set_y = pickle.load(open('test_set_y.pkl', 'rb'))

    pred_outcomes = []

    k = knn_regression(40)
    k.fit(train_set_x, train_set_y)
    for i in test_set_x:
        ps = k.prediction(i)
        pred_outcomes.append(ps)

    print()
    print('------------------------------')
    print('        KNN Prediction        ')
    print('------------------------------')
    team1_prediction = k.prediction(team1_input)
    team2_prediction = k.prediction(team2_input)
    print(team1_name, 'predicted goals: ', round(team1_prediction, 3))
    print(team2_name, 'predicted goals: ', round(team2_prediction, 3))
    print()

    if team1_prediction - team2_prediction > 0.3:
        print(team1_name, 'victory!')
    elif team1_prediction - team2_prediction < -0.3:
        print(team2_name, 'victory!')
    else:
        print('Draw!')

    print()
    print('------------------------------')
    print('         KNN Metrics          ')
    print('------------------------------')
    print('root mean square: ', root_mean_squares(test_set_y, pred_outcomes))
    print('r2_score: ', r2_score(test_set_y, pred_outcomes))
    print('explained variance: ', explained_variance_score(test_set_y, pred_outcomes))
    print()

# test
if __name__ == '__main__':
    # test set features for chelsea vs man city (home games for chelsea)
    x = [[1,1,1,12,3,11,2,0,0,14,4,0,0,0,0,4,6,1,1],
         [2,1,1,15,6,15,3,0,0,11,2,0,0,0,0,6,6,3,0],
         [1,1,1,3,1,12,0,0,0,13,2,0,0,0,0,2,1,0,0],
         [2,1,1,10,8,14,2,0,0,12,3,0,0,0,0,5,4,1,0],
         [1,2,1,11,11,9,2,0,1,14,5,1,0,0,0,3,3,5,0],
         [2,2,1,11,3,15,3,0,0,10,1,0,1,0,0,7,1,3,0]]

    # test set features for chelsea vs man city (away game for chelsea)
    y = [[1,1,1,25,12,11,3,0,0,11,0,0,0,0,0,3,17,5,0],
         [2,1,1,17,6,12,3,0,0,11,4,0,0,0,0,6,5,4,2],
         [1,1,1,16,13,16,4,1,0,12,1,0,0,0,0,4,8,4,0],
         [2,1,1,6,2,12,4,0,0,14,1,0,0,0,0,2,1,2,1],
         [1,1,2,18,5,19,4,0,0,13,1,0,0,0,0,8,6,4,0],
         [2,1,2,10,1,13,2,0,0,18,3,0,0,0,0,3,3,3,1],
         [1,1,1,15,9,13,2,0,2,8,2,1,0,0,0,5,6,3,1],
         [2,1,1,10,2,8,3,0,0,13,1,1,0,0,0,4,5,1,0]]

    team1, team2 = prediction_feature_list(x, y, ret='avg')

    knn(team1, team2, 'chelsea', 'man city')
