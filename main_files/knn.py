from main_files.math_functions import euclidean_distance
import pickle
import numpy as np
from sklearn.metrics import r2_score, explained_variance_score
from main_files.math_functions import root_mean_squares

class knn_regression():
    def __init__(self, k, weights=None):
        self.k = k
        self.weights = weights

    # fit(train_X, train_Y)
    def fit(self, x, y):
        np_x = np.array(x)
        np_y = np.array(y)
        self.training_features = np_x
        self.training_label = np_y

    # get_neighbors(train_X, test_Y)
    def get_neighbors(self, training_set, test_set, k):
        np_train_set = np.array(training_set)
        np_test_set = np.array(test_set)
        assert (k <= len(np_train_set)), 'K must be less than or equal to the length of the training set.'
        distances = []
        for i in range(len(np_train_set)):
            dist = euclidean_distance(np_train_set[i], np_test_set)
            distances.append((np_train_set[i], dist, i))
        distances.sort(key=lambda x: x[1])
        neighbors = []
        for i in range(k):
            neighbors.append(distances[i][2])
        return neighbors

    def get_neighbors_v(self, train_set, test_set, k):
        np_train_set = np.array(train_set)
        np_test_set = np.array(test_set)
        euc_distance = np.sqrt(np.sum((np_train_set - np_test_set)**2, axis=1))
        return np.argsort(euc_distance)[0:k]

    # predict(test_X)
    def prediction(self, test_instance):
        np_test_instance = np.array(test_instance)
        nearest_point = self.get_neighbors_v(self.training_features, np_test_instance, self.k)
        total = 0.0
        for i in nearest_point:
            total += self.training_label[i][0]
        return total/self.k

def knn(team1_input, team2_input, team1_name, team2_name):

    train_set_x = pickle.load(open('../datasets/train_set_x.pkl', 'rb'))
    train_set_y = pickle.load(open('../datasets/train_set_y.pkl', 'rb'))
    test_set_x = pickle.load(open('../datasets/test_set_x.pkl', 'rb'))
    test_set_y = pickle.load(open('../datasets/test_set_y.pkl', 'rb'))

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

# test
if __name__ == '__main__':
    knn([1, 1, 1, 14, 5, 6, 3, 0, 0, 17, 2, 1, 0, 0, 0, 6, 3, 5, 0],
        [2, 1, 1, 16, 7, 17, 2, 0, 0, 6, 3, 0, 0, 0, 0, 5, 6, 5, 0],
        'chelsea', 'man city')
