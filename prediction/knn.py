from prediction.math_functions import euclidean_distance
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from prediction.math_functions import r2_scores, rmse_metric

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

if __name__ == '__main__':

    train_set_x = pickle.load(open('train_set_x.pkl', 'rb'))
    train_set_y = pickle.load(open('train_set_y.pkl', 'rb'))
    test_set_x = pickle.load(open('test_set_x.pkl', 'rb'))
    test_set_y = pickle.load(open('test_set_y.pkl', 'rb'))

    pred_outcomes = []
    clf = knn_regression(50)
    clf.fit(train_set_x, train_set_y)
    for i in test_set_x:
        ps = clf.prediction(i)
        pred_outcomes.append(ps)
    print(pred_outcomes)

    print(len(pred_outcomes), len(test_set_y))
    difference = 0.0
    difference_sq = 0.0
    for i in range(len(test_set_y)):
        print(str(pred_outcomes[i]) + ' ' + str(test_set_y[i][0]))
        difference = difference + abs(pred_outcomes[i] - test_set_y[i][0])
        difference_sq = difference_sq + (pred_outcomes[i] - test_set_y[i][0]) * (
                pred_outcomes[i] - test_set_y[i][0])

    print()
    print('------------------------------')
    print()
    print('difference: ', difference)
    print('difference squared: ', difference_sq)
    print('root mean square: ', rmse_metric(test_set_y, pred_outcomes))
    print('r2_score: ', r2_score(test_set_y, pred_outcomes))
    print('explained variance: ', explained_variance_score(test_set_y, pred_outcomes))