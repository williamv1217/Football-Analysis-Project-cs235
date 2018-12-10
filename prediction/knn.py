from prediction.math_functions import euclidean_distance
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from prediction.math_functions import root_mean_square_error, r2_scores

class knn_regression():
    def __init__(self, k, weights=None):
        self.k = k
        self.weights = weights

    # fit(train_X, train_Y)
    def fit(self, training_features, training_label):
        self.training_features = training_features
        self.training_label = training_label

    # get_neighbors(train_X, test_Y)
    def get_neighbors(self, training_set, test_set, k):
        assert (k <= len(training_set)), 'K must be less than or equal to the length of the training set.'
        distances = []
        for i in range(len(training_set)):
            dist = euclidean_distance(training_set[i], test_set)
            distances.append((training_set[i], dist, i))
        distances.sort(key=lambda x: x[1])
        neighbors = []
        for i in range(k):
            neighbors.append(distances[i][2])
        return neighbors

    def get_neighbors_v(self, train_set, test_set, k):
        euc_distance = np.sqrt(np.sum((train_set - test_set)**2, axis=1))
        return np.argsort(euc_distance)[0:k]

    # predict(test_X)
    def prediction(self, test_instance):
        nearest_point = self.get_neighbors_v(self.training_features, test_instance, self.k)
        total = 0.0
        for i in nearest_point:
            total += self.training_label[i][0]
        return total/self.k

if __name__ == '__main__':

    tr_set_x = pickle.load(open('train_set_x.pkl', 'rb'))
    tr_set_y = pickle.load(open('train_set_y.pkl', 'rb'))
    te_set_x = pickle.load(open('test_set_x.pkl', 'rb'))
    te_set_y = pickle.load(open('test_set_y.pkl', 'rb'))

    np_tr_set_x = np.array(tr_set_x)
    np_tr_set_y = np.array(tr_set_y)
    np_te_set_x = np.array(te_set_x)
    np_te_set_y = np.array(te_set_y)

    print('create train sets')
    training_set_x, training_set_y = np_tr_set_x, np_tr_set_y

    print('create test sets')
    test_set_x, test_set_y = np_te_set_x, np_te_set_y

    p = []
    clf = knn_regression(10)
    clf.fit(training_set_x, training_set_y)
    count = 0
    print(len(test_set_x))
    for i in test_set_x:
        count += 1
        ps = clf.prediction(i)
        p.append(ps)
        print(count)
    print(p)
    mse = np.sqrt(mean_squared_error(np_te_set_y, p))
    print('rmse: ', mse)
    print('wrmse: ', root_mean_square_error(np_te_set_y, p))
    print('r2: ', r2_score(np_te_set_y, p))
    print('wr2: ', r2_scores(np_te_set_y, p))