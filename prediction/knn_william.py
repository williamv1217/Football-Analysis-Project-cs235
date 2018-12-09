from prediction.math_functions import euclidean_distance


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

    # predict(test_X)
    def prediction(self, test_instance):
        nearest_point = self.get_neighbors(self.training_features, test_instance, self.k)
        total = 0.0
        for i in nearest_point:
            total += self.training_label[i]
        return total/self.k

if __name__ == '__main__':
    print('hello')
    # iris = load_iris()
    # x = iris.data
    # y = iris.target
    #
    # xtrain, xtest, ytrain, ytest = cross_validation.train_test_split(iris.data, iris.target, test_size=0.4, random_state=1)
    #
    # preds = []
    #
    # k = 5
    # clf = knn_regression(2)
    # clf.fit(xtrain, ytrain)
    #
    # ps = clf.prediction(xtest)
    # difference = 0.0
    # difference_sq = 0.0
    # for i in range(len(ytest)):
    #     print(str(ps[i][0]) + ' ' + str(ytest[i][0]))
    #     difference = difference + abs(ps[i][0] - ytest[i][0])
    #     difference_sq = difference_sq + (ps[i][0] - ytest[i][0]) * (
    #                 ps[i][0] - ytest[i][0])

    # for a in xtrain:
    #     ps = clf.prediction(a)
    #     preds.append(ps)
    # iris_target_pred = np.array(preds)
    # print(np.array(preds))

    # clf.fit(train_set, labels)
    # iris_pred = []
    # for x in train_set:
    #     pred = clf.prediction(x)
    #     iris_pred.append(pred)
    # print(iris_pred)