import numpy as np
import pickle
from sklearn.metrics import r2_score, explained_variance_score
from math_functions import root_mean_squares
from feature_sets import prediction_feature_list


# linear regression classifier
# code written with help from various online resources
class lin_regression():
    def __init__(self, iters=2000, alpha=0.001):
        self.iters = iters
        self.alpha = alpha

    # fitting the classifier
    def fit(self, train_x, train_y):
        x = np.array(train_x)
        y = np.array(train_y)
        b = np.zeros([len(y[0]), len(x[0])])
        self.g, self.cost = self.gradient_descent(x, y, b, self.iters, self.alpha)

    # making the prediction
    def predict(self, test_x):
        return test_x.dot(self.g.T)

    # compute the cost
    def compute_cost(self, x, y, b):
        m = len(x)
        a = (x.dot(b.T) - y) ** 2
        return np.sum(a) / (2 * m)

    # gradient descent
    def gradient_descent(self, x, y, b, iters, alpha):
        cost = np.zeros(iters)
        for i in range(iters):
            h = np.sum(x * (x.dot(b.T) - y), axis=0)
            b -= (alpha / len(x)) * h
            cost[i] = self.compute_cost(x, y, b)
        return b, cost

# gets the datasets and does the linear regression classification
def linear_regression(team1_input, team2_input, team1_name, team2_name):
    tr_set_x = pickle.load(open('train_set_x.pkl', 'rb'))
    tr_set_y = pickle.load(open('train_set_y.pkl', 'rb'))
    te_set_x = pickle.load(open('test_set_x.pkl', 'rb'))
    te_set_y = pickle.load(open('test_set_y.pkl', 'rb'))

    np_test_x = np.array(te_set_x)
    np_test_y = np.array(te_set_y)

    lr = lin_regression()
    lr.fit(tr_set_x, tr_set_y)
    p_outcoum = lr.predict(np_test_x)


    np_team1 = np.array(team1_input)
    np_team2 = np.array(team2_input)

    print()
    print('------------------------------')
    print('       LinReg Prediction      ')
    print('------------------------------')
    team1_prediction = lr.predict(np_team1)
    team2_prediction = lr.predict(np_team2)
    print(team1_name, 'predicted goals: ', round(team1_prediction[0], 3))
    print(team2_name, 'predicted goals: ', round(team2_prediction[0], 3))
    print()

    if team1_prediction - team2_prediction > 0.3:
        print(team1_name, 'victory!')
    elif team1_prediction - team2_prediction < -0.3:
        print(team2_name, 'victory!')
    else:
        print('Draw!')

    print()
    print('------------------------------')
    print('         LinReg Metrics       ')
    print('------------------------------')
    print('root mean square: ', root_mean_squares(np_test_y, p_outcoum))
    print('r2_score: ', r2_score(np_test_y, p_outcoum))
    print('explained variance: ', explained_variance_score(np_test_y, p_outcoum))




# testing linear regression on a simple dataset
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

    linear_regression(team1, team2, 'chelsea', 'man city')


    # plots to find the best number of iterations for linear regression

    # exp_var = []
    # root_ms = []
    # r2sc = []
    # for i in range(0, 1000, 100):
    # ev, rmss, r2s = linear_regression(te_set_x, te_set_y, tr_set_x, tr_set_y, 1000)
    #linear_regression(te_set_x, te_set_y, tr_set_x, tr_set_y, 1000)
    # exp_var.append(ev)
    # root_ms.append(rmss)
    # r2sc.append(r2s)
        # print(i)
    # plt.figure(figsize=(12, 6))
    # plt.plot(range(0, 1000, 100), exp_var, color='green')
    # plt.title('Expected Variance')
    # plt.xlabel('K Value')
    # plt.ylabel('Expected Variance')
    # plt.show()
    #
    # plt.figure(figsize=(12, 6))
    # plt.plot(range(0, 1000, 100), root_ms, color='cyan')
    # plt.title('Root Mean Square')
    # plt.xlabel('K Value')
    # plt.ylabel('Root Mean Square')
    # plt.show()
    #
    # plt.figure(figsize=(12, 6))
    # plt.plot(range(0, 1000, 100), exp_var, color='orange')
    # plt.title('R2 Score')
    # plt.xlabel('K Value')
    # plt.ylabel('R2 Score')
    # plt.show()