import numpy as np
import pickle
from sklearn.metrics import r2_score, explained_variance_score
from prediction.math_functions import root_mean_squares
import matplotlib.pyplot as plt

class lin_regression():
    def __init__(self, iters=2000, alpha=0.001):
        self.iters = iters
        self.alpha = alpha

    def fit(self, train_x, train_y):
        x = np.array(train_x)
        y = np.array(train_y)
        b = np.zeros([len(y[0]), len(x[0])])
        self.g, self.cost = self.gradient_descent(x, y, b, self.iters, self.alpha)

    def predict(self, test_x):
        return test_x.dot(self.g.T)

    def compute_cost(self, x, y, b):
        m = len(x)
        a = (x.dot(b.T) - y) ** 2
        return np.sum(a) / (2 * m)

    def gradient_descent(self, x, y, b, iters, alpha):
        cost = np.zeros(iters)
        for i in range(iters):
            h = np.sum(x * (x.dot(b.T) - y), axis=0)
            b -= (alpha / len(x)) * h
            cost[i] = self.compute_cost(x, y, b)
        return b, cost

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

if __name__ == '__main__':

    team1_home =[[1,1,2,13,5,15,1,0,0,19,2,0,0,0,0,6,3,4,0]
,[2,1,2,10,4,19,3,0,1,15,2,0,0,0,0,4,3,3,0]
,[1,1,1,7,3,13,3,0,0,10,2,0,0,0,0,2,1,3,1]
,[2,1,1,15,7,10,1,0,0,13,3,0,0,0,0,2,7,5,1]
,[1,2,1,16,10,6,0,0,0,8,1,0,0,0,0,5,6,5,0]
,[2,2,1,17,10,8,3,0,0,6,1,0,0,0,0,7,6,4,0]
,[1,1,1,14,5,6,3,0,0,17,2,1,0,0,0,6,3,5,0]
,[2,1,1,16,7,17,2,0,0,6,3,0,0,0,0,5,6,5,0]]

    team1_away = [[1,1,1,19,4,12,3,0,0,13,0,0,0,0,0,7,10,2,0]
,[2,1,1,9,7,13,6,1,0,12,2,0,0,0,0,4,3,2,0]
,[1,1,2,12,5,16,4,0,0,6,2,0,0,0,0,2,3,5,2]
,[2,1,2,8,3,5,2,0,0,16,0,0,0,0,0,3,5,0,0]]


    team1_home= np.array(team1_home)
    team1_away = np.array(team1_away)

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

    team1 = np.average(team1, axis=0)
    team2 = np.average(team2, axis=0)

    linear_regression(team1, team2, 'chelsea', 'brighton')



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



# # old lin reg
# def lin_reg(test_x, test_y, train_x, train_y, iters=2000,alpha=0.001):
#     x = np.array(train_x)
#     y = np.array(train_y)
#     b = np.zeros([len(y[0]), len(x[0])])
#
#     g, cost = gradient_descent(x,y,b,iters,alpha)
#     # final_cost = compute_cost(x,y,g)
#
#     np_test_x = np.array(test_x)
#     predicted_outcome = np_test_x.dot(g.T)
#     np_test_y = np.array(test_y)
#
#     print('------------------------------')
#     print()
#     print('root mean square: ', root_mean_squares(np_test_y, predicted_outcome))
#     print('r2_score: ', r2_score(np_test_y, predicted_outcome))
#     print('explained variance: ', explained_variance_score(np_test_y, predicted_outcome))
#     return g

# def compute_cost(x, y, b):
#     m = len(x)
#     a = (x.dot(b.T) - y) ** 2
#     return np.sum(a)/(2 * m)

# def gradient_descent(x, y, b, iters, alpha):
#     cost = np.zeros(iters)
#     for i in range(iters):
#         h = np.sum(x * (x.dot(b.T) - y), axis=0)
#         b -= (alpha / len(x)) * h
#         cost[i] = compute_cost(x, y, b)
#     return b, cost