import numpy as np
import pickle
from sklearn.metrics import r2_score, explained_variance_score
from main_files.math_functions import root_mean_squares
from datasets.add_lists_script import prediction_feature_list


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
    tr_set_x = pickle.load(open('../datasets/train_set_x.pkl', 'rb'))
    tr_set_y = pickle.load(open('../datasets/train_set_y.pkl', 'rb'))
    te_set_x = pickle.load(open('../datasets/test_set_x.pkl', 'rb'))
    te_set_y = pickle.load(open('../datasets/test_set_y.pkl', 'rb'))

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

    x = [[1,2,2,17,9,6,1,0,0,12,2,0,0,0,0,2,8,7,0]
,[2,2,2,12,4,12,4,0,0,6,1,0,0,0,0,3,4,5,0]]
    y = [[1,3,1,6,0,19,4,0,0,9,1,1,0,0,0,2,2,2,0]
,[2,3,1,13,5,9,3,0,0,19,2,1,0,0,0,4,5,3,1]]

    team1, team2 = prediction_feature_list(x, y, ret='avg')
    print(team1)
    print(team2)

    linear_regression(team1, team2, 'chelsea', 'man utd')



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