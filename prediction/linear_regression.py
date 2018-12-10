import numpy as np
import pickle
from prediction.math_functions import root_mean_square_error, r2_scores

def linear_regression(test_x, test_y, train_x, train_y, iters=1000,alpha=0.001):
    x = np.array(train_x)
    y = np.array(train_y)
    b = np.zeros([len(y[0]), len(x[0])])

    g, cost = gradient_descent(x,y,b,iters,alpha)
    print(g)
    final_cost = compute_cost(x,y,g)
    print(final_cost)

    np_test_x = np.array(test_x)
    predicted_outcome = np_test_x.dot(g.T)
    print(predicted_outcome)
    np_test_y = np.array(test_y)
    print('rms: ', root_mean_square_error(np_test_y, predicted_outcome))
    print('r2 score: ', r2_scores(np_test_y, predicted_outcome))


def compute_cost(x, y, b):
    m = len(x)
    a = (x.dot(b.T) - y) ** 2
    return np.sum(a)/(2 * m)

def predict(test_x, g):
    return test_x.dot(g.T)

def gradient_descent(x, y, b, iters, alpha):
    cost = np.zeros(iters)
    for i in range(iters):
        h = np.sum(x * (x.dot(b.T) - y), axis=0)
        b -= (alpha / len(x)) * h
        cost[i] = compute_cost(x, y, b)
    return b, cost

if __name__ == '__main__':
    tr_set_x = pickle.load(open('train_set_x.pkl', 'rb'))
    tr_set_y = pickle.load(open('train_set_y.pkl', 'rb'))
    te_set_x = pickle.load(open('test_set_x.pkl', 'rb'))
    te_set_y = pickle.load(open('test_set_y.pkl', 'rb'))

    linear_regression(te_set_x, te_set_y, tr_set_x, tr_set_y)