import numpy as np
from sklearn.preprocessing import normalize
import pickle
from main_files.math_functions import root_mean_squares
from sklearn.metrics import r2_score,explained_variance_score

class Neural_Network(object):
    def __init__(self):
        self.input_size = 19
        self.output_size = 1
        self.hidden_size = 5

        self.weight_1 = np.random.randn(self.input_size, self.hidden_size)
        self.weight_2 = np.random.randn(self.hidden_size, self.output_size)

    def forward(self, X):
        self.z = np.dot(X, self.weight_1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.weight_2)
        o = self.sigmoid(self.z3)
        return o

    def backward(self, X, y, o):
        self.o_error = y - o
        self.o_delta = self.o_error * self.sigmoid_derivative(o)

        self.z2_error = self.o_delta.dot(self.weight_2.T)
        self.z2_delta = self.z2_error * self.sigmoid_derivative(self.z2)

        self.weight_1 += X.T.dot(self.z2_delta)
        self.weight_2 += self.z2.T.dot(self.o_delta)

    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)

    def predict(self, test_set_x, test_set_y):
        print("Predicted data based on trained weights: ")
        print("Input (scaled): \n" + str(test_set_x))

        results = self.forward(test_set_x)

        pred_y = test_set_y
        print("Output: \n" + str(results))
        print()
        print('------------------------------')
        print()
        print('root mean square: ', root_mean_squares(pred_y, results))
        print('r2_score: ', r2_score(pred_y, results))
        print('explained variance: ', explained_variance_score(pred_y, results))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, s):
        # derivative of sigmoid
        return s * (1 - s)

def neural_network():
    tr_set_x = pickle.load(open('../datasets/train_set_x.pkl', 'rb'))
    tr_set_y = pickle.load(open('../datasets/train_set_y.pkl', 'rb'))
    te_set_x = pickle.load(open('../datasets/test_set_x.pkl', 'rb'))
    te_set_y = pickle.load(open('../datasets/test_set_y.pkl', 'rb'))

    x, y = np.array(tr_set_x), np.array(tr_set_y)
    pred_x, pred_y = np.array(te_set_x), np.array(te_set_y)

    # scale units
    x = normalize(x)
    y = normalize(y)
    pred_x = normalize(pred_x)
    pred_y = normalize(pred_y)

    NN = Neural_Network()
    for i in range(500):  # trains the NN 500 times
        # print("Loss: " + str(np.mean(np.square(y - NN.forward(x)))))  # mean sum squared loss
        NN.train(x, y)
    NN.predict(pred_x, pred_y)


if __name__ == '__main__':
    neural_network()
