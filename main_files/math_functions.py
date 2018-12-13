import math
import numpy as np


# ---------- kNN stuff ----------
def euclidean_distance(a, b):
    assert (len(a) == len(b)), 'Euclidean Distance feature length of the input must match'
    points = zip(a, b)
    dist = [pow(x - y, 2) for (x, y) in points]
    return math.sqrt(sum(dist))

def get_distance(data1, data2):
    points = zip(data1, data2)
    diffs_squared_distance = [pow(a - b, 2) for (a, b) in points]
    return math.sqrt(sum(diffs_squared_distance))


# ---------- metrics ----------

def root_mean_squares(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return np.sqrt(mean_error)[0]

# ---------- neural network stuff ----------
def sigmoid_function(x):
    if type(x[0]) == list or type(x[0]) == np.ndarray:
        output = [[0 for i in range(len(x[0]))] for j in range(len(x))]
        for i in range(len(x)):
            for j in range(len(x[0])):
                output[i][j] = (1.0 / (1.0 + math.pow(math.e, -(x[i][j]))))
        return output
    else:
        output2 = []
        for value in x:
            output2.append(1.0 / (1.0 + math.pow(math.e, -(value))))
        return output2

# ---------- testing ----------

if __name__ == '__main__':
    data1 = [2, 2, 2]
    data2 = [2, 2, 2]
    a = np.array(data1)
    b = np.array(data2)
    d2 = euclidean_distance(a, b)
    print(d2)
    d3 = np.linalg.norm(a-b)
    print(d3)
    print(euclidean_distance([1,2],[2,3]))
    # print(accuracy(a, b))