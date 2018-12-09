import math
import numpy as np
import sklearn as sk

def euclidean_distance(a, b):
    assert (len(a) == len(b)), 'Euclidean Distance feature length of the input must match'
    points = zip(a, b)
    dist = [pow(x - y, 2) for (x, y) in points]
    return math.sqrt(sum(dist))

def get_distance(data1, data2):
    points = zip(data1, data2)
    diffs_squared_distance = [pow(a - b, 2) for (a, b) in points]
    return math.sqrt(sum(diffs_squared_distance))


# def root_mean_square_error(y, y_prediction):
#     return np.sqrt(np.mean((y - y_prediction)**2))
#

# def accuracy(y, y_pred):
#     error = np.sum((y - y_pred)**2)
#     acc = 100 - (error/len(y))*100
#     return acc

# test

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