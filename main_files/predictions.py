from main_files.knn import knn
from main_files.linear_regression import linear_regression
import numpy as np

def prediction():
    knn(np.array([1, 1, 1, 14, 5, 6, 3, 0, 0, 17, 2, 1, 0, 0, 0, 6, 3, 5, 0]),
        np.array([2, 1, 1, 16, 7, 17, 2, 0, 0, 6, 3, 0, 0, 0, 0, 5, 6, 5, 0]), 'chelsea', 'man city')
    linear_regression(np.array([1, 1, 1, 14, 5, 6, 3, 0, 0, 17, 2, 1, 0, 0, 0, 6, 3, 5, 0]),
                      np.array([2, 1, 1, 16, 7, 17, 2, 0, 0, 6, 3, 0, 0, 0, 0, 5, 6, 5, 0]), 'chelsea', 'man city')


if __name__ == '__main__':
    prediction()