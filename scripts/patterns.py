import random

import numpy as np

from model.mnist_data import mnist

class _pattern:
    def __init__(self):
        self.__black = np.zeros((784, 1))

        self.__gray = np.full((784, 1), 0.5)

        self.__white = np.ones((784, 1))
        
        self.__vertical_stripe = np.full((784, 1), 0.0)
        for i in range(0, 784, 2):
            self.__vertical_stripe[i][0] = 1.0

        self.__chessboard = np.full((784, 1), 0.0)
        for i in range(0, 28):
            for j in range(0, 28):
                self.__chessboard[28 * i + j][0] = (i + j) % 2

    def noise(self):
        return np.random.random((784, 1))
    
    def white(self):
        return self.__white
    
    def gray(self):
        return self.__gray

    def black(self):
        return self.__black
    
    def vertical_stripe(self):
        return self.__vertical_stripe

    def chessboard(self):
        return self.__chessboard

    def random_test_number(self):
        N = np.max(mnist.test.img.shape)

        i = random.randint(0, N - 1)

        return mnist.test.img[i].reshape((784, 1))
    
pattern = _pattern()