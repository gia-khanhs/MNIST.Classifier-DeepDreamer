import random

import numpy as np

from model.mnist_data import mnist

class _pattern:
    def __init__(self):
        self.noise = None

        self.black = np.full((784, 1), 0)

        self.white = np.full((784, 1), 1)
        
        self.vertical_stripe = np.full((784, 1), 0)
        for i in range(0, 784, 2):
            self.vertical_stripe[i][0] = 1

        self.chessboard = np.full((784, 1), 0)
        for i in range(0, 28):
            for j in range(0, 28):
                self.chessboard[28 * i + j][0] = (i + j) % 2

    def noise(self):
        self.noise = self.X = np.random.random((784, 1))
        return self.noise
    
    def random_test_number(self):
        N = np.max(mnist.test.img.shape)

        i = random.randint(0, N - 1)

        return mnist.test.img[i]
    
pattern = _pattern()