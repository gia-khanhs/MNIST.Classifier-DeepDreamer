import random

import numpy as np

from model.activation_func import *
from model.mnist_data import mnist

from scripts.patterns import pattern

class mlp_generator:
    def __init__(self, mlp_classifier):
        self.mlp_classifier = mlp_classifier
        self.W1 = mlp_classifier.W1
        self.b1 = mlp_classifier.b1
        self.W2 = mlp_classifier.W2
        self.b2 = mlp_classifier.b2

        self.X = None
    
    def forward(self):
        self.Z1 = np.dot(self.W1, self.X) + self.b1
        self.A1 = leaky_ReLU(self.Z1)

        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = softmax(self.Z2)

    def back_prop(self):
        dZ2 = self.A2 - self.target
        
        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * dleaky_ReLU(self.Z1)

        dX = np.dot(self.W1.T, dZ1)
        
        return dX
    
    def gradient_descent(self, learning_rate):
        self.forward()
        dX = self.back_prop()
        # dX += 0.01 * self.X   # L2 penalty
        self.X = self.X - learning_rate * dX
        self.X = np.clip(self.X, 0.0, 1.0)

    def generate(self, number, pattern=pattern.black):
        self.X = pattern

        self.target = np.full((10, 1), 0)
        self.target[number, 0] = 1

        for i in range(1000):
            self.gradient_descent(0.5)

        prediction, prob = self.mlp_classifier.predict(self.X)
        print(prediction)
        print(prob[number])
        return self.X