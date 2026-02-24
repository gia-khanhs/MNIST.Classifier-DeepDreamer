import numpy as np

from model.mnist_data import mnist

from scripts.display_number import display_number

class mlp_generator:
    def __init__(self, mlp_classifier):
        self.mlp_classifier = mlp_classifier
        self.W1 = mlp_classifier.W1
        self.b1 = mlp_classifier.b1
        self.W2 = mlp_classifier.W2
        self.b2 = mlp_classifier.b2

        self.X = None
    
    def ReLU(self, Z):
        return np.maximum(0, Z)

    def softmax(self, Z):
        expZ = np.exp(Z)
        denominator = np.sum(expZ, axis=0, keepdims=True)
        return expZ / denominator

    def forward(self):
        self.Z1 = np.dot(self.W1, self.X) + self.b1
        self.A1 = self.ReLU(self.Z1)

        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = self.softmax(self.Z2)

    def back_prop(self):
        dZ2 = self.A2 - self.target
        
        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * (self.Z1 > 0)

        dX = np.dot(self.W1.T, dZ1)
        
        return dX
    
    def gradient_descent(self, learning_rate):
        self.forward()
        dX = self.back_prop()
        dX += 0.01 * self.X   # L2 penalty
        self.X = self.X - learning_rate * dX
        self.X = np.clip(self.X, 0.0, 1.0)

    def generate(self, number):
        self.X = np.random.random((784, 1))
        self.X = mnist.test.img[0].T
        self.X = self.X.reshape((784, 1))
        display_number(self.X)
        self.target = np.zeros((10, 1))
        self.target[number, 0] = 1

        for i in range(10000):
            self.gradient_descent(0.005)

        prediction, prob = self.mlp_classifier.predict(self.X)
        print(prediction)
        print(prob[number])
        return self.X