import numpy as np

def ReLU(Z):
    return np.maximum(0, Z)

def leaky_ReLU(Z, alpha=0.1):
    return np.where(Z > 0, Z, alpha * Z)

def dleaky_ReLU(Z, alpha=0.1):
    return np.where(Z > 0, 1.0, alpha)

# def softmax(Z):
#     expZ = np.exp(Z)
#     denominator = np.sum(expZ, axis=0, keepdims=True)
#     return expZ / denominator

def softmax(Z):
    Z = Z - np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Z)
    return expZ / (np.sum(expZ, axis=0, keepdims=True) + 1e-9)