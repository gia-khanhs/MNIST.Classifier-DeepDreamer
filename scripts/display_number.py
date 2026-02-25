from matplotlib import pyplot as plt

from model.mnist_data import mnist

def display_number(X):
    pixels = X

    if pixels.shape != (28, 28):
        pixels = pixels.reshape((28, 28))

    plt.imshow(pixels, cmap='gray', vmin=0, vmax=1)
    plt.show()