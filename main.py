import numpy as np 
import matplotlib.pyplot as plt

from model.mnist_data import mnist

from model.mlp_classifier import load_model
from model.mlp_generator import mlp_generator

from scripts.patterns import pattern
from scripts.display_number import display_number
from scripts.classify_drawing import drawing_window

#=====================================================
mnist_classifier = load_model(training=False)
classifier_window = drawing_window(mnist_classifier)

mnist_generator = mlp_generator(mnist_classifier)
img = mnist_generator.generate(3, pattern.chessboard)
display_number(img)
#=====================================================
# display_number(mnist.train.img[0])

# classifier_window.run()