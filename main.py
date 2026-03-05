import numpy as np 
import matplotlib.pyplot as plt

from model.mnist_data import mnist

from model.mlp_classifier import load_model
from model.mlp_dreamer import mlp_generator

from scripts.patterns import Pattern
from scripts.display_number import display_number
from scripts.classify_drawing import drawing_window
from scripts.deep_dream import deep_dream

#=====================================================
mnist_classifier = load_model(training=False)
classifier_window = drawing_window(mnist_classifier)

mnist_dreamer = mlp_generator(mnist_classifier)

pattern = classifier_window.run()

# dreamer = deep_dream(mnist_dreamer, Pattern.black())
dreamer = deep_dream(mnist_dreamer, pattern)
dreamer.dream()
#=====================================================
# display_number(mnist.train.img[0])

# classifier_window.run()