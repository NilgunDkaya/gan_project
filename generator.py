import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(100,)),
            layers.Dense(512, activation='relu'),
            layers.Dense(1024, activation='relu'),
            layers.Dense(28*28, activation='tanh'),
            layers.Reshape((28, 28, 1))
        ])
    
    def call(self, x):
        return self.net(x)