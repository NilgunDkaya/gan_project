import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = keras.Sequential([
            layers.Flatten(input_shape=(28, 28, 1)),
            layers.Dense(512),
            layers.LeakyReLU(0.2),
            layers.Dense(256),
            layers.LeakyReLU(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
    
    def call(self, x):
        return self.net(x)