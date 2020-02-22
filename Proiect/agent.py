from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense

class Agent():
    def __init__(self, env):
        self.env = env
        self.nn = self.init_nn()
        print("Init")

    def init_nn(self):
        model = Sequential([
            Dense(4, input_shape=(4, ), activation='relu'),
            Dense(2, activation='softmax')
        ])

        return model

    