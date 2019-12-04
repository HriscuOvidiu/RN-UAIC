import numpy as np

class Node():

    def __init__(self):
        self.weights = np.random.uniform(low = -1.0, high = 1.0, size = (784,))
        self.bias = np.random.rand() * 2 - 1
        self.learning_rate = 0.005

    def train(self, input, target):
        output = (np.dot(input, self.weights) + self.bias)/784
        # output = self.activation_function(output)
        # output = self.sigmoid(output)
        output = self.relu(output)
        # output = np.tanh(output)
        err = target - output   

        self.weights += self.learning_rate * err * input
        self.bias += self.learning_rate * err 

    def predict(self, x):
        return np.dot(self.weights, x)/784

    def activation_function(self, x):
        return 1 if x > 0 else 0
    
    def sigmoid(self, x):
        return 1/(1+ np.exp(-x))

    def relu(self, x):
        return max(0, x)