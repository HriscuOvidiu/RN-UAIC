from node import Node

import numpy as np

class NeuralNetwork():
    
    def __init__(self):
        self.nodes = []

        for i in range(10):
            new_node = Node()
            self.nodes.append(new_node)    

    def train(self, x, y):
        for j in range(len(x)):
            for (i, node) in enumerate(self.nodes):            
                node.train(x[j], 1 * (y[j] == i))

    def predict(self, x):
        max = -1
        imax = -1
        for (i, node) in enumerate(self.nodes):
            output = node.predict(x)
            if(output > max):
                max = output
                imax = i
        
        return imax