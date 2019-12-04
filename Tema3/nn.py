import numpy as np
import json

class Layer():
	def __init__(self, number_of_inputs, number_of_nodes, activation):
		self.number_of_nodes = number_of_nodes 
		self.number_of_inputs = number_of_inputs
		self.activation = activation
		self.weights = 2 * np.random.random(size=(number_of_inputs, number_of_nodes)) - 1
		self.bias = 2 * np.random.random(size=(1, number_of_nodes)) - 1

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, loss):
        self.lr = 0.3     
        self.layers = []
        self.loss = loss

    def add_layer(self, layer):
        self.layers.append(layer)

    def __feedforward(self, x):
        outputs = []

        for i in range(len(self.layers)):
            if i == 0:
                v = x
            else:
                v = outputs[-1]

            out = self.layers[i].activation(np.dot(v, self.layers[i].weights) + self.layers[i].bias)
            outputs.append(out)

        return outputs

    def __backprop(self, x, y, outputs):
        deltas = []

        d = self.loss(outputs[-1], y)
        deltas.insert(0, d)

        for i in reversed(range(len(self.layers) - 1)):
            d = np.dot(deltas[0], np.transpose(self.layers[i+1].weights))
            d = d * self.layers[i].activation(outputs[i], d=True)
            deltas.insert(0, d) 

        for i in range(len(self.layers)):
            if i == 0:
                v = x
            else:
                v = outputs[i-1]

            self.layers[i].weights -= self.lr * np.dot(np.transpose(v), deltas[i])
            self.layers[i].bias -= self.lr * np.sum(deltas[i])

    def __shuffle(self, arr_of_arrs):
        permutation = np.random.permutation((len(arr_of_arrs[0]),)) - 1
        for arr in arr_of_arrs:
            arr = arr[permutation]

        return arr_of_arrs

    def save(self):
        data = []
        for (i, layer) in enumerate(self.layers):
            l = {}
            l['weights'] = layer.weights.tolist()
            l['bias'] = layer.bias.tolist()
            data.append(l)
            
        with open("last_json_data.json", "w") as f:
            json.dump(data, f)

    def load(self):
        with open("last_json_data.json", "r") as f:
            data = json.load(f)
            for i in range(len(data)):
                self.layers[i].weights = np.asarray(data[i]['weights'], dtype=float)
                self.layers[i].bias = np.asarray(data[i]['bias'], dtype=float)

    def test_acc(self, x, y):
        correct = 0
        
        for i in range(len(x)):
            out = self.__feedforward(x[i])[-1]
            pred = np.argmax(out)
            print(pred, y[i])
            if pred == y[i]:
                correct += 1
                
        return 1.0 * correct/len(x)

    def train(self, x, y, epochs, mini_batch_size):
        print("Training...")

        for i in range(epochs):
            print("Epoch " + str(i + 1))
            [x, y] = self.__shuffle([x, y])
            for i in range(int(len(x)/mini_batch_size)):
                mini_batch_x = x[mini_batch_size*i: mini_batch_size*(i+1)]
                mini_batch_y = y[mini_batch_size*i: mini_batch_size* (i+1)]
                outputs = self.__feedforward(mini_batch_x)
                self.__backprop(mini_batch_x, mini_batch_y, outputs)

        self.save()
