import numpy as np
import gzip
import pickle
from nn import NeuralNetwork, Layer
import activations
import losses

def LoadMNISTSets():
	file_name = 'mnist.pkl.gz'
	f = gzip.open(file_name, 'rb')

	train_set, valid_set, test_set = pickle.load(f)
	f.close()

	x_train = np.array(train_set[0])
	y_train = np.array(train_set[1])
	x_test = np.array(test_set[0])
	y_test = np.array(test_set[1])
	return [x_train, y_train, x_test, y_test]

def one_hot_encoding(y):
	enc = []

	for i in y:
		base = np.zeros(10)

		base[i] = 1
		enc.append(base)

	return np.asarray(enc)

[x_train, y_train, x_test, y_test] = LoadMNISTSets()

y_train = one_hot_encoding(y_train)

nn = NeuralNetwork(784, 100, 10, loss=losses.cross_entropy)
# x_train = activations.sigmoid(x_train)
# x_test = activations.sigmoid(x_test)
nn.add_layer(Layer(784, 784, activation=activations.sigmoid))
nn.add_layer(Layer(784, 100, activation=activations.sigmoid))
nn.add_layer(Layer(100, 10, activation=activations.softmax))

# nn.train(x_train, y_train, epochs=30, mini_batch_size=100)
nn.load()

print(nn.test_acc(x_test, y_test))