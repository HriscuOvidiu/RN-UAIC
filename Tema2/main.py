import gzip
import pickle
import numpy as np
import math

from neural_network import NeuralNetwork

NUMBER_OF_ITERATIONS = 10

def preprocess_labels(y):
    final_y = []

    for el in y:
        new_y = np.zeros((10,), dtype=int)
        new_y[el] = 1
        final_y.append(new_y)

    return final_y

def calculate_accuracy(nn, x, y):
    right = 0
    for i in range(len(x)):
        prediction = nn.predict(x[i])
        if(prediction == y[i]):
            right+=1

    return 1.0 * right / len(x)

def shuffle(arr_of_arrs):
    permutation = np.random.permutation((len(arr_of_arrs[0]),)) - 1
    for arr in arr_of_arrs:
        arr = arr[permutation]

    return arr_of_arrs

file_name = 'mnist.pkl.gz'
f = gzip.open(file_name, 'rb')

train_set, valid_set, test_set = pickle.load(f)
f.close()

x_train = train_set[0]
y_train = train_set[1]
x_test = test_set[0]
y_test = test_set[1]

nn = NeuralNetwork()

print("Training...")
for i in range(NUMBER_OF_ITERATIONS):
    print("Iteration " + str(i + 1) + "...")
    [x_train, y_train] = shuffle([x_train, y_train])
    nn.train(x_train, y_train)

print("Calculating accuracy...")
accuracy = calculate_accuracy(nn, x_test, y_test)

print(accuracy)