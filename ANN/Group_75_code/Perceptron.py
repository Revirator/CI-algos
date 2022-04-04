import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, inputs, learning_rate):
        self.bias = np.random.uniform()
        self.weights = np.random.uniform(size=inputs)
        self.learning_rate = learning_rate

    def transfer_function(self, data):
        return np.dot(data, self.weights)

    def activation_function(self, data):
        return 1 if self.transfer_function(data) + self.bias >= 0 else 0

    def learn(self, data, label):
        loss = label - self.activation_function(data)
        self.weights += (self.learning_rate * loss) * data
        self.bias += self.learning_rate * loss

    def accuracy(self, labels):
        correct = 0
        total = 0
        for el, label in zip(inputs, labels):
            total += 1
            if self.activation_function(el) == label:
                correct += 1
        return correct / total


inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

AND = np.array([0, 0, 0, 1])
OR = np.array([0, 1, 1, 1])
XOR = np.array([0, 1, 1, 0])
epochs = 100
epochs_list = [*range(0, epochs)]

# AND
perceptron_AND = Perceptron(2, 0.1)
AND_accuracy = []
for i in range(0, epochs):
    for el, label in zip(inputs, AND):
        perceptron_AND.learn(el, label)
    AND_accuracy.append(perceptron_AND.accuracy(AND))

plt.plot(epochs_list, AND_accuracy)
plt.title("AND accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

# OR
perceptron_OR = Perceptron(2, 0.1)
OR_accuracy = []
for i in range(0, epochs):
    for el, label in zip(inputs, OR):
        perceptron_OR.learn(el, label)
    OR_accuracy.append(perceptron_OR.accuracy(OR))

plt.plot(epochs_list, OR_accuracy)
plt.title("OR accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

# XOR
perceptron_XOR = Perceptron(2, 0.1)
XOR_accuracy = []
for i in range(0, epochs):
    for el, label in zip(inputs, XOR):
        perceptron_XOR.learn(el, label)
    XOR_accuracy.append(perceptron_XOR.accuracy(XOR))

plt.plot(epochs_list, XOR_accuracy)
plt.title("XOR accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()
