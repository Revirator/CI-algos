import numpy as np


# Function used to compute the activation using the sigmoid function
def compute_activation(data, weights, bias):
    z = np.dot(weights, data) + bias
    return 1 / (1 + np.exp(-z))


# Function used to compute the mean squared error
def loss(output, y):
    return ((output - y) ** 2) / 2


## START BACKPROPAGATION

# Function used to update the weights between the hidden and the output layer
def update_output(output_final, output_hidden, y, size_hidden):
    update_weights_output = ((output_final - y) * output_final * (
            np.ones(output_final.shape) - output_final)).reshape((7, 1)) @ output_hidden.reshape((1, size_hidden))
    update_bias_output = ((output_final - y) * output_final * (np.ones(output_final.shape) - output_final))
    return update_weights_output, update_bias_output


# Function used to update the weights between the input and the hidden layer
def update_hidden(output_final, output_hidden, weights_output, y, line, size_hidden):
    update_weights_hidden = ((output_final - y) * output_final * (
        (np.ones(output_final.shape) - output_final)).reshape((1, 7)) @ weights_output).reshape((size_hidden, 1)) * (
                               (output_hidden * (np.ones(output_hidden.shape) - output_hidden))).reshape(
        (size_hidden, 1)) @ line.reshape((1, 10))
    update_bias_hidden = (((output_final - y) * output_final * (
        (np.ones(output_final.shape) - output_final)).reshape((1, 7)) @ weights_output) * (
                             (output_hidden * (np.ones(output_hidden.shape) - output_hidden)))).reshape((size_hidden,))
    return update_weights_hidden, update_bias_hidden


## END BACKPROPAGATOION

# Function used to train the ANN using forward propagation
def train(x_train, y_train, size_input, size_hidden, size_output, n_epochs,
          learning_rate, x_test, y_test, stop_num, epsilon, weight=None):
    # np.random.seed(42)
    if weight is None:
        weights_hidden = np.random.uniform(low=-1, high=1, size=(size_hidden, size_input))
        bias_hidden = np.random.uniform(low=-1, high=1, size=size_hidden)
        weights_output = np.random.uniform(low=-1, high=1, size=(size_output, size_hidden))
        bias_output = np.random.uniform(low=-1, high=1, size=size_output)
    else:
        weights_hidden = np.full(shape=(size_hidden, size_input), fill_value=weight).astype('float64')
        bias_hidden = np.full(shape=size_hidden, fill_value=weight).astype('float64')
        weights_output = np.full(shape=(size_output, size_hidden), fill_value=weight).astype('float64')
        bias_output = np.full(shape=size_output, fill_value=weight).astype('float64')

    accuracies = []
    test_accuracies = []
    errors = []
    test_errors = []
    no_change_iterations = 0

    for k in range(n_epochs):
        err = []
        test_err = []
        # learning_rate = 1 / n_epochs
        for i, line in enumerate(x_train):
            output_hidden = compute_activation(line, weights_hidden, bias_hidden)
            output_final = compute_activation(output_hidden, weights_output, bias_output)
            y = np.zeros(output_final.shape)
            y[y_train[i] - 1] = 1
            err.append(loss(output_final, y))

            update_weights_output, update_bias_output = update_output(output_final, output_hidden, y, len(weights_hidden))
            update_weights_hidden, update_bias_hidden = update_hidden(output_final, output_hidden, weights_output, y, line, len(weights_hidden))
            weights_output -= learning_rate * update_weights_output
            weights_hidden -= learning_rate * update_weights_hidden
            bias_output -= learning_rate * update_bias_output
            bias_hidden -= learning_rate * update_bias_hidden

        for i, line in enumerate(x_test):
            output_hidden = compute_activation(line, weights_hidden, bias_hidden)
            output_final = compute_activation(output_hidden, weights_output, bias_output)
            y = np.zeros(output_final.shape)
            y[y_test[i] - 1] = 1
            test_err.append(loss(output_final, y))

        accuracies.append(compute_accuracy(x_train, y_train, weights_output, weights_hidden, bias_output, bias_hidden))
        test_accuracies.append(compute_accuracy(x_test, y_test, weights_output, weights_hidden, bias_output, bias_hidden))
        errors.append(np.mean(err))
        test_errors.append(np.mean(test_err))

        if k > 0 and abs(accuracies[-1] - accuracies[-2]) < epsilon:
            no_change_iterations += 1
        else:
            no_change_iterations = 0

        if no_change_iterations > stop_num:
            break

    return weights_output, weights_hidden, bias_output, bias_hidden, accuracies, test_accuracies, errors, test_errors


# Function used to compute the accuracy for a given feature set
# using the trained and optimized weights and biases
def compute_accuracy(x_test, y_test, weights_output, weights_hidden, bias_output, bias_hidden):
    error = 0
    for i, line in enumerate(x_test):
        output_hidden = compute_activation(line, weights_hidden, bias_hidden)
        output_final = compute_activation(output_hidden, weights_output, bias_output)
        if np.argmax(output_final) + 1 != y_test[i]:
            error += 1
    return 1 - error / len(x_test)


# Function used to predict the labels of the unknown dataset
# using the trained and optimized weights and biases
def prediction(data, weights_output, weights_hidden, bias_output, bias_hidden):
    predictions = []
    for line in data:
        output_hidden = compute_activation(line, weights_hidden, bias_hidden)
        output_final = compute_activation(output_hidden, weights_output, bias_output)
        predictions.append(np.argmax(output_final) + 1)
    return predictions
