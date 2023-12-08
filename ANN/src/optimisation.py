import numpy as np
import matplotlib.pyplot as plt
import neuron


# Code used for creating multiple folds of data used for cross-validation
def fold_split(data, targets, folds):
    sz = int(len(data) / folds)
    d = list(data)
    t = list(targets)
    x_folds = []
    y_folds = []

    for i in range(folds):
        x_currFold = []
        y_currFold = []
        for j in range(sz):
            idx = np.random.randint(len(d))
            x_currFold.append(d.pop(idx))
            y_currFold.append(t.pop(idx))
        x_folds.append(x_currFold)
        y_folds.append(y_currFold)

    return x_folds, y_folds


# Split data into to sets of the given percentage size
def split_train_validate(data, targets, test_size):
    assert test_size < 1
    sz = int(len(data) * test_size)
    d = list(data)
    t = list(targets)
    x_test = []
    y_test = []

    for i in range(sz):
        idx = np.random.randint(len(d))
        x_test.append(d.pop(idx))
        y_test.append(t.pop(idx))

    return d, x_test, t, y_test


# Function used to perform cross-validation on the given data input
def cross_validate(x_folds, y_folds, hidden_neurons, size_input, size_output, n_epochs, learning_rate, stopping_number_of_iterations, epsilon):
    acc = 0
    for i in range(len(x_folds)):
        x_test = x_folds[i]
        y_test = y_folds[i]
        x_train = []
        y_train = []
        # Use a single fold for validation and others for training
        for j in range(len(x_folds)):
            if j != i:
                x_train = x_train + x_folds[j]
                y_train = y_train + y_folds[j]

        weights_output, weights_hidden, bias_output, bias_hidden, ac, tac, err, terr = \
            neuron.train(x_train, y_train, size_input, hidden_neurons, size_output, n_epochs,
                         learning_rate, x_test, y_test, stopping_number_of_iterations, epsilon)
        accuracy = neuron.compute_accuracy(x_test, y_test, weights_output, weights_hidden, bias_output, bias_hidden)
        print(accuracy)
        acc += accuracy
    return weights_output, weights_hidden, bias_output, bias_hidden, acc / len(x_folds)


# Function used to plot the confusion matrix from the given data
def plot_confusion_matrix(size_output, x_test, y_test, weights_output, weights_hidden, bias_output, bias_hidden):
    actual = predicted = size_output
    # Make all the predictions
    conf_matrix = [[0 for x in range(actual)] for y in range(predicted)]
    y_test_predictions = neuron.prediction(x_test, weights_output, weights_hidden, bias_output, bias_hidden)
    for actual_label, predicted_label in zip(y_test, y_test_predictions):
        conf_matrix[actual_label - 1][predicted_label - 1] += 1

    # Round all the data to show as percentages
    for i, line in enumerate(conf_matrix):
        conf_matrix[i] = np.round(line / np.sum(line), decimals=2)
        print(line)

    # plot confusion matrix
    fig, ax = plt.subplots()
    im = ax.imshow(conf_matrix)

    ax.set_xticks(range(0, 7))
    ax.set_yticks(range(0, 7))
    ax.set_xticklabels(list(range(1, 8)))
    ax.set_yticklabels(list(range(1, 8)))

    # Add text for each tile
    for i in range(7):
        for j in range(7):
            text = ax.text(j, i, conf_matrix[i][j],
                           ha="center", va="center", color="tomato")
    plt.show()
