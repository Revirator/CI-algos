import numpy as np
import matplotlib.pyplot as plt
import neuron
import optimisation


# Function used to load and parse the features data
def load_features(filename):
    with open(filename) as f:
        lines = [line.rstrip("\n") for line in f]

    data = []
    for line in lines:
        features = line.split(",")
        vector = [float(feat) for feat in features]
        data.append(np.array(vector))
    return np.array(data)


# Function used to load and parse the targets data
def load_targets(filename):
    with open(filename) as f:
        lines = [line.rstrip("\n") for line in f]
    line_as_int = lambda l: int(l.rstrip('\n'))
    return np.array(list(map(line_as_int, lines)))


# Function used for a question 10 to cycle through multiple weights and plot the output
def test_different_weights():
    acc_list = []

    for i in range(10):
        weights_output, weights_hidden, bias_output, bias_hidden, accuracies, test_accuracies, errors, test_errors = \
            neuron.train(x_train, y_train, size_input, size_hidden, size_output, n_epochs, learning_rate, x_validation,
                         y_validation, stopping_number_of_iterations, epsilon)

        acc = neuron.compute_accuracy(x_test, y_test, weights_output, weights_hidden, bias_output, bias_hidden)
        print(acc)
        acc_list.append(acc)

    print(len(acc_list))
    plt.plot(range(1, 11), acc_list)
    plt.xticks(range(1, 11))
    plt.title("Initialization over accuracy (random weights)")
    plt.xlabel("Initialization")
    plt.ylabel("Accuracy")
    plt.show()


### START INPUT PARAMETERS


# Input parameters used to tweak the data (hyper-parameters)
data = load_features("data/features.txt")
targets = load_targets("data/targets.txt")
unknown_data = load_features("data/unknown.txt")
size_input = 10
size_hidden = 17
size_output = 7
n_epochs = 200
learning_rate = 0.0125
stopping_number_of_iterations = 10
epsilon = 1e-3

# Option to show the plots after single training
show_plots = True

### END INPUT PARAMETERS

# 0.8-0.1-0.1 split
x_train, x_test, y_train, y_test = optimisation.split_train_validate(data, targets, test_size=0.2)
x_validation, x_test, y_validation, y_test = optimisation.split_train_validate(x_test, y_test, test_size=0.5)

## START CROSS VALIDATION

# x_folds, y_folds = optimisation.fold_split(x_train, y_train, 10)
# weights_output, weights_hidden, bias_output, bias_hidden, out = optimisation.cross_validate(x_folds, y_folds, size_hidden, size_input, size_output, n_epochs, learning_rate, stopping_number_of_iterations, epsilon)
# print("Mean accuracy for " + str(size_hidden) + " is " + str(out))

## END CROSS VALIDATION

## START PARAMETER OPTIMIZATION

# x_folds, y_folds = optimisation.fold_split(x_train, y_train, 10)
# accuracies = []
# for i in range(11):
#     weights_output, weights_hidden, bias_output, bias_hidden, out = optimisation.cross_validate(x_folds, y_folds, size_hidden, size_input, size_output, n_epochs, learning_rate, stopping_number_of_iterations, epsilon)
#     accuracies.append(out)
#     print("Mean accuracy for " + str(size_hidden + i) + " is " + str(out))
#
# plt.plot([*range(10, len(accuracies) + 10)], accuracies, label="mean test accuracy")
# plt.xlabel("Number of hidden neurons")
# plt.ylabel("Mean accuracy")
# plt.legend()
# plt.show()

### END PARAMETER OPTIMIZATION

### START SINGLE TRAINING

weights_output, weights_hidden, bias_output, bias_hidden, accuracies, validation_accuracies, errors, test_errors = neuron.\
    train(x_train, y_train, size_input, size_hidden, size_output, n_epochs, learning_rate, x_validation, y_validation, stopping_number_of_iterations, epsilon)

accuracy = neuron.compute_accuracy(x_test, y_test, weights_output, weights_hidden, bias_output, bias_hidden)
print(accuracy)

if show_plots:
    plt.plot([*range(0, len(accuracies))], accuracies, label="train accuracy")
    plt.plot([*range(0, len(validation_accuracies))], validation_accuracies, label="validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    plt.plot([*range(0, len(errors))], errors, label="train error")
    plt.plot([*range(0, len(test_errors))], test_errors, label="validation error")
    plt.xlabel("Epochs")
    plt.ylabel("Error (MSE)")
    plt.legend()
    plt.show()

# Unknown Data Prediction
predictions = neuron.prediction(unknown_data, weights_output, weights_hidden, bias_output, bias_hidden)
with open("./Group_75_classes.txt", "w") as f:
    f.write(",".join(map(str, predictions)))

# Confusion Matrix
if show_plots:
    optimisation.plot_confusion_matrix(size_output, x_test, y_test, weights_output, weights_hidden, bias_output, bias_hidden)

### END SINGLE TRAINING

# Test Random Weights
# test_different_weights()