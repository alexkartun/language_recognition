import mlp1
import utils
import xor_data
import numpy as np
import random

STUDENT = {'name': 'Alex Kartun',
           'ID': '324429216'}


def feats_to_vec(features):
    result = np.zeros(len(utils.F2I))
    for feature in features:
        if feature in utils.F2I:
            index = utils.F2I[feature]
            result[index] += 1
    return result


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:

        x = feats_to_vec(features)
        y = utils.L2I[label]

        y_hat = mlp1.predict(x, params)
        if y == y_hat:
            good += 1
        else:
            bad += 1
        pass
    return good / (good + bad)


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """

    for I in xrange(num_iterations):
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features)  # convert features to a vector.
            y = utils.L2I[label]
            loss, grads = mlp1.loss_and_gradients(x, y, params)
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.
            params = np.subtract(params, np.multiply(learning_rate, grads))

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print I, train_loss, train_accuracy, dev_accuracy
    return params

def create_test_pred_file(test_data, params):
    """
    creates a 'test.pred' file
    :param test_data: test data to be predicted
    :param params: trained params
    :return:
    """
    f = open("test.pred", 'w')
    for label, features in test_data:
        x = feats_to_vec(features)
        y_hat = mlp1.predict(x, params)
        for l, i in utils.L2I.items():
            if y_hat == i:
                label = l
                break
        f.write(label + "\n")
    f.close()


LR = 0.001
NUM_ITERATIONS = 15
HIDDEN_LAYER_SIZE = 10

if __name__ == '__main__':

    params = mlp1.create_classifier(len(utils.F2I), HIDDEN_LAYER_SIZE, len(utils.L2I))
    trained_params = train_classifier(utils.TRAIN, utils.DEV, NUM_ITERATIONS, LR, params)

    TEST = [(l, utils.text_to_bigrams(t)) for l, t in utils.read_data("test")]
    create_test_pred_file(TEST, trained_params)
