import numpy as np
import loglinear as ll

STUDENT={'name': 'Alex Kartun',
         'ID': '324429216'}

def classifier_output(x, params):
    # YOUR CODE HERE.
    W = params[0]
    b = params[1]
    scores = np.dot(x, W) + b
    hidden_set = [(U, b_tag) for U, b_tag in zip(params[2::2], params[3::2])]
    for U, b_tag in hidden_set:
        scores = np.dot(np.tanh(scores), U) + b_tag
    return ll.softmax(scores)

def predict(x, params):
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """
    # YOU CODE HERE
    hidden_set = [(U, b_tag) for U, b_tag in zip(params[0::2], params[1::2])]

    calc_list = [x]
    util_list = [1]
    for U, b_tag in hidden_set[:-1]:
        util_list.append(np.dot(calc_list[-1], U) + b_tag)
        calc_list.append(np.tanh(np.dot(calc_list[-1], U) + b_tag))

    y_hat = classifier_output(x, params)
    loss = -np.log(y_hat[y])

    temp = y_hat
    temp[y] -= 1
    grads = []

    for i, (U, b_tag) in enumerate(reversed(hidden_set)):
        grads.append(temp)
        grads.append(np.outer(calc_list[-i - 1], temp))
        temp = np.dot(U, temp) * (1 - pow(np.tanh(util_list[-i - 1]), 2))

    return loss, list(reversed(grads))

def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    params = []
    root_six = np.sqrt(6)
    hidden_dims = [(in_dim, out_dim) for in_dim, out_dim in zip(dims[0:], dims[1:])]
    for in_dim, out_dim in hidden_dims:
        eps = root_six / (np.sqrt(in_dim + out_dim))
        params.append(np.random.uniform(-eps, eps, [in_dim, out_dim]))
        eps = root_six / (np.sqrt(out_dim))
        params.append(np.random.uniform(-eps, eps, out_dim))
    return params


if __name__ == '__main__':
    from grad_check import gradient_check

    def _loss_and_W_grad(W):
        global b, U1, b_tag1, U2, b_tag2
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U1, b_tag1, U2, b_tag2])
        return loss, grads[0]

    def _loss_and_b_grad(b):
        global W, U1, b_tag1, U2, b_tag2
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U1, b_tag1, U2, b_tag2])
        return loss, grads[1]

    def _loss_and_U1_grad(U1):
        global W, b, b_tag1, U2, b_tag2
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U1, b_tag1, U2, b_tag2])
        return loss, grads[2]

    def _loss_and_b_tag1_grad(b_tag1):
        global W, b, U1, U2, b_tag2
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U1, b_tag1, U2, b_tag2])
        return loss, grads[3]

    def _loss_and_U2_grad(U2):
        global W, b, U1, b_tag1, b_tag2
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U1, b_tag1, U2, b_tag2])
        return loss, grads[4]

    def _loss_and_b_tag2_grad(b_tag2):
        global W, b, U1, b_tag1, U2
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U1, b_tag1, U2, b_tag2])
        return loss, grads[5]


    for _ in xrange(10):
        W, b, U1, b_tag1, U2, b_tag2 = create_classifier([3, 5, 2, 4])

        gradient_check(_loss_and_b_tag2_grad, b_tag2)
        gradient_check(_loss_and_U2_grad, U2)
        gradient_check(_loss_and_b_tag1_grad, b_tag1)
        gradient_check(_loss_and_U1_grad, U1)
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)


