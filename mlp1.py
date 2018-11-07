import numpy as np
import loglinear as ll

STUDENT={'name': 'Alex Kartun',
         'ID': '324429216'}


def classifier_output(x, params):
    # YOUR CODE HERE.
    W, b, U, b_tag = params
    scores = np.dot(np.tanh(np.dot(x, W) + b), U) + b_tag
    return ll.softmax(scores)


def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """
    # YOU CODE HERE
    W, b, U, b_tag = params

    y_hat = classifier_output(x, params)
    loss = -np.log(y_hat[y])

    y_hat[y] -= 1
    gb_tag = y_hat
    gU = np.outer(np.tanh(np.dot(x, W) + b), gb_tag)

    gb = np.dot(U, y_hat) * (1 - pow(np.tanh(np.dot(x, W) + b), 2))
    gW = np.outer(x, gb)

    return loss, [gW, gb, gU, gb_tag]


def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    root_six = np.sqrt(6)
    eps_W = root_six / (np.sqrt(in_dim + hid_dim))
    eps_b = root_six / (np.sqrt(hid_dim))
    eps_U = root_six / (np.sqrt(hid_dim + out_dim))
    eps_b_tag = root_six / (np.sqrt(out_dim))
    W = np.random.uniform(-eps_W, eps_W, [in_dim, hid_dim])
    b = np.random.uniform(-eps_b, eps_b, hid_dim)
    U = np.random.uniform(-eps_U, eps_U, [hid_dim, out_dim])
    b_tag = np.random.uniform(-eps_b_tag, eps_b_tag, out_dim)
    params = [W, b, U, b_tag]
    return params


if __name__ == '__main__':
    from grad_check import gradient_check

    def _loss_and_W_grad(W):
        global b, U, b_tag
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag])
        return loss, grads[0]

    def _loss_and_b_grad(b):
        global W, U, b_tag
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag])
        return loss, grads[1]

    def _loss_and_U_grad(U):
        global W,b, b_tag
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag])
        return loss, grads[2]

    def _loss_and_b_tag_grad(b_tag):
        global W, b, U
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag])
        return loss, grads[3]


    for _ in xrange(10):
        W, b, U, b_tag = create_classifier(3, 5, 4)

        gradient_check(_loss_and_b_tag_grad, b_tag)
        gradient_check(_loss_and_U_grad, U)
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)
