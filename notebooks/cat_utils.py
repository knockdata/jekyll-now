import numpy as np
import h5py


def load_dataset():
    train_dataset = h5py.File('../datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('../datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def load_normalized_dataset():
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()

    m_train = train_set_x_orig.shape[0]                           # number of train samples
    m_test = test_set_x_orig.shape[0]                             # number of test samples
    num_px = train_set_x_orig.shape[1]                            # number pixel on x and y dimension

    train_set_x = train_set_x_orig.reshape(m_train, -1).T / 255   # normalize pixel value to [0, 1]
    test_set_x = test_set_x_orig.reshape(m_test, -1).T / 255

    return train_set_x, train_set_x_orig, train_set_y_orig, test_set_x, test_set_x_orig, test_set_y_orig, classes


def accuracy_2_layers(W1, b1, W2, b2, test_set_x, test_set_y, A, train_set_y):
    train_predict = np.where(A >= 0.5, 1, 0)
    train_accuracy = np.sum(train_predict == train_set_y) / train_set_y.shape[1]

    X = test_set_x
    Z1 = np.dot(W1, X) + b1                     # [n_1, n_x] . [n_x, m]     => [n_1, m]
    A1 = np.tanh(Z1)                            #                              [n_1, m]
    Z2 = np.dot(W2, A1) + b2                    # [n_y, n_1] . [n_1, m]     => [n_y, m]
    A2 = 1.0 / (1.0 + np.exp(-Z2))              #                              [n_y, m]

    test_predict = np.where(A2 >= 0.5, 1, 0)
    test_accuracy = np.sum(test_predict == test_set_y) / test_set_y.shape[1]

    return train_predict, train_accuracy, test_predict, test_accuracy

def accuracy_n_layers(W, b, test_set_x, test_set_y, A, train_set_y):
    train_predict = np.where(A >= 0.5, 1, 0)
    train_accuracy = np.sum(train_predict == train_set_y) / train_set_y.shape[1]
#    print("train accuracy", train_accuracy)

    L = len(W) - 1

    Z  = [l for l in range(L + 1)]
    A  = [l for l in range(L + 1)]
    A[0] = test_set_x

    for l in range(1, L):
        Z[l] = np.dot(W[l], A[l-1]) + b[l]      # [n_1, m] <= [n_1, n_x] . [n_x, m]
        A[l] = np.tanh(Z[l])                    # [n_1, m]

    ZL = np.dot(W[L], A[L-1]) + b[L]            # [n_y, m] <= [n_y, n_1] . [n_1, m]
    AL = 1.0 / (1.0 + np.exp(-ZL))              # [n_y, m]

    test_predict = np.where(AL >= 0.5, 1, 0)
    test_accuracy = np.sum(test_predict == test_set_y) / test_set_y.shape[1]
#    print("test accuracy", test_accuracy)

    return train_predict, train_accuracy, test_predict, test_accuracy
