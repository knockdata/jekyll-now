
# In[1]:
# !pwd

import numpy as np
import time
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
import cat_utils

def forward_progagate_2_layers(X, Y, W1, b1, W2, b2):
    """
    X: [n_x, m]      train data set
    Y: [n_y,  m]     label data
    W1:[n_1, n_x]    weight for the first layer
    b1:[n_1, 1]      bias for the first layer
    W2:[n_y, n_1]    weight for the second layer
    b1:[n_y, 1]      bias for the first layer
    returns:         A1, A2
    """

    Z1 = np.dot(W1, X) + b1                     # [n_1, m]   <= [n_1, n_x] . [n_x, m]
    A1 = np.tanh(Z1)                            # [n_1, m]
    Z2 = np.dot(W2, A1) + b2                    # [n_y, m]   <= [n_y, n_1] . [n_1, m]
    A2 = 1.0 / (1.0 + np.exp(-Z2))              # [n_y, m]

    return A1, A2


def backward_propagate_2_layers(X, Y, A1, A2, W1, b1, W2, b2):
    """
    X:  [n_x, m]      train data set
    Y:  [n_y, m]      label data
    A1: [n_1, m]      first layer output
    A2: [1,   m]      second layer output
    W1: [n_1, n_x]    weight for the first layer
    b1: [n_1, 1]      bias for the first layer
    W2: [n_y, n_1]    weight for the second layer
    b1: [n_y, 1]      bias for the first layer
    returns:          dW1, db1, dW2, db2
    """
    n_x, m = X.shape

    dZ2 = A2 - Y                                   # [n_y, m]
    dW2 = np.dot(dZ2, A1.T) / m                    # [n_y, n_1] <= [n_y, m] .   [n_1, m].T
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m   # [n_y, 1]   <= [n_y, m]

    dgZ1 = 1 - np.power(A1, 2)                     # [n_1, m]
    dZ1  = np.multiply(np.dot(W2.T, dZ2), dgZ1)    # [n_1, m]   <= [n_y, n_1].T . [n_y, m]
    dW1  = np.dot(dZ1, X.T) / m                    # [n_1, n_x] <= [n_1, m]     . [n_x, m].T
    db1  = np.sum(dZ1, axis=1, keepdims=True) / m  # [n1, 1]    <= [n_1, m]

    return dW1, db1, dW2, db2


def neural_network_2_layers(X, Y, n_1, num_iterations=10, learning_rate=0.01,
                            early_stop_cost=0., msg_interval=1, print_interval=100):
    """
    X: [n_x, m]      train data set
    Y: [n_y, m]      n_y=1 in this case
    n_1:             first hidden layer dimension
    num_iterations:  number iterations
    learning_rate:   learning rate alpha
    early_stop_cost: early stop cost, if the cost small than this number, the train will stop
    returns:         W1, b1, W2, b2, A2, msgs, costs (A2, is the output)
    """
    n_x, m = X.shape
    n_y = 1

    W1 = np.random.randn(n_1, n_x) * 0.01
    b1 = np.zeros([n_1, 1])
    W2 = np.random.randn(n_y, n_1) * 0.01
    b2 = np.zeros([n_y, 1])
    print("init weights", W1[0][0], W2[0][0])

    msgs = []

    start = time.time()

    for i in range(num_iterations):
        A1, A2 = forward_progagate_2_layers(X, Y, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward_propagate_2_layers(X, Y, A1, A2, W1, b1, W2, b2)

        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2

        cost = np.sum(-(np.dot(Y, np.log(A2.T)) + np.dot(1-Y, np.log(1-A2.T)))) / m
        cost = np.squeeze(cost)

        if cost < early_stop_cost:
            break

        if i % msg_interval == 0:
            train_predict, train_accuracy, test_predict, test_accuracy = cat_utils.accuracy_2_layers(
                W1, b1, W2, b2, test_set_x, test_set_y, A2, train_set_y)
            msg = {
                "iterations": i,
                "dimensions": n_1,
                "learning_rate": learning_rate,
                "cost": cost,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "training_time": time.time() - start}

            msgs.append(msg)

            if i % print_interval == 0:
                print(i, msg)
                if 'debug' in globals() and debug:
                    print(i, " derivitive", dW1[0][0], dW2[0][0], db1[0][0], db2[0][0])
                    print(i, " weights", W1[0][0], W2[0][0])


    print(i, msgs[-1])

    return W1, b1, W2, b2, A2, msgs, train_predict, train_accuracy, test_predict, test_accuracy


train_set_x, train_set_x_orig, train_set_y, test_set_x, test_set_x_orig, test_set_y, classes = cat_utils.load_normalized_dataset()

print(train_set_x.shape)


def train(params):
    n_1 = params["n_1"]
    num_iterations = params["num_iterations"]
    learning_rate = params["learning_rate"]
    early_stop_cost = params["early_stop_cost"]

    np.random.seed(1)     # set seed, so that the result is comparable

    W1, b1, W2, b2, A2, msgs, train_predict, train_accuracy, test_predict, test_accuracy = neural_network_2_layers(
        train_set_x, train_set_y, n_1, num_iterations, learning_rate, early_stop_cost)

    return W1, b1, W2, b2, msgs, train_predict, train_accuracy, test_predict, test_accuracy


markdown_rows = ["|iterations|n_1|learning_rate|stop_cost|train_accuracy|test_accuracy|training_time|",
            "|:--|:------------|:--------|:---------|:------------|:-------------|:------------|"]
debug=False
param = {"n_1": 100, "num_iterations": 2001, "learning_rate": 0.005, "early_stop_cost": 0.05}
W1, b1, W2, b2, msgs, train_predict, train_accuracy, test_predict, test_accuracy = train(param)

costs = [msg["cost"] for msg in msgs]

plt.ioff()
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(param["learning_rate"]))
plt.show()

# markdown_row = "|" + "|".join([str(m) for m in msgs[-1]]) + "|" + str(training_time) + "|"
#
# markdown_rows.append(markdown_row)
# display(Markdown("\n".join(markdown_rows)))

print(msgs[-1])


# ## Accuracy for Train and Test Data Set
#
# Accuracy for train set is used to measure how good the model is, for the data it has seen.
#
# Accuracy for train set is used to measure how good the model is, for the data it has **NOT** seen.
# In other words, it is used to measure how general the model is.
#
# With the model we get, we can write followning code to get the accuracy

# In[66]:





# ## Results with Different Hyper Parameters
#
# Choose different hyper parameters will get different result.
#
# Here are some results from variance configurations. The best test accuracy got is 80%, better then 70% with logistics regression. From the test, we also see that the accuracy is not always higher when use more neurals.

# |n_1|learning rate | stop cost | # iterations| training time | train set accuracy | test accuracy |
# |:----|:----|:----------|
# | 100 |0.0003 | 0.1  | 2000  |  8 |1 | 0.78 |
# | 100 |0.0003 | 0.002  | 2000  |  34 |1 | 0.76 |
# | 500 |0.0003 | 0.1      | 471  |  60 |0.9856 | 0.78 |
# | 500 |0.0003 | 0.00187  | 2000  |  260 |1 | 0.76 |
# | 1000 |0.0003 | 0.1  | 480  |  128 |0.98 | 0.8 |
# | 1000 |0.0003 | 0.0017  | 2000  |  530 |1 | 0.74 |
# | 2000 |0.0003 | 0.1  | 2000  |  530 |1 | 0.74 |

# ## Show Wrong Classficiations

# In[29]:




# get_ipython().run_line_magic('matplotlib', 'inline')
wrong_index = np.argmax(train_predict != train_set_y)
print("wrong predict on train sample ", wrong_index, " to ", train_predict[:, wrong_index],
      "which should be", train_set_y[:, wrong_index])
plt.imshow(train_set_x_orig[wrong_index])


# In[30]:


import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
wrong_index = np.argmax(test_predict != test_set_y)
print("wrong predict on test sample ", wrong_index, " to ", test_predict[:, wrong_index],
      "which should be", test_set_y[:, wrong_index])
plt.imshow(test_set_x_orig[wrong_index])
