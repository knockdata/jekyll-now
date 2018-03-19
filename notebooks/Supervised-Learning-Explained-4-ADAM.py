
# In[1]:
# !pwd


import numpy as np
import time
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
import cat_utils

train_set_x, train_set_x_orig, train_set_y, test_set_x, test_set_x_orig, test_set_y, classes = cat_utils.load_normalized_dataset()

print(train_set_x.shape)

# ## Goes to More Layers
#
# In the code written before, it is fixed to 2 layers.
#
# With a little bit extension, it will be able to extend to multiple layers.

# In[23]:


def initilize_parameters_n_layers(X, Y, hidden_layer_dims, activations):
    """
    layer_dims: [n_x, n_1, n_2, ..., n_y]
    activations: ['not_used', 'relu', 'relu', ..., 'sigmoid']
    """

    n_x, m = X.shape
    n_y, _ = Y.shape

    # And input layer, and output layer
    layer_dims = [n_x] + hidden_layer_dims + [n_y]
    L = len(layer_dims) - 1

    W = [l for l in range(L + 1)]  # W[0] - W[L], W[0] is not used
    b = [l for l in range(L + 1)]  # b[0] - b[L], b[0] is not used

    # Initialize parameters
    for l in range(1, L + 1):  # 1 - L
        if activations[l] == "relu":
            norm = np.sqrt(2.0 / layer_dims[l-1]) # He Initialization, He et al., 2015
        elif activations[l] == "tanh":
            norm = 0.01
        elif activations[l] == "sigmoid":
            norm = 0.01
        else:
            norm = 1
        if 'debug' in globals() and debug:
            print("layer", l, "[", layer_dims[l], layer_dims[l-1], "]", activations[l], norm)

        W[l] = np.random.randn(layer_dims[l], layer_dims[l-1]) * norm # (n[l], n[l-1])
        b[l] = np.zeros((layer_dims[l], 1))


    # print one param for debug
    if 'debug' in globals() and debug:
        print("init weights", W[1][0][0], W[2][0][0])

    return W, b


# In[32]:


debug = False


# In[33]:


def forward_propagate_n_layers(X, W, b, activations, iteration=-1):
    """
    X: numpy                  - shape [n_x, m]
    W: numpy array            - [0, W[1], W[2], ..., W[L]]
    b: numpy array            - [0, b[1], b[2], ..., b[L]]
    activations: string array - ['not_used', 'relu', 'relu', ..., 'sigmoid']
    iteration:     integer    - this is used for debug infor
    returns: (A, Z)
        A: numpy array        - [A[0]=X,            A[1], A[2] ..., A[L]]
        Z: numpy array        - [Z[0]=0 (not used), Z[1], Z[2], ..., Z[L]]
    """
    L = len(W) - 1

    Z  = [l for l in range(L+1)]
    A  = [l for l in range(L+1)]

    A[0] = X

    for l in range(1, L+1):
        Z[l] = np.dot(W[l], A[l - 1]) + b[l] # (n[l], m) <= (n[l], n[l-1]) . (n[l-1], m) + (n[l], 1)

        if activations[l] == "relu":
            A[l] = np.max(0, Z[l])
        elif activations[l] == "tanh":
            A[l] = np.tanh(Z[l])
        elif activations[l] == "sigmoid":
            A[l] = 1.0 / (1.0 + np.exp(-Z[l]))
        else:
            raise Exception("activation " + activations[l] + "not supported")

        if iteration == 0 and 'debug' in globals() and debug:
            shape_info = "Z[l] = np.dot(W[l], A[l - 1]) + b[l] {0} <= {1} . {2} + {3}".format(
                Z[l].shape, W[l].shape, A[l-1].shape, b[l].shape)
            print(l, activations[l], shape_info)

    return Z, A


# In[54]:


def backward_propagate_n_layers(X, Y, Z, A, W, b, activations, iteration=-1, lambd=0):
    """
    X: numpy                  - shape [n_x, m]
    Y: numpy                  - shape [n_y, 1]
    Z: numpy array            - [0, Z[1], Z[2], ..., Z[L]]
    A: numpy array            - [0, A[1], A[2], ..., A[L]]
    W: numpy array            - [0, W[1], W[2], ..., W[L]]
    b: numpy array            - [0, b[1], b[2], ..., b[L]]
    activations: string array - ['not_used', 'relu', 'relu', ..., 'sigmoid']
    iteration:     integer    - this is used for debug infor
    returns: (dW, db)
        dW: numpy array        - [dW[0]=0, dW[1], dW[2], ..., dW[L]]
        db: numpy array        - [db[0]=0, db[1], db[2], ..., db[L]]
    """
    n_x, m = X.shape
    L = len(W) - 1

    dA = [l for l in range(L + 1)] # index 0 not used
    dZ = [l for l in range(L + 1)]
    dW = [l for l in range(L + 1)]
    db = [l for l in range(L + 1)]

    #
    # Backward propagation for the last layer
    if activations[L] == "sigmoid":
        dA[L] = -(np.divide(Y, A[L]) - np.divide(1 - Y, 1 - A[L]))
        # The way to calculate dZL is different than other layers, due to different activation function
        dZ[L] = A[L] - Y
    else:
        raise Exception("activation " + activations[L] + "not supported")

    dW[L] = np.dot(dZ[L], A[L-1].T) / m  + lambd * W[L] / m
    db[L] = np.sum(dZ[L], axis=1, keepdims=True) / m

    # Backward propagation for other layers
    for l in reversed(range(1, L)):
        if iteration == 0 and 'debug' in globals() and debug:
            print(l, "W[l].shape", W[l].shape, "dZ[l+1].shape", dZ[l+1].shape)

        dAl = np.dot(W[l+1].T, dZ[l+1])
        if activations[l] == "tanh":
            dgZl = 1 - np.power(A[l], 2)
            dZ[l] = np.multiply(dAl, dgZl)
        elif activations[l] == "relu":
            dZ[l] = np.array(dAl, copy=True) # just converting dz to a correct object.
            # When z <= 0, you should set dz to 0 as well.
            dZ[l][Z[l] <= 0] = 0
        else:
            raise Exception("activation " + activations[L] + "not supported")

        dW[l] = np.dot(dZ[l], A[l-1].T) / m + lambd * W[l] / m
        db[l] = np.sum(dZ[l], axis=1, keepdims=True) / m

    return dW, db


# In[55]:


def calculate_cost(AL, Y, W, lambd=0):
    """
    AL: numpy                 - shape [n_y, m]
    Y: numpy                  - shape [n_y, 1]
    W: numpy array            - [0, W[1], W[2], ..., W[L]]
    returns: cost
    """

    n_y, m = Y.shape

    # calculate cost

    cross_entropy_cost = -np.sum((np.dot(Y, np.log(AL.T)) + np.dot(1-Y, np.log(1-AL.T)))) / m

    regulation = 0
    if lambd > 0:
        for l in range(1, len(W)):
            regulation += np.sum(W[l])

    return np.squeeze(cross_entropy_cost) + regulation / m


# In[68]:


def neural_network_n_layers(X, Y, test_set_x, test_set_y, hidden_layer_dims, activations,
                            num_iterations=10, learning_rate=0.01, stop_cost=0., lambd=0,
                            msg_interval=1, print_interval=100):
    """
    X: numpy                  - shape [n_x, m]
    Y: numpy                  - shape [n_y, m]
    test_set_x: numpy         - shape [n_x, m_test]
    test_set_y: numpy         - shape [n_y, m_test]
    Z: numpy array            - [0, Z[1], Z[2], ..., Z[L]]
    A: numpy array            - [0, A[1], A[2], ..., A[L]]
    W: numpy array            - [0, W[1], W[2], ..., W[L]]
    b: numpy array            - [0, b[1], b[2], ..., b[L]]
    hidden_layer_dims: integer array - [n_1, n_2, ..., n_l]
    activations: string array - ['not_used', 'relu', 'relu', ..., 'sigmoid']
    num_iterations:     integer    - this is used for debug infor
    learning_rate:     integer    - this is used for debug infor
    stop_cost:     integer    - this is used for debug infor
    lambd:     integer    - this is used for debug infor
    msg_interval:     integer    - this is used for debug infor
    print_interval:     integer    - this is used for debug infor
    returns: (dW, db)
        dW: numpy array        - [dW[0]=0, dW[1], dW[2], ..., dW[L]]
        db: numpy array        - [db[0]=0, db[1], db[2], ..., db[L]]
    """
#     n_x, m = X.shape
#     n_y, _ = Y.shape

    # And input layer, and output layer
    L = len(hidden_layer_dims) + 1

    activations.insert(0, "not_used")

    W, b = initilize_parameters_n_layers(X, Y, hidden_layer_dims, activations)

    start = time.time()

    msgs = []
    for i in range(num_iterations):
        Z, A = forward_propagate_n_layers(X, W, b, activations, i)
        dW, db = backward_propagate_n_layers(X, Y, Z, A, W, b, activations, i, lambd)

        # update parameters
        for l in range(1, L + 1):
            W[l] = W[l] - learning_rate * dW[l]
            b[l] = b[l] - learning_rate * db[l]

        if i == 0 and 'debug' in globals() and debug:
            print("dW[1][0][0], dW[2][0][0]", dW[1][0][0], dW[2][0][0])
            print("db[1][0][0], db[2][0][0]", db[1][0][0], db[2][0][0])
            print(" W[1][0][0],  W[2][0][0]",  W[1][0][0],  W[2][0][0])

        cost = calculate_cost(A[L], Y, W, lambd)

        if cost < stop_cost:
            break

        if i % msg_interval == 0:
            train_predict, train_accuracy, test_predict, test_accuracy = cat_utils.accuracy_n_layers(
                W, b, test_set_x, test_set_y, A[-1], train_set_y)
            msg = {
                "iterations": i,
                "dimensions": hidden_layer_dims,
                "learning_rate": learning_rate,
                "cost": cost,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "training_time": time.time() - start}

            msgs.append(msg)

            if i % print_interval == 0:
                print(msg)
                if 'debug' in globals() and debug:
                    print(i, " derivitive", dW[1][0][0], dW[2][0][0], db[1][0][0], db[2][0][0])
                    print(i, " weights", W[1][0][0], W[2][0][0])

    print(msgs[-1])

    return W, b, A, msgs, train_predict, train_accuracy, test_predict, test_accuracy


# ## Sanity Check for the Multi Layer Code
#
# Run training with same configuration to see if it has same result

# In[57]:




def train_n_layers(params):
    dims = params["dims"]
    activations = params["activations"]
    num_iterations = params["num_iterations"]
    learning_rate = params["learning_rate"]
    stop_cost = params["stop_cost"]

    np.random.seed(1)     # set seed, so that the result is comparable
    W, b, A, msgs, train_predict, train_accuracy, test_predict, test_accuracy = neural_network_n_layers(
        train_set_x, train_set_y, test_set_x, test_set_y,
        hidden_layer_dims=dims,
        activations=activations,
        num_iterations=num_iterations, learning_rate=learning_rate, stop_cost=stop_cost, lambd=0.1)

    return W, b, msgs, train_predict, train_accuracy, test_predict, test_accuracy


# In[66]:


param = {"dims": [100], "activations": ["tanh", "sigmoid"], "num_iterations": 801, "learning_rate": 0.005, "stop_cost": 0.05}
W, b, msgs, train_predict, train_accuracy, test_predict, test_accuracy = train_n_layers(param)


costs = [msg["cost"] for msg in msgs]
plt.ioff()
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(param["learning_rate"]))
plt.show()

# In[67]:




# In[79]:



md_rows = ["|dims|learning_rate|stop_cost|iterations|training_time|train_accuracy|test_accuracy|",
            "|:--|:------------|:--------|:---------|:------------|:-------------|:------------|"]
debug=False
param = {"dims": [100], "activations": ["tanh", "sigmoid"], "num_iterations": 801, "learning_rate": 0.03, "stop_cost": 0.05}

W, b, msgs, train_predict, train_accuracy, test_predict, test_accuracy = train_n_layers(param)
md_row = "|" + "|".join([str(m) for m in msgs[-1]]) + "|"
md_rows.append(md_row)
display(Markdown("\n".join(md_rows)))


