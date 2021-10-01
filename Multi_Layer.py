
import matplotlib
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt


class LinearLayer:

    #ll functions to be executed by a linear layerin a computational graph

    def __init__(self, input_shape, n_out, ini_type="plain"):
        self.m = input_shape[1]  # number of examples in training data
        # `params` store weights and bias in a python dictionary
        self.params = self.initialize_parameters(input_shape[0], n_out,
                                                 ini_type='xavier')  # initialize weights and bias
        self.Z = np.zeros((self.params['W'].shape[0], input_shape[1]))  # create space for resultant Z output

    def initialize_parameters(self, n_in, n_out, ini_type):

        params = dict()  # initialize empty dictionary of neural net parameters W and b

        if ini_type == 'plain':
            params['W'] = np.random.randn(n_out, n_in) * 0.01  # set weights 'W' to small random gaussian
        elif ini_type == 'xavier':
            params['W'] = np.random.randn(n_out, n_in) / (np.sqrt(n_in))  # set variance of W to 1/n
        elif ini_type == 'he':
            # Good when ReLU used in hidden layers
            # Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
            # Kaiming He et al. (https://arxiv.org/abs/1502.01852)
            # http: // cs231n.github.io / neural - networks - 2 /  # init
            params['W'] = np.random.randn(n_out, n_in) * np.sqrt(2 / n_in)  # set variance of W to 2/n

        params['b'] = np.zeros((n_out, 1))  # set bias 'b' to zeros

        return params

    def forward(self, A_prev):

        self.A_prev = A_prev  # store the Activations/Training Data coming in
        self.Z = np.dot(self.params['W'], self.A_prev) + self.params['b']  # compute the linear function

    def backward(self, upstream_grad):

        # derivative of Cost w.r.t W
        self.dW = np.dot(upstream_grad, self.A_prev.T)

        # derivative of Cost w.r.t b, sum across rows
        self.db = np.sum(upstream_grad, axis=1, keepdims=True)

        # derivative of Cost w.r.t A_prev
        self.dA_prev = np.dot(self.params['W'].T, upstream_grad)

    def update_params(self, learning_rate=0.1):
        self.params['W'] = self.params['W'] - learning_rate * self.dW  # update weights
        self.params['b'] = self.params['b'] - learning_rate * self.db  # update bias(es)


class ActivationLayer:

    #sigmoid layer
    def __init__(self, shape):
        self.A = np.zeros(shape)  # create space for the resultant activations

    def forward(self, Z):
        self.A = 1 / (1 + np.exp(-Z))  # compute activations

    def backward(self, upstream_grad):
        # couple upstream gradient with local gradient, the result will be sent back to the Linear layer
        self.dZ = upstream_grad * self.A * (1 - self.A)


class Utils:

    def __init__(self):
        self.class_name = "Utils"

    def compute_cost(self, Y, Y_hat):
        m = Y.shape[1]

        cost = (1 / (2 * m)) * np.sum(np.square(Y - Y_hat))
        cost = np.squeeze(cost)  # remove extraneous dimensions to give just a scalar

        dY_hat = -1 / m * (Y - Y_hat)  # derivative of the squared error cost function

        return cost, dY_hat

    def predict(self, X, Y, Zs, As):

        m = X.shape[1]
        n = len(Zs)  # number of layers in the neural network
        p = np.zeros((1, m))

        # Forward propagation
        Zs[0].forward(X)
        As[0].forward(Zs[0].Z)
        for i in range(1, n):
            Zs[i].forward(As[i - 1].A)
            As[i].forward(Zs[i].Z)
        probas = As[n - 1].A

        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0, i] > 0.5:  # 0.5 is threshold
                p[0, i] = 1
            else:
                p[0, i] = 0

        # print results
        # print ("predictions: " + str(p))
        # print ("true labels: " + str(y))
        accuracy = np.sum((p == Y) / m)

        return p, probas, accuracy * 100

    def plot_learning_curve(self, costs, learning_rate, total_epochs, save=False):

        # plot the cost
        plt.figure()

        steps = int(total_epochs / len(costs))  # the steps at with costs were recorded
        plt.ylabel('Cost')
        plt.xlabel('Iterations ')
        plt.title("Learning rate =" + str(learning_rate))
        plt.plot(np.squeeze(costs))
        locs, labels = plt.xticks()
        plt.xticks(locs[1:-1], tuple(np.array(locs[1:-1], dtype='int') * steps))  # change x labels of the plot
        plt.xticks()
        if save:
            plt.savefig('Cost_Curve.png', bbox_inches='tight')
        plt.show()

    def get_data(self):
        # Load the dataset
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        # total Number of pixels
        num_pixels = X_train.shape[1] * X_train.shape[2]

        # reshape input features
        X_train = X_train.reshape(X_train.shape[0], num_pixels).T
        X_test = X_test.reshape(X_test.shape[0], num_pixels).T

        # reshaping labels ( classes)
        y_train = y_train.reshape(y_train.shape[0], 1)
        y_test = y_test.reshape(y_test.shape[0], 1)

        # convert to floating points ( for training)
        X_train = X_train.astype("float32")
        X_test = X_test.astype("float32")

        # Convert to floating points ( for testing)
        y_train = y_train.astype("float32")
        y_test = y_test.astype("float32")

        # Normalize features
        X_train = X_train / 255
        X_test = X_test / 255

        # We want to have a binary classification: digit 0 is classified 1 and
        # all the other digits are classified 0

        # For seek of binary classification
        y_new = np.zeros(y_train.shape)
        y_new[np.where(y_train == 0.0)[0]] = 1
        y_train = y_new

        # For seek of binary classification
        y_new = np.zeros(y_test.shape)
        y_new[np.where(y_test == 0.0)[0]] = 1
        y_test = y_new

        y_train = y_train.T
        y_test = y_test.T

        #  Number of training examples
        m = X_train.shape[1]  # number of examples

        # Now, we shuffle the training set
        np.random.seed(138)
        shuffle_index = np.random.permutation(m)
        X_train, y_train = X_train[:, shuffle_index], y_train[:, shuffle_index]

        return X_train, y_train, X_test, y_test

    def create_network(self):

        # Our network architecture has the shape:
        #                   (input)--> [Linear->Sigmoid] -> [Linear->Sigmoid] -->(output)

        # ------ LAYER-1 ----- define hidden layer that takes in training data
        Layer_1_z1 = LinearLayer(input_shape=X_train.shape, n_out=64, ini_type='xavier')
        Layer_1_a = ActivationLayer(Layer_1_z1.Z.shape)

        return Layer_1_z1, Layer_1_a


if __name__ == "__main__":

    utils = Utils()
    X_train, y_train, X_test, y_test = utils.get_data()

    # define training constants
    learning_rate = 1
    number_of_epochs = 4
    np.random.seed(48)  # set seed value so that the results are reproduceable

    # Our network architecture has the shape:
    #                   (input)--> [Linear->Sigmoid] -> [Linear->Sigmoid] -->(output)

    # ------ LAYER-1 ----- define hidden layer that takes in training data
    Layer_1_z = LinearLayer(input_shape=X_train.shape, n_out=64, ini_type='xavier')
    Layer_1_a = ActivationLayer(Layer_1_z.Z.shape)

    # ------ LAYER-2 ----- define output layer that take is values from hidden layer
    Layer_2_z = LinearLayer(input_shape=Layer_1_a.A.shape, n_out=1, ini_type='xavier')
    Layer_2_a = ActivationLayer(Layer_2_z.Z.shape)

    # see what random weights and bias were selected and their shape
    costs = []  # initially empty list, this will store all the costs after a certian number of epochs
    predict_output = []
    # Start training
    for epoch in range(number_of_epochs):

        # ------------------------- forward-prop -------------------------
        Layer_1_z.forward(X_train)
        Layer_1_a.forward(Layer_1_z.Z)

        Layer_2_z.forward(Layer_1_a.A)
        Layer_2_a.forward(Layer_2_z.Z)

        # ---------------------- Compute Cost ----------------------------
        cost, dA2 = utils.compute_cost(Y=y_train, Y_hat=Layer_2_a.A)

        # print and store Costs every 100 iterations.
        if (epoch % 2) == 0:
            print("Cost at epoch#{}: {}".format(epoch, cost))
            costs.append(cost)

        # ------------------------- back-prop ----------------------------
        Layer_2_a.backward(dA2)
        Layer_2_z.backward(Layer_2_a.dZ)

        Layer_1_a.backward(Layer_2_z.dA_prev)
        Layer_1_z.backward(Layer_1_a.dZ)

        # ----------------------- Update weights and bias ----------------
        Layer_2_z.update_params(learning_rate=learning_rate)
        Layer_1_z.update_params(learning_rate=learning_rate)

    pred_outputs, _, accuracy = utils.predict(X=X_test, Y=y_test, Zs=[Layer_1_z, Layer_2_z], As=[Layer_1_a, Layer_2_a])
    print("The predicted outputs:\n {}".format(pred_outputs))
    print("The accuracy of the model is: {}%".format(accuracy))
    utils.plot_learning_curve(costs=costs, learning_rate=learning_rate, total_epochs=number_of_epochs)
