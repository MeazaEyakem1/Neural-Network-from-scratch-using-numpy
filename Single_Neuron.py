from __future__ import print_function
import numpy as np

# In this first part, we just prepare our data (mnist)
# for training and testing

# import keras
from numpy import where

from tensorflow.keras.datasets import mnist


def process_data():
    # retrieving the data from mnist dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    num_pixels = X_train.shape[1] * X_train.shape[2]

    # reshaping the data to a column vector
    X_train = X_train.reshape(X_train.shape[0], num_pixels).T
    X_test = X_test.reshape(X_test.shape[0], num_pixels).T
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    X_train = X_train / 255
    X_test = X_test / 255

    # We want to have a binary classification: digit 0 is classified 1 and
    # all the other digits are classified 0

    y_new = np.zeros(y_train.shape)
    y_new[np.where(y_train == 0.0)[0]] = 1
    y_train = y_new

    y_new = np.zeros(y_test.shape)
    y_new[np.where(y_test == 0.0)[0]] = 1
    y_test = y_new

    y_train = y_train.T
    y_test = y_test.T

    m = X_train.shape[1]  # number of examples

    # Now, we shuffle the training set
    np.random.seed(138)
    shuffle_index = np.random.permutation(m)
    X_train, y_train = X_train[:, shuffle_index], y_train[:, shuffle_index]

    dataset = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}

    return dataset


def sigmod_der(z):
    # caclculating the derivative of the sigmoid
    y = sigmoid(z)
    result = y * (1 - y)
    return result


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Cross Entropy for Binary class classification
def CrossEntropy(y, yHat, classes=2):
    if classes == 2:
        loss = -(y * np.log(yHat, where=yHat != 0) + (1 - y) * np.log(1 - yHat, where=(1 - yHat) != 0))
        loss = loss
    else:  # if multiclass classification
        loss = -np.sum(y * np.log(yHat))

    return loss


def plot_curve(train_losses, valid_losses, valid_accuracy, train_accuracy):
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, '-o', label='Training loss')
    plt.plot(epochs, valid_losses, '-o', label='Validation loss')

    plt.legend()
    plt.title('Learning curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    plt.plot(epochs, train_accuracy, '-o', label="Training_Accuracy")
    plt.plot(epochs, valid_accuracy, '-o', label="Validation_Accuracy")
    plt.title('Accuracy curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()


class Network():

    def __init__(self, size):
        self.weight = np.random.rand(size, 1) * 0.001
        self.bias = np.random.rand(1, 1) * 0.001

    def forward(self, dataset):
        z = np.dot(self.weight.T, dataset) + self.bias
        y_pred = sigmoid(z)

        return z, y_pred

    def get_gradiants(self, dataset):
        x_train = dataset["X_train"]
        y_train = dataset["y_train"]

        z, y_new = self.forward(x_train)

        # calculating the derivative of the loss with respect to the weight and bias
        # we use chain rule to get those
        # first we caluclate dl_dz,dl_da,

        # dl_da = (-y_train / y_new) + ((1 - y_train) / (1 - y_new))
        # da_dz = sigmod_der(z)

        m = x_train.shape[1]
        dz_dw = x_train.T

        # using a chain rule
        # dl_dz = dl_da * da_dz
        dl_dz = y_new - y_train

        dl_dw = np.dot(dl_dz, dz_dw) / m

        dl_db = np.sum(dl_dz, axis=1, keepdims=True) / m

        der = {'dl_dw': dl_dw, 'dl_db': dl_db}
        return der

    def back_propagation(self, dataset, lr):
        # update the weights and bias using the obtained derivatives
        der = self.get_gradiants(dataset)

        self.weight = self.weight - lr * der["dl_dw"].T
        self.bias = self.bias - lr * der["dl_db"]

    def fit(self, epochs, dataset, lr):
        train_losses = []
        valid_losses = []
        valid_accuracy = []
        train_accuracy = []

        for epoch in range(epochs):
            train_loss, accuracy_t = self.train_one_epoch(dataset, lr)
            valid_loss, accuracy = self.test_one_epoch(dataset)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            valid_accuracy.append(accuracy)
            train_accuracy.append(accuracy_t)

            print("Epoch {}% loss: {}%  valid_loss : {}%  train_accuracy: {}% Valide_Accuracy: {}% ".format(epoch,
                                                                                                            train_loss,
                                                                                                            valid_loss,
                                                                                                            accuracy_t * 100,
                                                                                                            accuracy * 100))

        plot_curve(train_losses, valid_losses, valid_accuracy, train_accuracy)
        return

    def train_one_epoch(self, dataset, lr):
        accuracy = []

        x_train = dataset["X_train"]

        z, y_pred = self.forward(x_train)

        loss = CrossEntropy(dataset["y_train"], y_pred)

        self.back_propagation(dataset, lr)

        mean_loss = np.array(loss).mean()

        y_new = np.zeros_like(y_pred)
        y_new[y_pred >= 0.5] = 1
        target = dataset["y_train"]
        accuracy = np.mean(y_new == target)

        return mean_loss, accuracy

    def test_one_epoch(self, dataset):
        loss = []
        accuracy = []

        x_test = dataset["X_test"]
        y, prediction = self.forward(x_test)

        loss = CrossEntropy(dataset["y_test"], prediction)

        y_pred = np.zeros_like(prediction)
        y_pred[prediction >= 0.5] = 1
        target = dataset["y_test"]
        accuracy = np.mean(y_pred == target)

        return np.array(loss).mean(), accuracy


if __name__ == "__main__":
    dataset = process_data()
    model = Network(784)
    prediction = model.fit(200, dataset, 5)

#
# i = 3
# print('y[{}]={}'.format(i, y_train[:,i]))
# plt.imshow(X_train[:,i].reshape(28,28), cmap = matplotlib.cm.binary)
# plt.axis("off")
# plt.show()

