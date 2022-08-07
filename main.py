# Neural Network From Scratch using only numpy 

'''
Resources :- https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras/notebook
Neural Networks from Scratch By Harrison Kinsely and 
Daniel Kukeila 
'''


from tensorflow import keras 
import numpy as np 
from matplotlib import pyplot as plt 


# Load in the data (Keras is used only for that)
data = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = data.load_data()


# Basic data processing
x_train = x_train / 255.0
x_test = x_test / 255.0 


# Creating the copies for the data to plot during predictions
x_train_copy = x_train
x_test_copy = x_test

training_size = 100 
x_train = x_train[0: training_size]
x_test = x_test[0: training_size]
y_train = y_train[0: training_size]
y_test = y_test[0: training_size]

x_train = np.reshape(x_train, (training_size, 784))
x_test = np.reshape(x_test, (training_size, 784))


# Initialize the weights and biases 
def init_params():
    W1 = 0.01 * np.random.randn(10, 784)
    b1 = 0.01 * np.random.randn(10, 1)

    W2 = 0.01 * np.random.randn(10, 10)
    b2 = 0.01 * np.random.randn(10, 1)

    return W1, b1, W2, b2


# Hyperparams
class ReLU():
    def forward(self, a):
        return np.maximum(a, 0)


    # Derivative of the relu activation function
    def backward(self, x):
        return x > 0

class Softmax():
    def forward(self, a):
        return np.exp(a) / sum(np.exp(a))

relu = ReLU()
softmax = Softmax()

# Forward prop 
def forward_prop(inputs, W1, b1, W2, b2):
    A1 = np.dot(inputs, W1.T) + b1.T
    Z1 = relu.forward(A1)

    A2 = np.dot(Z1, W2) + b2.T
    Z2 = softmax.forward(A2)

    return A1, Z1, A2, Z2


# One hot encoder
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1

    # Transpose to get 10 class names in the y_train data 
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


# Backprop 
def backward_prop(inputs, Z1, A1, Z2, A2, W1, W2, num_of_samples):
    one_hot_Y = one_hot(y_train)

    # Calculating derivatives from dW2 to dW1
    dZ2 = A2.T - one_hot_Y
    dW2 = A1.T.dot(dZ2.T)
    dB2 = np.sum(dZ2)
    drelu = relu.backward(Z1)
    dZ1 = dZ2.T.dot(W2) * drelu
    dW1 = dZ1.T.dot(inputs)
    dB1 = np.sum(dZ1)

    return dW1, dB1, dW2, dB2


# Creating the param update function 
def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2 
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2


def get_predictions(A2):
    # print(A2.T.shape)
    return np.argmax(A2.T, 0)

def get_accuracy(predictions, Y):
    # print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

W1, b1, W2, b2 = init_params()
for i in range(1, 100 + 1):
    A1, Z1, A2, Z2 = forward_prop(x_train, W1, b1, W2, b2)
    dW1, db1, dW2, db2 = backward_prop(x_train, Z1, A1, Z2, A2, W1, W2, training_size)
    W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, 0.001)
    predictions = get_predictions(A2)
    accuracy = get_accuracy(predictions, y_train)
    print(f"Accuracy: {accuracy}")


def make_predictions(X, W1, b1, W2, b2):
    _, _, A2, _ = forward_prop(X, W1, b1, W2, b2)
    predictions = get_predictions(A2)
    return predictions, A2


for i in range(10):
    pred, A_2 = make_predictions(x_test[i], W1, b1, W2, b2)
    plt.imshow(x_test_copy[i], cmap='binary')
    plt.title(f"Prediction: {pred}")
    plt.xlabel(f"Actual: {y_test[i]}")
    plt.show()

    print(pred)
