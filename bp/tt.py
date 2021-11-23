import numpy as np
import matplotlib.pyplot as plt

def init_paras(layers_dim):
    parameters = {}
    parameters["w1"] = np.random.random([layers_dim[1], layers_dim[0]])
    parameters["w2"] = np.random.random([layers_dim[2], layers_dim[1]])

    return parameters

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def forward(x, parameters):
    caches = {}

    in1 = x.dot(parameters["w1"].T)
    out1 = sigmoid(in1)

    in2 = out1.dot(parameters["w2"].T)
    y_hat = out2 = sigmoid(in2)

    caches["h_i"] = in1
    caches["h_o"] = out1

    caches["x"] = x

    """
    hidden_input = parameters["w1"].dot(x)
    hidden_output = sigmoid(hidden_input)

    y_hat = parameters["w2"].dot(hidden_output)
    caches["h_i"] = hidden_input
    caches["h_o"] = hidden_output
    caches["x"] = x
    """

    return caches, y_hat

def backward(parameters, caches, y_hat, y):
    grades = {}
    m = y.shape[1]

    grades["dz2"] = y_hat - y
    print(grades["dz2"])
    print('---')
    print(caches["h_o"])
    grades["dw2"] = grades["dz2"].dot(caches["h_o"].T) / m

    grades["dz1"] = grades["dz2"].dot(parameters["w2"]) * sigmoid_prime(caches["h_i"])
    grades["dw1"] = grades["dz1"].T.dot(caches["x"]) / m
    return grades

def update_grades(parameters, grades, learning_rate):
    parameters["w1"] -= learning_rate * grades["dw1"]
    parameters["w2"] -= learning_rate * grades["dw2"]
    return parameters

def compute_loss(y_hat, y):
    return np.mean(np.square(y_hat - y))

def load_data():
    x = np.array([[0.1, 0.2, 0.3]])
    y = np.array([[0.6]])
    #y = np.array([[0.3, 0.6]])
    #x = np.arange(0.1, 1.0, 0.5)
    #y = 20 * np.sin(2 * np.pi * x)
    #y = 3 * x + 2
    return x, y

if __name__ == "__main__":
    np.random.seed(1)
    np.set_printoptions(suppress=True)

    x, y = load_data()
    #x = x.reshape(1, 2)
    #y = y.reshape(1, 2)
    #plt.scatter(x, y)
    #parameters = init_paras([1, 4, 1])
    input_dim = 3
    hidden_dim = 2
    output_dim = 1
    parameters = {}
    parameters["w1"] = np.random.random([hidden_dim, input_dim])
    parameters["w2"] = np.random.random([output_dim, hidden_dim])
    print(parameters)
    y_hat = 0

    for i in range(1):
        caches, y_hat = forward(x, parameters)
        grades = backward(parameters, caches, y_hat, y)
        parameters = update_grades(parameters, grades, learning_rate=0.3)
    
    print(parameters)
    #out1 = sigmoid(x.dot(parameters["w1"].T))
    #out2 = out1.dot(parameters["w2"].T)
    print(y_hat)
    #plt.scatter(x, y_hat)
    plt.show()
