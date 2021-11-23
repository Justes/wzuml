import numpy as np
import matplotlib.pyplot as plt

def init_paras(layers_dim):
    L = len(layers_dim)
    parameters = {}
    for i in range(1, L):
        parameters["w" + str(i)] = np.random.random([layers_dim[i], layers_dim[i-1]])
        parameters["b" + str(i)] = np.zeros((layers_dim[i], 1))

    return parameters

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def forward(x, parameters):
    a = []
    z = []
    caches = {}
    a.append(x)
    z.append(x)
    layers = len(parameters) // 2

    for i in range(1, layers):
        z_temp = parameters["w" + str(i)].dot(a[i-1])  + parameters["b" + str(i)]
        z.append(z_temp)
        a.append(sigmoid(z_temp))

    z_temp = parameters["w" + str(layers)].dot(a[layers - 1]) + parameters["b" +str(layers)]
    z.append(z_temp)
    a.append(z_temp)
    caches["z"] = z
    caches["a"] = a

    return caches, a[layers]

def backward(parameters, caches, al, y):
    layers = len(parameters) // 2
    grades = {}
    m = y.shape[1]

    grades["dz" + str(layers)] = al - y
    grades["dw" + str(layers)] = grades["dz" + str(layers)].dot(caches["a"][layers - 1].T) / m
    grades["db" + str(layers)] = np.sum(grades["dz" + str(layers)], axis = 1, keepdims=True) / m

    for i in reversed(range(1, layers)):
        grades["dz" +str(i)] = parameters["w" + str(i+1)].T.dot(grades["dz" + str(i+1)]) * sigmoid_prime(caches["z"][i])
        grades["dw" +str(i)] = grades["dz" + str(i)].dot(caches["a"][i-1].T) / m
        grades["db" +str(i)] = np.sum(grades["dz" +str(i)], axis=1, keepdims=True) / m
    return grades

def update_grades(parameters, grades, learning_rate):
    layers = len(parameters) // 2
    for i in range(1, layers + 1):
        parameters["w" + str(i)] -= learning_rate * grades["dw" + str(i)]
        parameters["b" + str(i)] -= learning_rate * grades["db" + str(i)]
    return parameters

def compute_loss(al, y):
    return np.mean(np.square(al - y))

def load_data():
    x = np.array([[0.1, 0.2]])
    y = np.array([[0.2, 0.4]])

    #x = np.arange(0.1, 1.0, 0.5)
    #y = 20 * np.sin(2 * np.pi * x)
    #y = 3 * x + 2
    return x, y

if __name__ == "__main__":
    x, y = load_data()
    x = x.reshape(1, 2)
    y = y.reshape(1, 2)
    print(y)
    plt.scatter(x, y)
    parameters = init_paras([1, 2, 1])
    print(parameters)
    al = 0
    np.set_printoptions(suppress=True)

    for i in range(8000):
        caches, al = forward(x, parameters)
        grades = backward(parameters, caches, al, y)
        parameters = update_grades(parameters, grades, learning_rate=0.3)

        #if i % 100 == 0:
            #print(compute_loss(al, y))
    
    print(al)
    plt.scatter(x, al)
    plt.show()
