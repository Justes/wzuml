import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient(w, X, y):
    w = np.matrix(w)
    parameters = int(w.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * w.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        grad[i] = np.sum(term) / len(X)

    return grad

def batch_gradient_descent(x, y, w):
    tmp = np.zeros(w.shape)
    paras = w.shape[1]
    grad = np.zeros(paras)
    xlen = len(x)

    error = sigmoid(x * w.T) - y
    for j in range(paras):
        val = np.multiply(error, x[:, j])
        grad[j] = np.sum(val) / len(x)

    return grad

def compute_cost(x, y, w):
    xlen = len(x)
    #reg = lamda * np.sum(np.multiply(w, w)) / (2 * xlen)
    return (np.sum(np.multiply(-y, np.log(sigmoid(x * w.T))) - np.multiply((1 - y), np.log(sigmoid(x * w.T))))) / xlen

def cost(w, X, y):
    w = np.matrix(w)
    first = np.multiply(-y, np.log(sigmoid(X * w.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * w.T)))
    return np.sum(first - second) / (len(X))

if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['simhei']
    plt.rcParams['axes.unicode_minus'] = False
    np.set_printoptions(suppress=True)

    path = 'ex2data1.txt'
    data = pd.read_csv(path, header=None, names=['Exam1', 'Exam2', 'Admitted'])

    positive = data[data['Admitted'].isin([1])]
    negative = data[data['Admitted'].isin([0])]

    data.insert(0, 'Ones', 1)
    cols = data.shape[1]

    x = data.iloc[:,0:cols-1]
    y = data.iloc[:,cols-1:cols]

    """
    plt.figure(facecolor='w', figsize=(12, 8))
    plt.scatter(positive['Exam1'], positive['Exam2'], s=50, c='b', marker='o', label='Admitted')
    plt.scatter(negative['Exam1'], negative['Exam2'], s=50, c='r', marker='x', label='Not Admitted')
    plt.legend()
    plt.title('人口收益关系图')
    plt.xlabel('Exam1')
    plt.ylabel('exam2')
    """
    #plt.show()

    x = np.matrix(np.array(x.values))
    y = np.matrix(np.array(y.values))
    w = np.zeros(3)

    print(x.shape)
    print(y.shape)
    print(w.shape)

    #grad = batch_gradient_descent(x, y, w)
    result = opt.fmin_tnc(func=cost, x0=w, fprime=gradient, args=(x, y))

    cost = cost(w, x, y)
    #print(grad)
    print(result)
    print(cost)
