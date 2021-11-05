import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt
from sklearn import linear_model

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient(w, x, y, alpha):
    w = np.matrix(w)
    x = np.matrix(x)
    y = np.matrix(y)
    paras = w.shape[1]
    grad = np.zeros(paras)
    xlen = len(x)

    error = sigmoid(x * w.T) - y
    for j in range(paras):
        val = np.multiply(error, x[:, j])

        if j == 0:
            grad[j] = np.sum(val) / xlen
        else :
            grad[j] = np.sum(val) / xlen + lamda / xlen * w[:,j]

    return grad

def gradient_descent(w, x, y, alpha, lamda, epoch):
    w = np.matrix(w)
    x = np.matrix(x)
    y = np.matrix(y)
    paras = w.shape[1]
    grad = np.zeros(w.shape)
    xlen = len(x)

    for i in range(epoch):
        error = sigmoid(x * w.T) - y
        for j in range(paras):
            val = np.multiply(error, x[:, j])

            if j == 0:
                grad[0,j] = w[0,j] - alpha * np.sum(val) / xlen
            else :
                grad[0,j] = w[0,j] - alpha * (np.sum(val) / xlen) + lamda / xlen * w[0,j]
        w = grad
    return w

def gradient_reg(w, x, y, lamda):
    w = np.matrix(w)
    x = np.matrix(x)
    y = np.matrix(y)
    paras = int(w.shape[1])
    grad = np.zeros(w.shape[1])
    xlen = len(x)

    error = sigmoid(x * w.T) - y
    for j in range(paras):
        val = np.multiply(error, x[:, j])

        if j == 0:
            grad[j] = np.sum(val) / xlen
        else :
            grad[j] = np.sum(val) / xlen + lamda / xlen * w[:,j]
    return grad

def cost(w, x, y, lamda):
    w = np.matrix(w)
    x = np.matrix(x)
    y = np.matrix(y)
    xlen = len(x)

    reg = lamda * np.sum(np.power(w[:, 1:w.shape[1]], 2)) / (2 * xlen)
    c = np.sum(np.multiply(-y, np.log(sigmoid(x * w.T))) - np.multiply((1 - y), np.log(1 - sigmoid(x * w.T)))) / xlen + reg

    return c

def predict(x, w):
    x = np.matrix(x)
    w = np.matrix(w)
    probability = sigmoid(x * w.T)
    return [1 if p >= 0.5 else 0 for p in probability]


if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['simhei']
    plt.rcParams['axes.unicode_minus'] = False
    np.set_printoptions(suppress=True)

    path = 'ex2data1.txt'
    data = pd.read_csv(path, header=None, names=['e1', 'e2', 'admitted'])
    cols = data.shape[1]
    x = data.iloc[:, 0:cols-1]
    y = data.iloc[:, cols-1:cols]

    positive = data[data['admitted'].isin([1])]
    negative = data[data['admitted'].isin([0])]

    plt.figure(facecolor='w', figsize=(12, 8))
    plt.scatter(positive['e1'], positive['e2'], s=50, c='b', marker='o', label='admitted')
    plt.scatter(negative['e1'], negative['e2'], s=50, c='r', marker='x', label='not admitted')
    plt.legend()
    plt.title('two tests result')
    plt.xlabel('e1')
    plt.ylabel('e2')
    #plt.show()

    """
    nums = np.arange(-10, 10, 1)
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(nums, sigmoid(nums), 'r')
    plt.show()
    """

    data.insert(0, 'Ones', 1)
    cols = data.shape[1]
    x = data.iloc[:, 0:cols-1]
    x = np.array(x.values)
    y = np.array(y.values)
    w = np.zeros(x.shape[1])

    epoch = 150000
    alpha = 0.01
    lamda = 1

    g = gradient_descent(w, x, y, alpha, lamda, epoch)
    print(g)
    c = cost(w, x, y, lamda)
    print(c)
    predictions = predict(x, g)
    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
    accuracy = (sum(map(int, correct)) % len(correct))
    print('accuracy = {0}%'.format(accuracy))

    result = opt.fmin_tnc(func=cost, x0=w, fprime=gradient, args=(x, y, lamda))
    print(result)

    w_min = np.matrix(result[0])
    predictions = predict(x, w_min)
    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
    accuracy = (sum(map(int, correct)) % len(correct))
    print('accuracy = {0}%'.format(accuracy))

    model = linear_model.LogisticRegression(solver='liblinear', penalty='l2', C=1.0)
    model.fit(x, y.ravel())
    score = model.score(x, y)
    print(model.intercept_, model.coef_)
    print(score)

    y_hat = model.predict(x)
    y_test = y.ravel()
    right = sum(y_hat == y_test)
    y_hat = np.hstack((y_hat.reshape(-1,1), y_test.reshape(-1,1))) 
    print('accuracy = {0}%'.format(int(right * 100.0 / y_hat.shape[0])))
