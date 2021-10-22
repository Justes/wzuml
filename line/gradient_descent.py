import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def batch_gradient_descent(x, y, w, alpha, epoch):
    tmp = np.matrix(np.zeros(w.shape))
    cost = np.zeros(epoch)
    paras = w.ravel().shape[1]

    for i in range(epoch):
        error = x * w.T - y
        for j in range(paras):
            val = np.multiply(error, x[:, j])
            tmp[0, j] = w[0, j] - alpha / len(x) * np.sum(val)
        w = tmp
        cost[i] = compute_cost(x, y, w)

    return w, cost

def stochastic_gradient_descent(x, y, w, alpha, epoch):
    tmp = np.matrix(np.zeros(w.shape))
    cost = np.zeros(epoch)
    paras = w.ravel().shape[1]

    for i in range(epoch):
        for a in range(len(x)):
            error = x[a:a+1] * w.T - y[a]

            for j in range(paras):
                tmp[0, j] = w[0, j] - alpha * error * x[a:a+1, j]
            w = tmp
            cost[i] = compute_cost(x, y, w)
    return w, cost

def mini_batch_gradient_descent(x, y, w, alpha, mb):
    tmp = np.matrix(np.zeros(w.shape))
    cost = np.zeros(epoch)
    paras = w.ravel().shape[1]
    xlen = len(x)

    for i in range(epoch):
        for a in range(0, xlen, mb):
            x1 = x[a:a+mb]
            y1 = y[a:a+mb]
            error = x1 * w.T - y1

            for j in range(paras):
                val = np.multiply(error, x1[:, j])
                tmp[0, j] = w[0, j] - alpha / len(x1) * np.sum(val)
            w = tmp
            cost[i] = compute_cost(x1, y1, w)

    return w, cost

def compute_cost(x, y, w):
    return np.sum(np.power(x * w.T - y, 2)) / (2 * x.shape[0])


if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    np.set_printoptions(suppress=True)

    path = 'regress_data1.csv'
    data = pd.read_csv(path)

    x = data.iloc[:, [0]]
    y = data.iloc[:, [1]]

    #plt.scatter(x, y, s=16)
    plt.figure(facecolor='w', figsize=(12, 8))
    plt.plot(x, y, 'o', ms=4)
    plt.xlabel('人口')
    plt.ylabel('收益')
    #plt.show()

    data.insert(0, 'Ones', 1)
    x = data.iloc[:, :2]
    w = np.matrix(np.zeros((1, 2)))
    x = np.matrix(x.values)
    y = np.matrix(y.values)
    
    alpha = 0.01
    epoch = 1000
    mb = 32
    g, cost = batch_gradient_descent(x, y, w, alpha, epoch)
    print("批量梯度下降")
    print(g)
    y_hat = x * g.T
    mse = np.average((np.array(y_hat) - np.array(y)) ** 2)
    print("损失")
    print(mse)

    g, cost = stochastic_gradient_descent(x, y, w, alpha, epoch)
    print("随机梯度下降")
    print(g)
    y_hat1 = x * g.T
    mse = np.average((np.array(y_hat1) - np.array(y)) ** 2)
    print("损失")
    print(mse)

    g, cost = mini_batch_gradient_descent(x, y, w, alpha, mb)
    print("小批量梯度下降")
    print(g)
    y_hat2 = x * g.T
    mse = np.average((np.array(y_hat2) - np.array(y)) ** 2)
    print("损失")
    print(mse)

    linear = LinearRegression()
    linear.fit(x, y)
    print("sklearn 线性回归")
    print(linear.intercept_, linear.coef_[0])

    y_hat3 = linear.predict(x)
    mse = np.average((y_hat3 - np.array(y)) ** 2)
    print("损失")
    print(mse)

    x = data.iloc[:, 1:2]

    plt.plot(x, y_hat, 'r', label='批')
    plt.plot(x, y_hat1, 'g', label="随机")
    plt.plot(x, y_hat2, 'b', label="小批")
    plt.plot(x, y_hat3, 'y', label="sk")
    plt.legend()
    plt.show()
