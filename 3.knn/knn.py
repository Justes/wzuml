import numpy as np
import pandas as pd
import math
import sys
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier

def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = [
        'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
    ]
    data = np.array(df.iloc[:, :])
    return df, data[:, :-1], data[:, -1]

if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    np.set_printoptions(threshold=sys.maxsize)

    df, x, y = create_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    plt.figure(figsize=(12, 8))
    plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
    plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
    plt.xlabel('sepal length', fontsize=18)
    plt.ylabel('sepal width', fontsize=18)
    plt.legend()
    #plt.show()

    clf_sk = KNeighborsClassifier(n_neighbors=3)
    clf_sk.fit(x_train, y_train)
    score = clf_sk.score(x_test, y_test)
    print(score)
