import numpy as np
import pandas as pd
import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.naive_bayes import GaussianNB

class NaiveBayes:
    def __init__(self):
        self.model = None

    @staticmethod
    def mean(x):
        return sum(x) / float(len(x))

    def stdev(self, x):
        avg = self.mean(x)
        return math.sqrt(sum([pow(i - avg, 2) for i in x]) / float(len(x)))

    def gaussian_probability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    def summarize(self, train_data):
        summaries = [(self.mean(i), self.stdev(i)) for i in zip(*train_data)]
        return summaries

    def fit(self, x, y):
        labels = list(set(y))
        data = {label:[] for label in labels}

        for f, label in zip(x, y):
            data[label].append(f)

        self.model = {
            label: self.summarize(value)
            for label, value in data.items()
        }
        return 'gaussianNB train done'

    def calculate_probabilities(self, input_data):
        probabilities = {}
        for label, value in self.model.items():
            probabilities[label] = 1
            for i in range(len(value)):
                mean, stdev = value[i]
                probabilities[label] *= self.gaussian_probability(input_data[i], mean, stdev)

        #print(probabilities.items())
        return probabilities

    def predict(self, x_test):
        values = self.calculate_probabilities(x_test).items()
        #dict_items([(0.0, 0.8000339998098573), (1.0, 7.729666640262751e-16)])
        #[(1.0, 7.729666640262751e-16), (0.0, 0.8000339998098573)]
        # 以 values 里元组的最后一个值从小到大排序, 排序后取最后一个元组里的第一个值 (label)
        # 可能性最大的是 0 还是 1
        label = sorted(values, key=lambda x:x[-1])[-1][0]
        return label

    def score(self, x_test, y_test):
        right = 0
        for x, y in zip(x_test, y_test):
            label = self.predict(x)
            if label == y:
                right += 1
        return right / float(len(x_test))

def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = [
        'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
    ]
    data = np.array(df.iloc[:100, :])
    return data[:, :-1], data[:, -1]

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    x, y = create_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    model = NaiveBayes()
    model.fit(x_train, y_train)
    print(model.predict([4.4, 3.2, 1.3, 0.2]))
    score = model.score(x_test, y_test)
    print(score)

    clf = GaussianNB()
    clf.fit(x_train, y_train)
    print(clf.predict([[4.4, 3.2, 1.3, 0.2]]))
    score = clf.score(x_test, y_test)
    print(score)
