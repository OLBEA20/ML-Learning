import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from jivago_streams import Stream
from typing import List
style.use('ggplot')


class SupportVectorMachine:

    def __init__(self, data, visulization=True):
        self.visulization = visulization
        self.colors = {1: 'r', -1: 'b'}
        if self.visulization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)
        self._initialization(data)

    def _initialization(self, data):
        self.data = data
        all_data = (Stream(self.data)
                    .map(lambda key: self.data[key])
                    .flat()
                    .flat()
                    .toList())
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)

    def fit(self,
            step_sizes: List[float],
            b_range_multiple: int = 5,
            b_multiple: int = 5):
        opt_dict = {}
        transforms = [[1, 1],
                      [-1, 1],
                      [-1, -1],
                      [1, -1]]

        step_sizes = (Stream(step_sizes)
                      .map(lambda step_size: step_size*self.max_feature_value)
                      .toList())
        latest_optimum = self.max_feature_value * 10

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            optimized = False
            while not optimized:
                for b in np.arange(-1 *
                                   self.max_feature_value * b_range_multiple,
                                   self.max_feature_value * b_range_multiple,
                                   step * b_multiple):
                    for transformation in transforms:
                        w_t = w * transformation
                        if self._all_data_satisfy_condition(w_t, b):
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]
                if w[0] < 0:
                    optimized = True
                    print('Optimized a step.')
                else:
                    w = w - step
            norms = sorted([n for n in opt_dict])

            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step * 2

    def _all_data_satisfy_condition(self, w_t, b):
        for key in self.data:
            for x_i in self.data[key]:
                y_i = key
                if not y_i * (np.dot(w_t, x_i) + b) >= 1:
                    return False
        return True

    def predict(self, features):
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification != 0 and self.visulization:
            self.ax.scatter(features[0],
                            features[1],
                            s=200,
                            marker='*',
                            c=self.colors[classification])

        return classification

    def visualize(self):
        [[self.ax.scatter(x[0],
                          x[1],
                          s=100,
                          color=self.colors[i])
            for x in data_dict[i]]
            for i in data_dict]

        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]\

        data_range = (self.min_feature_value * 0.9,
                      self.max_feature_value * 1.1)
        hyperplane_x_min = data_range[0]
        hyperplane_x_max = data_range[1]

        psv1 = hyperplane(hyperplane_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyperplane_x_max, self.w, self.b, 1)
        self.ax.plot([hyperplane_x_min, hyperplane_x_max], [psv1, psv2], 'k')

        nsv1 = hyperplane(hyperplane_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyperplane_x_max, self.w, self.b, -1)
        self.ax.plot([hyperplane_x_min, hyperplane_x_max], [nsv1, nsv2], 'k')

        db1 = hyperplane(hyperplane_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyperplane_x_max, self.w, self.b, 0)
        self.ax.plot([hyperplane_x_min, hyperplane_x_max], [db1, db2], 'y--')
        plt.show()


data_dict = {-1: np.array([[1, 7],
                           [2, 8],
                           [3, 8]]),
             1: np.array([[5, 1],
                          [6, -1],
                          [7, 3]])}

svm = SupportVectorMachine(data=data_dict)
step_sizes = [0.1, 0.01, 0.001]
svm.fit(step_sizes)

predict_us = [[0, 10],
              [1, 3],
              [3, 4],
              [5, 5],
              [5, 6],
              [6, -5],
              [5, 8]]

for p in predict_us:
    svm.predict(p)

svm.visualize()
