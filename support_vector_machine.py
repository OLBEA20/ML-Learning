from typing import Dict

from matplotlib import pyplot
from matplotlib import style
import numpy

style.use("ggplot")

TRANSFORMS = [[1, 1], [-1, 1], [-1, -1], [1, -1]]


class SupportVectorMachine:
    def __init__(self, visualization=True) -> None:
        self.visualization = visualization
        self.colors = {1: "r", -1: "b"}

        if self.visualization:
            self.figure = pyplot.figure()
            self.axis = self.figure.add_subplot(1, 1, 1)

    def fit(self, data):
        self.data = data
        optimization = {}

        all_data = []
        for yi in self.data:
            for feature_set in self.data[yi]:
                for feature in feature_set:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = max(all_data)
        all_data = None

        step_sizes = [
            self.max_feature_value * 0.1,
            self.max_feature_value * 0.01,
            self.max_feature_value * 0.001,
        ]

        b_range_multiple = 5
        b_multiple = 5

        latest_optimum = self.max_feature_value * 10
        for step in step_sizes:
            w = numpy.array([latest_optimum, latest_optimum])
            optimized = False
            while not optimized:
                for b in numpy.arange(
                    -1 * self.max_feature_value * b_range_multiple,
                    self.max_feature_value * b_range_multiple,
                    step * b_multiple,
                ):
                    for transformation in TRANSFORMS:
                        transformed_w = w * transformation
                        found_option = True
                        for data_class in self.data:
                            for xi in self.data[data_class]:
                                yi = data_class
                                if not yi * (numpy.dot(transformed_w, xi) + b) >= 1:
                                    found_option = False
                        if found_option:
                            optimization[numpy.linalg.norm(transformed_w)] = [
                                transformed_w,
                                b,
                            ]

                if w[0] < 0:
                    optimized = True
                    print("optimized a step")
                else:
                    w = w - step

            magnitudes = sorted([magnitude for magnitude in optimization])
            optimization_choice = optimization[magnitudes[0]]
            self.w = optimization_choice[0]
            self.b = optimization_choice[1]
            latest_optimum = optimization_choice[0][0] + step * 2

    def predict(self, data):
        classification = numpy.sign(numpy.dot(numpy.array(features), self.w) + self.b)

        return classification


features_a = [[1, 7], [2, 8], [3, 8]]
features_b = [[6, 1], [6, -1], [7, 3]]

data = {-1: numpy.array(features_a), 1: numpy.array(features_b)}
