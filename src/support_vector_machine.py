from typing import Dict, Iterator, List

import numpy

from src.support_vector_machine_visualization import (
    SupportVectorMachineVisualization,
    SupportVectorMachineConfiguration,
)


TRANSFORMS = [[1, 1], [-1, 1], [-1, -1], [1, -1]]


class SupportVectorMachine:
    def __init__(self) -> None:
        self.b_range_multiple = 5
        self.b_multiple = 5
        self.data = {}

    def fit(self, data: Dict[int, numpy.array], step_size_coefficient: List[float]):
        self.data = data

        self.max_feature_value = max(self.features)
        self.min_feature_value = min(self.features)
        step_sizes = [
            coefficient * self.max_feature_value
            for coefficient in step_size_coefficient
        ]
        optimizations = self._calculate_optimizations(step_sizes)

        self.w = self._best(optimizations)[0]
        self.b = self._best(optimizations)[1]

    def _calculate_optimizations(self, step_sizes: List[float]):
        optimizations = {}
        latest_optimum = self.max_feature_value * 10
        for step in step_sizes:
            optimizations.update(
                self._calculate_optimizations_of_step(step, latest_optimum)
            )
            latest_optimum = self._best(optimizations)[0][0] + step * 2

        return optimizations

    def _calculate_optimizations_of_step(
        self, step: float, latest_optimum: float
    ) -> Dict:
        step_optimizations = {}
        w = numpy.array([latest_optimum, latest_optimum])
        while not self._optimized(w):
            for b in self._b_range_for(step):
                step_optimizations.update(self._calculate_optimization_for(w, b))

            w = w - step

        return step_optimizations

    def _optimized(self, w) -> bool:
        return w[0] < 0

    def _b_range_for(self, step) -> Iterator:
        return numpy.arange(
            -1 * self.max_feature_value * self.b_range_multiple,
            self.max_feature_value * self.b_range_multiple,
            step * self.b_multiple,
        )

    def _calculate_optimization_for(self, w, b):
        optimization = {}
        for transformation in TRANSFORMS:
            transformed_w = w * transformation
            if self._transformation_is_a_possible_optimization(transformed_w, b):
                optimization[magnitude_of(transformed_w)] = [transformed_w, b]

        return optimization

    def _transformation_is_a_possible_optimization(self, transformed_w, b) -> bool:
        for data_class, feature_set in self.data.items():
            for feature in feature_set:
                hyperplane_fit_feature = (
                    data_class * (dot_product(transformed_w, feature) + b) >= 1
                )
                if not hyperplane_fit_feature:
                    return False

        return True

    def _best(self, optimization):
        return optimization[min(optimization)]

    @property
    def features(self) -> List[int]:
        features = [
            feature for feature_set in self.data.values() for feature in feature_set
        ]
        return [value for feature in features for value in feature]

    def predict(self, features) -> int:
        return numpy.sign(dot_product(numpy.array(features), self.w) + self.b)

    def visualize(self, visualization: SupportVectorMachineVisualization):
        visualization.draw_data_points(self.data)
        visualization.draw_hyperplanes(
            SupportVectorMachineConfiguration(
                self.w, self.b, self.min_feature_value, self.max_feature_value
            )
        )
        visualization.show()


def magnitude_of(vector) -> float:
    return numpy.linalg.norm(vector)


def dot_product(vector_1, vector_2):
    return numpy.dot(vector_1, vector_2)
