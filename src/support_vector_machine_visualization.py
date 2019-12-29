from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

from matplotlib import pyplot, style

style.use("ggplot")


Number = Union[int, float]


class SupportVectorMachineVisualization:
    def __init__(self, color_mapping: Dict[int, str]) -> None:
        self.color_mapping = color_mapping
        self.axis = pyplot.figure().add_subplot(1, 1, 1)

    def draw_data_points(self, data) -> None:
        for data_class, feature_set in data.items():
            for feature in feature_set:
                self._draw_point(feature, data_class)

    def draw_prediction_point(
        self, data_point: Tuple[Number, Number], data_class: int
    ) -> None:
        self._draw_point(data_point, data_class, size=200, marker="*")

    def _draw_point(
        self,
        data_point: Tuple[Number, Number],
        data_class: int,
        size: int = 100,
        marker: Optional[str] = None,
    ) -> None:
        self.axis.scatter(
            data_point[0],
            data_point[1],
            s=size,
            marker=marker,
            color=self.color_mapping[data_class],
        )

    def draw_hyperplanes(
        self, configuration: SupportVectorMachineConfiguration
    ) -> None:
        self._draw_hyperplane(configuration, 1, "k")
        self._draw_hyperplane(configuration, 0, "y--")
        self._draw_hyperplane(configuration, -1, "k")

    def _draw_hyperplane(
        self, configuration: SupportVectorMachineConfiguration, offset: int, color: str
    ) -> None:
        def hyperplane(x, v):
            w, b = configuration.w, configuration.b
            return (-w[0] * x - b + v) / w[1]

        hyperplane_x_min = configuration.min_feature_value * 0.9
        hyperplane_x_max = configuration.max_feature_value * 1.1
        hyperplane_y_min = hyperplane(hyperplane_x_min, offset)
        hyperplane_y_max = hyperplane(hyperplane_x_max, offset)
        self._draw_line(
            [hyperplane_x_min, hyperplane_y_min],
            [hyperplane_x_max, hyperplane_y_max],
            color,
        )

    def _draw_line(
        self, point_A: Tuple[Number, Number], point_B: Tuple[Number, Number], color: str
    ) -> None:
        self.axis.plot([point_A[0], point_B[0]], [point_A[1], point_B[1]], color)

    def show(self) -> None:
        pyplot.show()


@dataclass
class SupportVectorMachineConfiguration:
    w: Tuple[float, float]
    b: float
    min_feature_value: float
    max_feature_value: float
