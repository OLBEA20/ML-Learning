import numpy

from src.support_vector_machine import SupportVectorMachine
from src.support_vector_machine_visualization import SupportVectorMachineVisualization

if __name__ == "__main__":

    features_a = [[1, 7], [2, 8], [3, 8]]
    features_b = [[5, 1], [6, -1], [7, 3]]

    data = {-1: numpy.array(features_a), 1: numpy.array(features_b)}
    support_vector_machine = SupportVectorMachine()
    support_vector_machine.fit(data, [0.1, 0.01, 0.001])

    data_points_to_predict = [
        [0, 10],
        [1, 3],
        [3, 4],
        [3, 5],
        [5, 5],
        [5, 6],
        [6, -5],
        [5, 8],
    ]

    visualization = SupportVectorMachineVisualization({1: "b", -1: "r"})
    for point in data_points_to_predict:
        prediction = support_vector_machine.predict(point)
        visualization.draw_prediction_point(point, prediction)

    support_vector_machine.visualize(visualization)
