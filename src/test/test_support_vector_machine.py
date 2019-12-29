import unittest
import numpy

from src.support_vector_machine import SupportVectorMachine

features_a = [[1, 7], [2, 8], [3, 8]]
features_b = [[5, 1], [6, -1], [7, 3]]
data = {-1: numpy.array(features_a), 1: numpy.array(features_b)}


class SupportVectorMachineTest(unittest.TestCase):
    def test_whenFittingData_thenDataIsCorrectlyFitted(self):
        support_vector_machine = SupportVectorMachine()

        support_vector_machine.fit(data, [0.1, 0.01])

        self.assertAlmostEqual(0.3200, support_vector_machine.w[0], delta=0.0001)
        self.assertAlmostEqual(-0.3200, support_vector_machine.w[1], delta=0.0001)
        self.assertAlmostEqual(0.3999, support_vector_machine.b, delta=0.0001)
