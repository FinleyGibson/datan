import unittest

import matplotlib.pyplot as plt

from datan.plot import plot_confusion_matrix

import numpy as np


class TestPlotConfusionMatrix(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.y = np.hstack((np.zeros(10), np.ones(10)))
        cls.y_ = cls.y.copy()
        cls.y_[15:] = 0.

    def test_confusion_matrix(self):
        fig = plot_confusion_matrix(self.y, self.y_)
        self.assertIsInstance(fig, plt.Figure)
        fig.show()


if __name__ == '__main__':
    # to run all tests:
    unittest.main()

    # or to be specific:
    # unittest.main(TestSomething)
