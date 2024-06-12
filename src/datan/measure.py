import numpy.typing as npt
from sklearn.metrics import r2_score


def accuracy(y: npt.NDArray[float], y_: npt.NDArray[float]) -> float:
    """
    Simple function to report the accuracy percentage based on predited
    values and real values.
    """
    assert y.shape == y_.shape
    return ((y == y_).sum()/len(y))*100


def r_squared(y: npt.NDArray[float], y_: npt.NDArray[float]) -> float:
    """
    Simple function to report the accuracy percentage based on predited
    values and real values.
    """
    assert y.shape == y_.shape
    return r2_score(y, y_)
