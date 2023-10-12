import numpy.typing as npt


def accuracy(y: npt.NDArray[float], y_: npt.NDArray[float]) -> float:
    """
    Simple function to report the accuracy percentage based on predited
    values and real values.
    """
    assert y.shape == y_.shape
    return ((y == y_).sum()/len(y))*100
