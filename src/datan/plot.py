import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score, confusion_matrix
import functools
from pathlib import Path
import numpy.typing as npt
from typing import Iterable
import toml

DIR_ROOT: Path = Path(__file__).parent.parent.parent

with open(DIR_ROOT / "config.toml", "r") as infile:
    CONFIG = toml.load(infile)


def optional_axes(figsize=None):
    """
    Decorator that ensures the first argument passed to the decorated function
    is a `matplotlib.axes.Axes` object. If an axes is not provided, a new figure
    and axes are created, and the decorated function is called with the created
    axes as its first argument. The decorator returns the created figure.

    Parameters:
        figsize (tuple, optional): A tuple (width, height) representing the size
                                  of the created figure in inches. If not provided,
                                  the default figure size is used.

    Returns:
        callable: A decorated function that handles axes creation, respects
                  the provided figsize, and returns the created figure.

    Example:

    .. code-block:: python

        @ensure_axes_and_return_figure(figsize=(8, 6))
        def plot_data(ax, x, y):
            ax.plot(x, y)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('Data Plot')

        x_data = [1, 2, 3, 4]
        y_data = [10, 5, 8, 3]

        # Case 1: Pass an existing axes
        fig1 = plt.figure()
        existing_ax = fig1.add_subplot(111)
        plot_data(existing_ax, x_data, y_data)

        # Case 2: Don't pass an axes
        fig2 = plot_data(x_data, y_data)
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if len(args) == 0 or not isinstance(args[0], plt.Axes):
                if figsize:
                    fig, ax = plt.subplots(figsize=figsize)
                else:
                    fig, ax = plt.subplots()
                args = (ax,) + args
            else:
                ax = args[0]
                fig = ax.figure

            func(*args, **kwargs)
            return fig

        return wrapper

    return decorator


@optional_axes(figsize=CONFIG['report']['figsize']['square'])
def yy_plot(ax: plt.Axes, y, y_):

    ul = max(max(y), max(y_))
    ll = min(min(y), min(y_))
    rng = ul-ll

    ax.plot([ll-rng/20, ul+rng/20], [ll-rng/20, ul+rng/20], c="k",
            alpha=0.4, zorder=0)

    ax.scatter(y, y_, c="C0", zorder=1, alpha=0.6, s=5)

    ax.set_xlabel("Target Value")
    ax.set_ylabel("Prediction")
    r2 = r2_score(y, y_)
    ax.set_title(r"$R2={:.3f}$".format(r2))


@optional_axes(figsize=CONFIG['report']['figsize']['long'])
def coefficient_significance(ax: plt.Axes,
                             coefs: Iterable[float],
                             names: Iterable[str],
                             absolute: bool = False):

    coefs = np.asarray(coefs).flatten()
    names = np.asarray(names).flatten()
    assert coefs.shape == names.shape

    inds = np.argsort(coefs)[::-1]
    x = names[inds]

    if absolute:
        y = np.abs(coefs[inds])
    else:
        y = coefs[inds]

    ax.scatter(x, y, marker="_")
    plt.xticks(rotation=90)


@optional_axes(figsize=CONFIG['report']['figsize']['square'])
def binary_confusion_matrix(ax: plt.Axes,
                            y: Iterable[float],
                            y_: Iterable[float], **kwargs):

    cf_matrix = confusion_matrix(y_, y)

    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']

    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                         cf_matrix.flatten()/np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names, group_counts, group_percentages)]

    labels = np.asarray(labels).reshape(2, 2)

    plot_args = {
        "cmap": "Blues",
        "fmt": "",
        }
    plot_args.update(**kwargs)

    sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=labels, **plot_args, ax=ax)

