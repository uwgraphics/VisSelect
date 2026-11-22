# --- Imports ------------------------------------------------------------------

# standard library
from itertools import combinations
from typing import Literal

# third party
import numpy as np
from scipy.cluster.vq import kmeans2


# --- Transform Functions ------------------------------------------------------

def scale(
    data: np.ndarray,
    interval: tuple[float, float] = (0.0, 1.0),
    indices: list[int] | None = None
) -> np.ndarray:
    """
    Scale the features at the indices to the interval, each from its own
    range of values, leaving the other features unchanged

    Args:
        data: The data array to transform
        interval: The interval to scale the features to
        indices: The indices of the features to scale, or all by default
    """
    if indices is None:
        indices = range(data.shape[1])
    selected = data[:, indices]
    low = selected.min(axis=0)
    high = selected.max(axis=0)
    spans = np.where(high - low == 0, 1, high - low)
    scaled = (selected - low) / spans
    scaled = scaled * (interval[1] - interval[0]) + interval[0]

    result = data.astype(float) # a copy that scaled values survive in
    result[:, indices] = scaled
    return result

def discretize(
    data: np.ndarray,
    bins: int = 5,
    indices: list[int] | None = None,
    strategy: Literal["uniform", "quantile", "kmeans"] = "uniform",
    seed: int | None = None
) -> np.ndarray:
    """
    Discretize the features at the indices into ordinal bin numbers, with the
    bins spaced evenly, by quantiles, or around clusters of the values

    Args:
        data: The data array to transform
        bins: The number of bins to discretize each feature into
        indices: The indices of the features to discretize, or all by
            default
        strategy: How the bins divide each feature: "uniform" bins span its
            range evenly, "quantile" bins hold even shares of its items, and
            "kmeans" bins form around clusters of its values
        seed: The random seed for reproducibility

    Raises:
        ValueError: If the strategy is not one of the three
    """
    if indices is None:
        indices = range(data.shape[1])
    selected = data[:, indices]
    discretized = np.empty(selected.shape, dtype=int)
    for column in range(selected.shape[1]):
        values = selected[:, column]
        if strategy == "uniform":
            edges = np.linspace(values.min(), values.max(), bins + 1)
        elif strategy == "quantile":
            shares = np.linspace(0, 1, bins + 1)
            edges = np.unique(np.quantile(values, shares))
        elif strategy == "kmeans":
            centers, _ = kmeans2(
                values.astype(float), bins, minit="++", seed=seed
            )
            centers = np.sort(centers)
            edges = np.concatenate((
                [values.min()], (centers[1:] + centers[:-1]) / 2,
                [values.max()]
            ))
        else:
            raise ValueError("Strategy is not uniform, quantile or kmeans")
        inner = edges[1:-1] # values at an inner edge join the higher bin
        discretized[:, column] = np.searchsorted(inner, values, side="right")
    return discretized

def encode(
    data: np.ndarray,
    indices: list[int] | None = None,
    order: int = 1
) -> np.ndarray:
    """
    One hot encode the discrete features at the indices in every group of
    order features, one column per category of a group that appears in the
    data, 1 for the items in that category, after the unencoded features: at
    an order of 1 each feature is encoded alone, and at higher orders the
    categories are the combinations appearing in each group of features

    Args:
        data: The data array to transform
        indices: The indices of the discrete features to encode, or all
            by default
        order: The number of features encoded jointly in each group

    Raises:
        ValueError: If the order is not between 1 and the features encoded
    """
    if indices is None:
        indices = range(data.shape[1])
    indices = list(indices)
    if not 1 <= order <= len(indices):
        raise ValueError("Order is not between 1 and the features encoded")

    encoded = []
    for group in combinations(indices, order):
        values = data[:, group].astype(int)
        if order > 1: # one category per combination that appears
            sizes = [int(data[:, index].max()) + 1 for index in group]
            values = np.ravel_multi_index(values.T, dims=sizes)
        values = values.reshape(len(data), -1)[:, 0]
        encoded.append((values[:, None] == np.unique(values)).astype(float))

    others = np.ones(data.shape[1], dtype=bool)
    others[indices] = False
    return np.hstack([data[:, others]] + encoded)
