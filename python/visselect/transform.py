# --- Imports ------------------------------------------------------------------

# standard library
from itertools import combinations
from typing import Literal

# third party
import numpy as np
from numpy.typing import ArrayLike
from scipy.cluster.vq import kmeans2


# --- Transform Tree -----------------------------------------------------------

class Transform:
    """
    A transform object is a node in the transform tree applying a transform
    to its parent and storing the resulting transformed data
    """

    def __init__(
        self,
        array: ArrayLike,
        name: str = "data",
        parent: "Transform | None" = None,
        transform: str | None = None,
        settings: dict | None = None
    ) -> None:
        """
        Initialize a transform node holding a derived array

        Args:
            array: The array the node holds, one row per dataset item
            name: The name the node is looked up by
            parent: The node this one derives from, or None at the root
            transform: The name of the transform method that derived it
            settings: The settings the transform was applied with

        Raises:
            ValueError: If the rows of the array and its parent differ
        """
        array = np.asarray(array)
        if parent is not None and array.shape[0] != parent.array.shape[0]:
            raise ValueError("Lengths of array rows and parent differ")
        array = array.view()
        array.flags.writeable = False

        self.array = array
        self.name = name
        self.parent = parent
        self.transform = transform
        self.settings = settings or {}
        self.children = []
        if parent is not None:
            parent.children.append(self)

    def scale(
        self,
        interval: tuple[float, float] = (0.0, 1.0),
        indices: list[int] | None = None,
        name: str = "scaled"
    ) -> "Transform":
        """
        Derive a child with the features at the indices scaled to the
        interval, each from its own range of values, leaving the other
        features unchanged

        Args:
            interval: The interval to scale the features to
            indices: The indices of the features to scale, or all by default
            name: The name the derived node is looked up by
        """
        data = self.array
        settings = {"interval": interval, "indices": indices}
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
        return Transform(result, name, self, "scale", settings)

    def discretize(
        self,
        bins: int = 5,
        indices: list[int] | None = None,
        strategy: Literal["uniform", "quantile", "kmeans"] = "uniform",
        seed: int | None = None,
        name: str = "discretized"
    ) -> "Transform":
        """
        Derive a child with the features at the indices discretized into
        ordinal bin numbers, with the bins spaced evenly, by quantiles, or
        around clusters of the values

        Args:
            bins: The number of bins to discretize each feature into
            indices: The indices of the features to discretize, or all by
                default
            strategy: How the bins divide each feature: "uniform" bins span
                its range evenly, "quantile" bins hold even shares of its
                items, and "kmeans" bins form around clusters of its values
            seed: The random seed for reproducibility
            name: The name the derived node is looked up by

        Raises:
            ValueError: If the strategy is not one of the three
        """
        data = self.array
        settings = {"bins": bins, "indices": indices, "strategy": strategy,
                    "seed": seed}
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
                raise ValueError(
                    "Strategy is not uniform, quantile or kmeans"
                )
            inner = edges[1:-1] # values at an inner edge join the higher bin
            discretized[:, column] = np.searchsorted(inner, values,
                                                     side="right")
        return Transform(discretized, name, self, "discretize", settings)

    def encode(
        self,
        indices: list[int] | None = None,
        order: int = 1,
        name: str = "encoded"
    ) -> "Transform":
        """
        Derive a child with the discrete features at the indices one hot
        encoded in every group of order features, one column per category of
        a group that appears in the data, 1 for the items in that category,
        after the unencoded features: at an order of 1 each feature is
        encoded alone, and at higher orders the categories are the
        combinations appearing in each group of features

        Args:
            indices: The indices of the discrete features to encode, or all
                by default
            order: The number of features encoded jointly in each group
            name: The name the derived node is looked up by

        Raises:
            ValueError: If the order is not between 1 and the features encoded
        """
        data = self.array
        settings = {"indices": indices, "order": order}
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
            encoded.append(
                (values[:, None] == np.unique(values)).astype(float)
            )

        others = np.ones(data.shape[1], dtype=bool)
        others[indices] = False
        result = np.hstack([data[:, others]] + encoded)
        return Transform(result, name, self, "encode", settings)
