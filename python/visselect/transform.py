# --- Imports ------------------------------------------------------------------

# standard library
from itertools import combinations
from typing import Literal, Sequence

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
        settings: dict | None = None,
        features: Sequence[str] | None = None
    ) -> None:
        """
        Initialize a transform node holding a derived array

        Args:
            array: The array the node holds, one row per dataset item
            name: The name the node is looked up by
            parent: The node this one derives from, or None at the root
            transform: The name of the transform method that derived it
            settings: The settings the transform was applied with
            features: The names of the array's column features, numbered
                from 0 by default

        Raises:
            ValueError: If the rows of the array and its parent differ, a
                transform of the name is already in the tree, or the
                lengths of features and array columns differ
        """
        # rows together in memory: evaluations gather rows constantly,
        # and pandas hands tables over column-major
        array = np.ascontiguousarray(array)
        if parent is not None:
            if array.shape[0] != parent.array.shape[0]:
                raise ValueError("Lengths of array rows and parent differ")

            # names are how nodes are looked up, so one tree holds each once
            nodes = [parent]
            while nodes[0].parent is not None:
                nodes = [nodes[0].parent]
            while nodes:
                node = nodes.pop()
                if node.name == name:
                    raise ValueError(
                        f"A transform named {name} is already in the tree"
                    )
                nodes.extend(node.children)

        if features is None:
            features = [str(index) for index in range(array.shape[1])]
        elif len(features) != array.shape[1]:
            raise ValueError("Lengths of features and array columns differ")
        array = array.view()
        array.flags.writeable = False

        self.array = array
        self.name = name
        self.parent = parent
        self.transform = transform
        self.settings = settings or {}
        self.features = list(features)
        self.children = []
        if parent is not None:
            parent.children.append(self)

    def select(
        self,
        features: str | int | Sequence[str | int],
        name: str | None = None
    ) -> "Transform":
        """
        Derive a child holding only the chosen features, in the order
        given; a numeric selection out of a mixed table becomes a numeric
        array, so objectives read clean numbers beside identity columns

        Args:
            features: The feature or features to keep, each by its name or
                by its index
            name: The name the derived node is looked up by: by default the
                feature's own name when one is kept, and "selected" when
                several are

        Raises:
            ValueError: If no features are chosen, or a feature is not a
                name or an index of the array
        """
        if isinstance(features, (str, int)):
            features = [features]
        if not len(features):
            raise ValueError("No features chosen")
        indices = []
        for feature in features:
            if isinstance(feature, str):
                if feature not in self.features:
                    raise ValueError(f"No feature named {feature}")
                feature = self.features.index(feature)
            elif not 0 <= feature < len(self.features):
                raise ValueError(f"No feature at index {feature}")
            indices.append(feature)

        names = [self.features[index] for index in indices]
        if name is None:
            name = names[0] if len(names) == 1 else "selected"
        array = self.array[:, indices]
        if array.dtype == object: # numeric features leave the mixed table
            try:
                array = array.astype(float)
            except (TypeError, ValueError):
                pass # identity features stay as they are
        return Transform(array, name, self, "select",
                         {"features": names}, names)

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

        Raises:
            ValueError: If the transform holds text, which cannot be scaled
        """
        data = self.array
        if data.dtype.kind in "OUS": # text columns have no scale
            raise ValueError(f"Transform {self.name} is not numeric")
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
        return Transform(result, name, self, "scale", settings, self.features)

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
        around clusters of the values, leaving the other features unchanged

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
            ValueError: If the strategy is not one of the three, or the
                transform holds text, which cannot be binned
        """
        data = self.array
        if data.dtype.kind in "OUS": # text columns have no bins
            raise ValueError(f"Transform {self.name} is not numeric")
        settings = {"bins": bins, "indices": indices, "strategy": strategy,
                    "seed": seed}
        if indices is None:
            indices = range(data.shape[1])

        result = data.astype(float) # a copy the bin numbers are written to
        for index in indices:
            values = data[:, index]
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
            result[:, index] = np.searchsorted(inner, values, side="right")
        return Transform(result, name, self, "discretize", settings,
                         self.features)

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

        others = np.ones(data.shape[1], dtype=bool)
        others[indices] = False
        features = [self.features[index] for index in np.flatnonzero(others)]

        encoded = []
        for group in combinations(indices, order):
            if order == 1: # the categories are the values that appear
                values = data[:, group[0]]
            else: # one category per combination of values that appears
                codes = []
                sizes = []
                for index in group:
                    _, inverse = np.unique(data[:, index],
                                           return_inverse=True)
                    codes.append(inverse)
                    sizes.append(inverse.max() + 1)
                values = np.ravel_multi_index(codes, dims=sizes)
            categories = np.unique(values)
            encoded.append(
                (values[:, None] == categories).astype(float)
            )
            # each category's column is named for its group and value
            group = "+".join(self.features[index] for index in group)
            features.extend(
                f"{group}={category:g}" if isinstance(category, float)
                else f"{group}={category}" for category in categories
            )

        result = np.hstack([data[:, others]] + encoded)
        if result.dtype == object: # encoding text columns of a mixed table
            try:
                result = result.astype(float)
            except (TypeError, ValueError):
                pass # text features kept beside the encoding stay as they are
        return Transform(result, name, self, "encode", settings, features)
