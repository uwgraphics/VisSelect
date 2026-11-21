# --- Imports ------------------------------------------------------------------

# third party
import numpy as np
from scipy.stats import skewnorm


# --- Random Dataset Generation ------------------------------------------------

def uniform(
    size: tuple[int, int] = (1000, 2),
    interval: tuple[float, float] = (1.0, 5.0),
    seed: int | None = None
) -> np.ndarray:
    """
    Generate random data from a uniform distribution

    Args:
        size: The number of items and features to generate
        interval: The range of values of the uniform distribution, inclusive
        seed: The seed for random generation to reproduce the data
    """
    rng = np.random.default_rng(seed)
    return _uniform(rng, interval, size)

def binary(
    size: tuple[int, int] = (1000, 2),
    probabilityInterval: tuple[float, float] = (0.5, 0.5),
    seed: int | None = None
) -> np.ndarray:
    """
    Generate random binary data of Bernoulli trials of random probability

    Args:
        size: The number of items and features to generate
        probabilityInterval: The range each feature's probability of
            success is drawn from, inclusive
        seed: The seed for random generation to reproduce the data
    """
    rng = np.random.default_rng(seed)
    probabilities = _uniform(rng, probabilityInterval, size[1]) # per feature
    return rng.binomial(1, probabilities, size=size)

def categories(
    size: tuple[int, int] = (1000, 2),
    categoriesInterval: tuple[int, int] = (2, 9),
    balance: float = 0.0,
    seed: int | None = None
) -> np.ndarray:
    """
    Generate random categorical data where each feature has a random number of
    categories, each with a random share of items

    Args:
        size: The number of items and features to generate
        categoriesInterval: The range each feature's number of categories
            is drawn from, inclusive
        balance: How balanced the categories' shares of items are, 0 to 1
        seed: The seed for random generation to reproduce the data

    Raises:
        ValueError: If the categories interval is not an increasing range from
            one category to at most one category per item
    """
    low, high = categoriesInterval
    if not 1 <= low <= high <= size[0]:
        raise ValueError(
            "Categories interval is not between 1 and the number of items"
        )

    rng = np.random.default_rng(seed)
    counts = rng.integers(
        categoriesInterval[0], categoriesInterval[1], size=size[1],
        endpoint=True
    )
    data = np.zeros(size, dtype=int)
    for i in range(size[1]):
        data[:, i] = _assignment(rng, size[0], counts[i], balance)

    return data

def normal(
    size: tuple[int, int] = (1000, 2),
    meanInterval: tuple[float, float] = (1.0, 5.0),
    sigmaInterval: tuple[float, float] = (0.1, 3.0),
    skewInterval: tuple[float, float] = (0.0, 0.0),
    modesInterval: tuple[int, int] = (1, 1),
    balance: float = 0.5,
    seed: int | None = None
) -> np.ndarray:
    """
    Generate random data from independent normal distributions, each with a 
    random location, scale and optionally skewness and modes

    Args:
        size: The number of items and features to generate
        meanInterval: The range each mode's mean is drawn from, inclusive,
            its location when it is skewed
        sigmaInterval: The range each mode's scale is drawn from, inclusive
        skewInterval: The range each mode's skewness is drawn from,
            inclusive
        modesInterval: The range each feature's number of modes is drawn
            from, inclusive
        balance: How balanced the modes' shares of items are, 0 to 1
        seed: The seed for random generation to reproduce the data
    """
    rng = np.random.default_rng(seed)
    modes = rng.integers(
        modesInterval[0], modesInterval[1], size=size[1], endpoint=True
    )

    data = np.zeros(size)
    for i in range(size[1]):
        mu = _uniform(rng, meanInterval, modes[i])
        sigma = _uniform(rng, sigmaInterval, modes[i])
        a = _uniform(rng, skewInterval, modes[i])

        mode = _assignment(rng, size[0], modes[i], balance)
        data[:, i] = skewnorm.rvs(
            a[mode], 
            loc = mu[mode], 
            scale = sigma[mode], 
            random_state = rng
        )

    return data

def blobs(
    size: tuple[int, int] = (1000, 2),
    centerInterval: tuple[float, float] = (1.0, 5.0),
    clustersInterval: tuple[int, int] = (6, 6),
    sigmaInterval: tuple[float, float] = (0.1, 3.0),
    balance: float = 1.0,
    seed: int | None = None
) -> np.ndarray:
    """
    Generate random data of a random number of normally distributed clusters

    Args:
        size: The number of items and features to generate
        centerInterval: The range each cluster's center is drawn from, inclusive
        clustersInterval: The range the number of clusters is drawn from,
            inclusive
        sigmaInterval: The range each cluster's standard deviation is
            drawn from, inclusive
        balance: How balanced the clusters' shares of items are, 0 to 1
        seed: The seed for random generation to reproduce the data
    """
    rng = np.random.default_rng(seed)
    clusters = rng.integers(
        clustersInterval[0], clustersInterval[1], endpoint=True
    )
    centers = _uniform(rng, centerInterval, (clusters, size[1]))
    sigma = _uniform(rng, sigmaInterval, clusters)

    labels = _assignment(rng, size[0], clusters, balance)
    return centers[labels] + sigma[labels, None] * rng.standard_normal(size)

def _assignment(
    rng: np.random.Generator,
    items: int,
    groups: int,
    balance: float
) -> np.ndarray:
    """
    Assign each item to a group, in a random order, giving each group at least
    one item and a share of the rest that blends equal and uniformly random
    shares by the balance

    Args:
        rng: The random generator of the dataset being generated
        items: The number of items to assign
        groups: The number of groups to assign them to
        balance: How balanced the shares of the groups are, from 0 for
            uniformly random to 1 for equal

    Raises:
        ValueError: If the balance is not between 0 and 1
    """
    if not 0 <= balance <= 1:
        raise ValueError("Balance is not between 0 and 1")

    equal = np.full(groups, 1 / groups)
    uniformShares = rng.dirichlet(np.ones(groups)) # uniformly random shares
    shares = balance * equal + (1 - balance) * uniformShares

    # round the shares of the spare items to whole counts, giving the items
    # left over by rounding down to the groups with the largest remainders
    spare = items - groups
    exact = shares * spare
    counts = np.floor(exact).astype(int)
    leftover = spare - counts.sum()
    counts[np.argsort(counts - exact, kind="stable")[:leftover]] += 1

    labels = np.repeat(np.arange(groups), counts + 1)
    return rng.permutation(labels)

def _uniform(
    rng: np.random.Generator,
    interval: tuple[float, float],
    size: int | tuple[int, ...]
) -> np.ndarray:
    """
    Draw uniformly random values from an interval that includes both bounds,
    unlike rng.uniform, which never returns the upper bound

    Args:
        rng: The random generator of the dataset being generated
        interval: The range of the values, inclusive
        size: The number or shape of values to draw
    """
    low, high = interval
    unit = rng.random(size) / np.nextafter(1.0, 0.0) # from 0 to 1 inclusive
    return np.clip(low + (high - low) * unit, low, high) # within the bounds
