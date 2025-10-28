import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import squareform, pdist
from scipy.special import binom

# Utilities
import time as time
from datetime import timedelta

from warnings import warn


def display_time(seconds):
    """
    Format a time duration in seconds into a human-readable string.

    Parameters
    ----------
    seconds : float
        Time in seconds.

    Returns
    -------
    str
        A string with time formatted as ``HH:MM:SS.ss``.
    """
    if seconds == np.round(seconds):
        seconds += 0.001
    return str(timedelta(seconds=seconds))[:-4]


def fix_dm(A):
    """
    Remove rounding errors in a distance matrix by zeroing its diagonal
    and symmetrizing it.

    Parameters
    ----------
    A : ndarray of shape (n, n)
        Input distance matrix.

    Returns
    -------
    ndarray of shape (n, n)
        Symmetrized distance matrix with a zero diagonal.
    """
    A -= np.diag(np.diag(A))
    A = np.maximum(A, A.T)

    return A


def invert_dict(map0):
    """
    Invert an injective dictionary.

    Takes a dictionary {key:value} where the key to value mapping is
    one-to-one and returns the dictionary {value:key}

    Parameters
    ----------
    map0 : dict
        Dictionary to invert.

    Returns
    -------
    dict
        Dictionary where keys and values are swapped.
    """
    map_inv = {map0[t]: t for t in map0.keys()}
    return map_inv


def stepwise_pdist(X_array):
    """
    Compute pairwise distances for each page of a 3D array.

    Parameters
    ----------
    X_array : ndarray of shape (n_steps, n_points, n_features)
        Array containing multiple sets of points.

    Returns
    -------
    ndarray of shape (n_steps, n_points, n_points)
        Array of pairwise distance matrices of each X_array[i,:,:].
    """
    n_steps = X_array.shape[0]
    N = X_array.shape[1]

    dists = np.zeros((n_steps, N, N))
    for idt in range(n_steps):
        X = X_array[idt]
        dists[idt, :, :] = squareform(pdist(X))

    return dists


# -------------------------------------
# Functions to handle squareform
# -------------------------------------
def index_to_squareform(i, j, n):
    """
    Convert a pair of matrix indices to its condensed (squareform) index.

    Parameters
    ----------
    i : int
        Row index in the full matrix.
    j : int
        Column index in the full matrix.
    n : int
        Size of the original n x n matrix.

    Returns
    -------
    int
        Index in the condensed vector that corresponds to M[i,j].

    Warns
    -----
    RuntimeWarning
        If ``i == j``, since the diagonal is not included in the condensed vector.
    """
    if i > j:
        temp = i
        i = j
        j = temp
    if i == j:
        warn(
            "You're trying to access a diagonal element from a condensed distance matrix",
            RuntimeWarning,
        )

    return int(binom(n, 2) - binom(n - i, 2) + (j - i - 1))


def squareform_to_index(idx, n):
    """
    Convert a condensed (squareform) index back to its pair of matrix indices.

    Parameters
    ----------
    idx : int
        Index in the condensed (squareform) distance vector.
    n : int
        Size of the original ``n x n`` matrix.

    Returns
    -------
    (int, int)
        Tuple of indices ``(i, j)`` such that ``M[i,j]`` is the element at position ``idx``.

    Raises
    ------
    IndexError
        If the index is out of range for the given size ``n``.
    """
    if binom(n, 2) <= idx:
        raise IndexError(
            f"squareform index out of range (idx={idx}, max={int(binom(n,2))-1})"
        )

    i = 0
    S = 0
    while S <= idx:
        i += 1
        S = binom(n, 2) - binom(n - i, 2)

    # Reverse last iteration
    i -= 1
    S = int(binom(n, 2) - binom(n - i, 2))

    j = i + 1 + (idx - S)

    return i, j


# Wrappers for the above functions
def sq_idx_fun(N):
    """
    Create a function that converts matrix indices to squareform indices for a fixed matrix.

    Parameters
    ----------
    N : int
        Size of the square matrix.

    Returns
    -------
    callable
        Function ``sq_idx`` such that
        ``sq_idx(i,j) == index_to_squareform(i, j, N)``.
    """

    def sq_idx(i, j):
        return index_to_squareform(i, j, N)

    return sq_idx


def sq_to_idx_fun(N):
    """
    Create a function that converts squareform indices back to matrix indices.

    Parameters
    ----------
    N : int
        Dimension of the square matrix.

    Returns
    -------
    callable
        Function ``sq_to_idx`` such that
        ``sq_to_idx(idx) == squareform_to_index(idx, N)``.
    """

    def sq_to_idx(idx):
        return squareform_to_index(idx, N)

    return sq_to_idx


def pairs_to_sqform(N, I):
    """
    Convert all pairs of indices from a list into squareform indices.

    We assume that ``I`` is a set of indices of a square matrix ``M``. If we
    stored ``V = squareform(M)``, this function constructs a list of indices
    ``pairs`` so that we can access ``M[i,j]`` for every ``i`` and ``j`` in
    ``I`` via ``V[i2]`` for some ``i2`` in ``pairs``.

    Parameters
    ----------
    N : int
        Size of a square matrix condensed with squareform.
    I : array-like of int
        Subset of range(N). Indices to use for forming pairs.

    Returns
    -------
    list of int
        List of squareform indices corresponding to all pairs of indices in I.
    """
    n = len(I)

    # (i,j) to squareform index
    sq_idx = sq_idx_fun(N)

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            idx = sq_idx(I[i], I[j])
            pairs.append(idx)

    return pairs


def select_pairs(I, J):
    """
    Select pairs of indices from two sets (I and J) and convert to squareform.

    We assume that ``I`` and ``J`` are sets of indices of a square matrix
    ``M`` of size ``len(I)+len(J)``. If we stored ``V = squareform(M)``, this
    function constructs a a list ``pairs`` so that we can access ``M[i,j]``
    for every ``i`` in ``I`` and ``j`` in ``J`` via ``V[i2]`` for some ``i2``
    in ``pairs``.

    Parameters
    ----------
    I : array-like of int
        First set of indices.
    J : array-like of int
        Second set of indices.

    Returns
    -------
    list of int
        List of squareform indices representing pairs within I and within J.
    """
    n = len(I)
    m = len(J)
    N = n + m

    # (i,j) to squareform index
    sq_idx = sq_idx_fun(N)

    pairs = pairs_to_sqform(N, I)
    pairs.extend(pairs_to_sqform(N, J))

    return pairs


# -------------------------------------
# Sampling from probability simplex
# -------------------------------------
# NOTE: exp_0 is different from np.min(np.log10(X)).
# Currently, I don't know how to predict the latter number.
def sampling_simplex_log(exp_0, d, n_samples=None):
    """
    Generate samples from the probability simplex using logarithmic scaling.

    This function samples points in log-space to explore extreme regions of
    the probability simplex.

    Parameters
    ----------
    exp_0 : int
        The ratio between the largest and smallest entries of each sample is
        at most 10**(exp_0-1).
    d : int
        Dimension of the simplex.
    n_samples : int or None, optional
        Number of points in the initial np.logspace step.
        The final number of samples is O(n_samples**d).

    Returns
    -------
    ndarray of shape (N, d)
        Array of points sampled from the d-simplex.
    """
    # Uniformly sample from logspace
    if n_samples is None:
        E0 = np.logspace(1, exp_0, exp_0)
    else:
        E0 = np.logspace(1, exp_0, n_samples)

    # Create a d-dim grid
    E_mesh = np.meshgrid(*[E0] * d)

    # Change order of first 2 elements
    # (somehow the order is not 100% right)
    temp = E_mesh[0]
    E_mesh[0] = E_mesh[1]
    E_mesh[1] = temp

    # Reshape arrays into columns
    E_cols = []
    for M in E_mesh:
        M = M.reshape((-1, 1))
        E_cols.append(M)

    # Create a single array
    E = np.concatenate(E_cols, axis=1)

    # Normalize the sum of each row
    S = np.sum(E, axis=1)
    X = E / np.repeat(S[:, np.newaxis], d, axis=1)

    # Remove duplicate points
    X = np.unique(X, axis=0)

    return X


def sample_simplex_unif(N, d, rng=None):
    """
    Uniformly sample points from the probability simplex.

    Parameters
    ----------
    N : int
        Number of samples to generate.
    d : int
        Dimension of the simplex.
    rng : int or numpy.random.Generator, optional
        Random number generator or seed for reproducibility.

    Returns
    -------
    ndarray of shape (N, d)
        Samples uniformly distributed over the d-dimensional simplex.
    """
    if rng is None:
        rng = np.random.default_rng()
    elif isinstance(rng, int):
        rng = np.random.default_rng(seed=rng)

    U = rng.uniform(size=(N, d))
    E = -np.log10(U)
    S = np.sum(E, axis=1)
    nus = E / np.repeat(S[:, np.newaxis], d, axis=1)

    return nus


# -------------------------------------
# Plotting a collection of couplings
# -------------------------------------
def show_couplings(Ts, figsize=(10, 10)):
    """
    Plot a collection of couplings as a grid of heatmaps.

    Parameters
    ----------
    Ts : ndarray of shape (N, N, m, n)
        Array containing coupling matrices between N distributions.
    figsize : tuple of float, optional
        Size of the figure, default is (10, 10).

    Returns
    -------
    (matplotlib.figure.Figure, ndarray of Axes)
        The created figure and its axes grid.
    """
    N = Ts.shape[0]
    fig, axes = plt.subplots(N, N, figsize=figsize)
    fig.suptitle("Optimal couplings")

    for i in range(N):
        # plt.delaxes(axes[i,i])

        for j in range(i, N):
            axes[i, j].imshow(Ts[i, j], vmin=0, vmax=1e-5)

            if i != j:
                plt.delaxes(axes[j, i])

    return fig, axes
