import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import squareform, pdist
from scipy.special import binom

# Utilities
import time as time
from datetime import timedelta

from warnings import warn

# Format the output of time
def display_time(seconds):
    if seconds == np.round(seconds):
        seconds += 0.001
    return str(timedelta(seconds=seconds))[:-4]


# Removes rounding errors by setting the diagonal to 0 and symmetrizing matrix
def fix_dm(A):
    A -= np.diag(np.diag(A))
    A = np.maximum(A, A.T)

    return A


# Note: map0 must be a bijection for this code to work
def invert_dict(map0):
    map_inv = {map0[t]: t for t in map0.keys()}
    return map_inv


def stepwise_pdist(X_array):
    nSteps = X_array.shape[0]
    N = X_array.shape[1]

    dists = np.zeros((nSteps, N, N))
    for idt in range(nSteps):
        X = X_array[idt]
        dists[idt, :, :] = squareform(pdist(X))

    return dists


# -------------------------------------
# Functions to handle squareform
# -------------------------------------
# Given an n-by-n symmetric matrix M, let SM = squareform(M).
# The function below satisfies SM[idx] = M[i,j]
# for idx = squareform_index(i,j,n)
def index_to_squareform(i, j, n):
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
    def sq_idx(i, j):
        return index_to_squareform(i, j, N)

    return sq_idx


def sq_to_idx_fun(N):
    def sq_to_idx(idx):
        return squareform_to_index(idx, N)

    return sq_to_idx


# Convert pairs of indices in I to squareform
def pairs_to_sqform(I, N):
    n = len(I)

    # (i,j) to squareform index
    sq_idx = sq_idx_fun(N)

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            idx = sq_idx(I[i], I[j])
            pairs.append(idx)

    return pairs


# Convert pairs of indices within I and within J to squareform
def select_pairs(I, J):
    n = len(I)
    m = len(J)
    N = n + m

    # (i,j) to squareform index
    sq_idx = sq_idx_fun(N)

    pairs = pairs_to_sqform(I, N)
    pairs.extend(pairs_to_sqform(J, N))

    return pairs


# -------------------------------------
# Sampling from probability simplex
# -------------------------------------
def sampling_simplex_log(exp_0, d, n_samples=None):
    """
    Logscale sampling from the probability simplex

    NOTE: exp_0 is different from np.min(np.log10(X)).
    Currently, I don't know how to predict the latter number.
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
    Uniform sample from the probability simplex
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
