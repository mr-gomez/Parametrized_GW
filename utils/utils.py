import numpy as np
from scipy.spatial.distance import squareform
from scipy.special import binom

# Graph functions
import networkx as nx

# Utilities
from time import time
from datetime import timedelta

from warnings import warn

# Compute a matrix with the shortest path between
# all elements of v_start and all elements in v_end
def shortest_paths_subsets(G, v_start, v_end):
    n = len(v_start)
    m = len(v_end)
    dm = np.zeros((n, m))

    # Find all shortest paths from each vertex in v_start
    for i, v in enumerate(v_start):
        for j, w in enumerate(v_end):
            dm[i, j] = nx.shortest_path_length(G, source=v, target=w, weight="weight")

    return dm


# Compute shortest paths with NetworkX
# This is a wrapper around NetworkX methods. It takes their output and
# constructs a distance matrix.
# NOTE: We use dijkstra (the default below) instead of floyd_warshall
# because our graphs are sparse. It turns out that there are about 3
# edges per node.
# NOTE: The progress variable is used to show how long it takes to complete
#       a certain percentage of the overall calculation. Set to None if you
#       don't want to display progress
def all_pairs_shortest_path_weighted(G, weight="weight", progress=0.1):
    nNodes = G.number_of_nodes()
    dm = np.zeros((nNodes, nNodes))

    # Compute distances from a single point to everything else
    time_1 = time()
    flag_prev = 0
    for i in range(nNodes):
        dist_dict = dict(nx.shortest_path_length(G, source=i, weight=weight))

        # Write into the distance matrix
        for j in range(i + 1, nNodes):
            dm[i, j] = dist_dict[j]
            dm[j, i] = dist_dict[j]

        if progress is not None:
            flag_new = np.floor(i / (nNodes * progress))
            if flag_new > flag_prev:
                time_2 = time()

                percent = np.floor(100 * i / nNodes)
                dt = display_time(time_2 - time_1)
                print(f"{percent}%: {dt}")
                time_1 = time_2
                flag_prev = flag_new
    return dm


def display_time(seconds):
    if seconds == np.round(seconds):
        seconds += 0.001
    return str(timedelta(seconds=seconds))[:-4]


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
