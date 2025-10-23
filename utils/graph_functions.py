import numpy as np
from scipy.linalg import expm

# Graph functions
import networkx as nx

# Utilities
import time as time
from utils.utils import display_time

# -------------------------------------
# Heat kernel
# -------------------------------------
def heat_kernel(G, t):
    """
    Heat kernel
    """
    L = nx.laplacian_matrix(G).todense()
    H = expm(-t * L)

    return H


# -------------------------------------
# Shortest path distance matrices
# -------------------------------------
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


# -------------------------------------
# Graph generators
# -------------------------------------
# Create several graphs with a graph generator and put them in a cycle
# by joining their 0-th vertices
def cycle_of_generators_variable(num_groups, generator, arglist):
    n_nodes = 0
    t_prev = 0
    for idx in range(num_groups):
        # Create graph and update number of nodes
        G_i = generator(*arglist[idx])
        n_i = G_i.number_of_nodes()

        # Store G_i if we have a single graph
        if idx == 0:
            G = G_i
            n_nodes += n_i
            continue

        # Otherwise, we join G_i and G
        mapping = {t: t + n_nodes for t in range(n_i)}
        G_i = nx.relabel_nodes(G_i, mapping)
        G = nx.compose(G, G_i)

        # Add an edge between G and G_i
        G.add_edge(t_prev, n_nodes)

        # Update markers
        t_prev = n_nodes
        n_nodes += n_i

    # Close the cycle
    G.add_edge(t_prev, 0)

    return G


# Use the same set of arguments for all generators
def cycle_of_generators(num_groups, generator, *args):
    arglist = [args] * num_groups
    return cycle_of_generators_variable(num_groups, generator, arglist)


# Remove the generator argument from cycle_of_generators
def cycle_of_generators_fun(generator):
    def fun(num_groups, *args):
        return cycle_of_generators(num_groups, generator, *args)

    return fun


# Creates a nested cycle
# arg[i] is the length of the cycles at level i
def nested_cycles(*args):
    num_args = len(args)

    generator = nx.complete_graph
    for _ in range(1, num_args):
        generator = cycle_of_generators_fun(generator)

    return generator(*args)


# -- Unused generators --
# Creates a set of graphs with generator(arg) for arg in arglist
# then joins all of them at their 0-th vertex
def wedge_of_generators(generator, arglist):
    n_graphs = len(arglist)
    n_nodes = 0

    for idx in range(n_graphs):
        # Create graph and update number of nodes
        G_i = generator(*arglist[idx])
        n_i = G_i.number_of_nodes()

        # Store G_i if we have a single graph
        if idx == 0:
            G = G_i
            n_nodes += n_i
            continue

        # Otherwise, we join G_i and G
        mapping = {t: t + n_nodes - 1 for t in range(1, n_i)}
        G_i = nx.relabel_nodes(G_i, mapping, copy=False)
        G = nx.compose(G, G_i)

        # Update number of nodes
        n_nodes += n_i

    return G


# Remove the generator argument from cycle_of_generators
def wedge_of_generators_fun(generator):
    def fun(arglist):
        return wedge_of_generators(generator, arglist)

    return fun


# -------------------------------------
# Plotting nested cycles
# -------------------------------------
# Arrange nodes of a cycle around a circle
def pos_cycle(m, R=1, c=[0, 0], t0=np.pi / 2):
    # Draw a complete graph around a circle
    tt = t0 + np.linspace(0, 2 * np.pi, m + 1)
    tt = np.delete(tt, -1)

    xx = R * np.cos(tt)
    yy = R * np.sin(tt)

    pos = np.column_stack([xx, yy])
    return pos


# Simple rotation matrix
def rotation_matrix(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


# Arrange a list with n positions pos_list around a cycle of length n.
def pos_cycle_of_graphs(n, pos_list, R=4, t0=np.pi):
    # pos_list is the default position of a single subgraph
    # We incorporate it into the outer cycle in this function

    # Angles in the outer cycle
    tt = np.linspace(0, 2 * np.pi, n + 1)
    tt = np.delete(tt, -1)

    # Position of the outer cycle
    pos_out = pos_cycle(n, R=R)

    # Place each inner subgraph at a vertex of the outer cycle
    pos_all = np.zeros((0, 2))
    for idx in range(n):
        t = tt[idx]
        M = rotation_matrix(t + t0)
        dR = pos_out[idx]

        # I need to rotate and translate the position of the inner cycle
        pos_0 = pos_list[idx]
        pos_all = np.concatenate([pos_all, dR + pos_0 @ M.T], axis=0)

    return pos_all


# Calls cycle_of_graphs assuming all subgraphs are the same
def pos_cycle_uniform(n, pos_0, R=4, t0=np.pi):
    return pos_cycle_of_graphs(n, [pos_0] * n, R=R, t0=t0)


# Positions for a nested cycle of graphs
def pos_nested_cycles(*args, scale=4):
    num_args = len(args)

    # It's more convenient for the user to pass arguments from
    # outer cycle to inner cycle. However, for us, it's easier to
    # consume the arguments from inner to outer
    args = args[::-1]

    # Recursively compute positions of nested cycles
    pos_new = pos_cycle(args[0])
    for idx in range(1, num_args):
        pos_old = pos_new
        pos_new = pos_cycle_uniform(args[idx], pos_old, R=scale**idx)

    return pos_new
