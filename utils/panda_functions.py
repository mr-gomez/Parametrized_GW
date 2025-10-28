import numpy as np
import matplotlib.pyplot as plt

# Graph functions
import networkx as nx

# Utilities
from utils.utils import fix_dm
from pathlib import Path

# -------------------------------------
# Panda generators
# -------------------------------------
def paste_to_vertices(nh, ear_dist=1, paste_edge=False):
    """
    Determine indices of vertices used for attaching ears to a panda head.

    Parameters
    ----------
    nh : int
        Number of vertices in the head.
    ear_dist : int, optional
        Distance between the attachment points of the ears (default is 1).
    paste_edge : bool, optional
        If True, attach ears along an edge instead of a vertex.
        (default is False).

    Returns
    -------
    tuple of int
        Indices ``(i1_1, i1_2, i2_1, i2_2)`` of the vertices where ears are
        pasted. If we are pasting over vertices, ``i1_1==i1_2`` and
        ``i2_1==i2_2``.
    """
    if not paste_edge:
        return nh - 1 - ear_dist, nh - 1 - ear_dist, nh - 1, nh - 1
    else:
        return nh - 3 - ear_dist, nh - 2 - ear_dist, nh - 2, nh - 1


def ear_maps(nh, ne1, ne2, ear_dist=1, paste_edge=False):
    """
    Construct node mapping dictionaries for attaching ears to the panda head.

    Returns two dictionaries ``map1`` and ``map2``, one for each ear. Each
    ``map`` is a function from the labels of an ear to the labels of a panda
    graph.

    Parameters
    ----------
    nh : int
        Number of vertices in the head.
    ne1 : int
        Number of vertices in the first ear.
    ne2 : int
        Number of vertices in the second ear.
    ear_dist : int, optional
        Distance between the attachment points of the ears (default is 1).
    paste_edge : bool, optional
        If True, attach ears along an edge instead of a vertex.
        (default is False).

    Returns
    -------
    map1, map2 : dict
        Dictionaries mapping ear node indices to indices in
        a panda graph.
    """
    i1_1, i1_2, i2_1, i2_2 = paste_to_vertices(
        nh, ear_dist=ear_dist, paste_edge=paste_edge
    )

    if not paste_edge:
        map1 = {t: t + nh - 1 for t in range(ne1)}
        map2 = {t: t + nh + ne1 - 2 for t in range(ne2)}

        map1[0] = i1_1
        map2[0] = i2_1
    else:
        map1 = {t: t + nh - 2 for t in range(ne1)}
        map2 = {t: t + nh + ne1 - 4 for t in range(ne2)}

        map1[0] = i1_1
        map1[1] = i1_2
        map2[0] = i2_1
        map2[1] = i2_2

    return map1, map2


def create_panda(
    nh,
    ne1,
    ne2,
    ear_dist=1,
    push_ears=False,
    paste_edge=False,
    add_neighbors=False,
    rng=None,
    seed=None,
    std=0,
):
    """
    Create a panda graph composed of a head and two ears along with its distance matrices.

    A panda graph is a cycle graph (the head) with two smaller cycles (the
    ears) attached. This function constructs such a graph and computes its
    shortest path distance matrix with small Gaussian noise. We also construct
    a pm-net with three pseudo-distance matrices by zeroing out the distances
    outside of the head and in each ear.

    Note: We symmetrize and reset the diagonal to 0 after adding noise to the
          distance matrix.

    Parameters
    ----------
    nh : int
        Number of vertices in the head.
    ne1 : int
        Number of vertices in the first ear.
    ne2 : int
        Number of vertices in the second ear.
    ear_dist : int, optional
        Distance between the attachment points of the ears (default is 1).
    push_ears : bool, optional
        If True, the indexing starts with the ears. Otherwise, the indexing starts with the head.
    paste_edge : bool, optional
        If True, attach ears along an edge instead of a vertex.
        (default is False).
    add_neighbors : bool, optional
        If True, keeps neighbors of the wedge vertex when computing the pseudo-metrics.
    rng : int or numpy.random.Generator, optional
        Random number generator or seed for reproducibility.
    std : float, optional
        Standard deviation of Gaussian noise added to the distance matrix.

    Returns
    -------
    N : int
        Total number of vertices in the panda graph.
    G : networkx.Graph
        The panda graph.
    dm : numpy.ndarray
        Shortest path distance matrix with noise.
    lC : list of numpy.ndarray
        List of restrictions of the distance matrix to the head and each ear.
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(seed)
    elif rng is None:
        rng = np.random.default_rng()

    # Head and ears
    H = nx.cycle_graph(nh)
    E1 = nx.cycle_graph(ne1)
    E2 = nx.cycle_graph(ne2)

    # Rename vertices
    map1, map2 = ear_maps(nh, ne1, ne2, ear_dist=ear_dist, paste_edge=paste_edge)
    E1 = nx.relabel_nodes(E1, map1)
    E2 = nx.relabel_nodes(E2, map2)

    # Paste ears
    G = nx.compose(H, E1)
    G = nx.compose(G, E2)
    N = G.number_of_nodes()

    # Compute distance matrix
    dm = nx.floyd_warshall_numpy(G)

    # Reorder the panda?
    if not push_ears:
        map = {t: t for t in range(N)}
    else:
        # Reorder vertices
        map = {t: (t - nh) % N for t in range(N)}
        G = nx.relabel_nodes(G, map)

        # Reorder distance matrix
        I_inv = [(t + nh) % N for t in range(N)]
        dm = dm[I_inv, :]
        dm = dm[:, I_inv]

    # Decompose dm into blocks
    I1 = [map[map1[t]] for t in range(ne1)]
    I2 = [map[map2[t]] for t in range(ne2)]

    # The ears are attached at their 0-th vertex
    # Remove the vertices that are not attached
    if not paste_edge:
        I_ears = I1[1:] + I2[1:]
    else:
        I_ears = I1[2:] + I2[2:]
    I0 = list(np.setdiff1d(range(N), I_ears))

    # Extract the block matrices
    lC = []
    for I in [I0, I1, I2]:
        # Print add the neighbors of the wedge point
        if add_neighbors and I != I0:
            for neighbor in G.neighbors(I[0]):
                if neighbor not in I:
                    I.append(neighbor)

        # Extract the block
        I_comp = np.setdiff1d(range(N), I)
        dm2 = dm.copy()
        dm2[I_comp, :] = 0
        dm2[:, I_comp] = 0

        # Add noise
        dm2 += std * rng.normal(size=(N, N))

        dm2 = fix_dm(dm2)
        lC.append(dm2)

    # Store the data
    return N, G, dm, lC


# -------------------------------------
# Finding position of panda vertices
# -------------------------------------
def head_position(nh, R=1, ear_dist=1, i1_end=None, i2_start=None):
    """
    Arranges the nodes of the head of a panda in a circle.

    Parameters
    ----------
    nh : int
        Number of vertices in the head.
    R : float, optional
        Radius of the head (default is 1).
    ear_dist : int, optional
        Distance between the attachment points of the ears (default is 1).
    i1_end : int, optional
        Index of the last vertex of the first ear.
    i2_start : int, optional
        Index of the first vertex of the second ear.

    Returns
    -------
    X_head : numpy.ndarray of shape ``(nh, 2)``
        Coordinates of the head vertices.
    tt : numpy.ndarray
        Angular positions of vertices in radians.
    """
    if i1_end is None or i2_start is None:
        i1_1, i1_2, i2_1, i2_2 = paste_to_vertices(
            nh, ear_dist=ear_dist, paste_edge=False
        )
        if i1_end is None:
            i1_end = i1_2
        if i2_start is None:
            i2_start = i2_1

    # Angles for the circle
    tt = np.linspace(0, 2 * np.pi, nh, endpoint=False)

    # Rotate so that the wedge points are at the top of the panda
    # and the ears are symmetric against the y-axis
    t_mid = (tt[i1_end] + tt[i2_start]) / 2
    tt = tt + np.pi / 2 - t_mid

    # Create points in the head
    tt = tt[:, np.newaxis]
    X_head = R * np.concatenate((np.cos(tt), np.sin(tt)), axis=1)

    return X_head, tt


def ear_position(ne, X_head, tt, i1, i2=None, R=1, r=None, paste_edge=False):
    """
    Arranges the nodes of an ear of a panda in a circle.

    Parameters
    ----------
    ne : int
        Number of vertices in the ear.
    X_head : numpy.ndarray
        Coordinates of the head vertices.
    tt : numpy.ndarray
        Angular positions of head vertices.
    i1 : int
        Index of the vertex of the head where we attach the ear.
    i2 : int, optional
        Index of the second attachment vertex (if ear is pasted over an edge).
    R : float, optional
        Radius of the head (default is 1).
    r : float, optional
        Radius of the ear; if None, computed proportionally to head size.
    paste_edge : bool, optional
        If True, attach ears along an edge instead of a vertex.
        (default is False).

    Returns
    -------
    X_ear : numpy.ndarray of shape ``(ne, 2)``
        Coordinates of the ear vertices.
    """
    # Compute inner radius if not given
    if r is None:
        nh = X_head.shape[0]
        r = ne / nh

    # Compute the index of the second wedge vertex if not given
    if i2 is None and not paste_edge:
        # Only one wedge vertex if not pasting over an edge
        i2 = i1
    elif i2 is None and paste_edge:
        # Next vertex if pasting over an edge
        i2 = (i1 + 1) % nh

    # Wedge point of the ear
    v0 = (X_head[i1, :] + X_head[i2, :]) / 2
    v0 = v0 / np.linalg.norm(v0)

    # Create a circle so that the wedge points in the ear
    # and the head coincide
    tt1 = -np.linspace(0, 2 * np.pi, ne, endpoint=False)
    tt1 = tt1 + (tt[i1, 0] + tt[i2, 0]) / 2 - np.pi
    if paste_edge:
        # Realign
        tt1 = tt1 + 2 * np.pi / (2 * ne)
    tt1 = tt1[:, np.newaxis]

    X_ear = np.concatenate((np.cos(tt1), np.sin(tt1)), axis=1)
    X_ear = (R + r) * v0 + r * X_ear

    return X_ear


def panda_position(
    nh, ne1, ne2, ear_dist=1, R=1, r1=None, r2=None, paste_edge=False, push_ears=False
):
    """
    Compute coordinates of a full panda graph.

    Parameters
    ----------
    nh : int
        Number of vertices in the head.
    ne1 : int
        Number of vertices in the first ear.
    ne2 : int
        Number of vertices in the second ear.
    ear_dist : int, optional
        Distance between the attachment points of the ears (default is 1).
    R : float, optional
        Radius of the head (default is 1).
    r1 : float, optional
        Radius of the first ear.
    r2 : float, optional
        Radius of the second ear.
    paste_edge : bool, optional
        If True, attach ears along an edge instead of a vertex.
        (default is False).
    push_ears : bool, optional
        If True, reorder coordinate indices to move ears first.

    Returns
    -------
    X_all : numpy.ndarray of shape ``(nh+ne1+ne2, 2)``
        Coordinates for all nodes of a panda graph.
    """
    # Ear-head mappings and indices of wedge points in the head
    map1, map2 = ear_maps(nh, ne1, ne2, ear_dist=ear_dist, paste_edge=paste_edge)
    i1_1, i1_2, i2_1, i2_2 = paste_to_vertices(
        nh, ear_dist=ear_dist, paste_edge=paste_edge
    )

    # Create circles for head
    X_head, tt = head_position(nh, R=R, ear_dist=ear_dist, i1_end=i1_2, i2_start=i2_1)

    # Create circles for the ears
    X_ear1 = ear_position(
        ne1, X_head, tt, i1=i1_1, i2=i1_2, R=R, r=r1, paste_edge=paste_edge
    )
    X_ear2 = ear_position(
        ne2, X_head, tt, i1=i2_1, i2=i2_2, R=R, r=r2, paste_edge=paste_edge
    )

    # Concatenate all coordinates according to the maps
    if not paste_edge:
        N = nh + ne1 + ne2 - 2
    else:
        N = nh + ne1 + ne2 - 4

    X_all = np.zeros((N, 2))
    X_all[:nh, :] = X_head

    for idx_0 in range(ne1):
        idx = map1[idx_0]
        if idx >= nh:
            X_all[idx, :] = X_ear1[idx_0, :]

    for idx_0 in range(ne2):
        idx = map2[idx_0]
        if idx >= nh:
            X_all[idx, :] = X_ear2[idx_0, :]

    if push_ears:
        # map = {t: (t - nh) % N for t in range(N)}
        X_all = np.roll(X_all, -nh, 0)
    return X_all


# -------------------------------------
# Graphing pandas
# -------------------------------------
def display_ms_pandas(Pandas, Pandas_pos, dm_pandas, lCs_pandas):
    """
    Display pandas, their distance matrices, and pm-nets.

    Parameters
    ----------
    Pandas : list of networkx.Graph
        List of panda graphs.
    Pandas_pos : list of numpy.ndarray
        Coordinates for vertices of the pandas.
    dm_pandas : list of numpy.ndarray
        Distance matrices for each panda.
    lCs_pandas : list of list of numpy.ndarray
        pm-nets for each panda.

    Returns
    -------
    figs : list of matplotlib.figure.Figure
        Figures for the plots.
    axes : list of list of matplotlib.axes.Axes
        Axes corresponding to each subplot.
    """
    nPandas = len(Pandas)
    nSteps = len(lCs_pandas[0])

    # Panda graphs, distance matrices, and multiscale distance matrices
    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))
    fig2, axes2 = plt.subplots(1, nPandas)
    fig3, axes3 = plt.subplots(nPandas, nSteps)

    for idx in range(nPandas):
        # Panda graphs and distance matrices
        nx.draw(Pandas[idx], Pandas_pos[idx], with_labels=True, ax=axes1[idx])
        axes2[idx].imshow(dm_pandas[idx])

        # Multiscale networks
        for idt in range(nSteps):
            axes3[idx, idt].imshow(lCs_pandas[idx][idt])

    return [fig1, fig2, fig3], [axes1, axes2, axes3]


def save_pandas(Pandas, Pandas_pos, dm_pandas, lCs_pandas, folder=""):
    """
    Save visualizations of pandas, distance matrices, and pm-nets.

    Parameters
    ----------
    Pandas : list of networkx.Graph
        List of panda graphs.
    Pandas_pos : list of numpy.ndarray
        Coordinates for vertices of the pandas.
    dm_pandas : list of numpy.ndarray
        Distance matrices for each panda.
    lCs_pandas : list of list of numpy.ndarray
        pm-nets for each panda.
    folder : str or Path, optional
        Directory to save the resulting figures (default is current folder).
    """
    nPandas = len(Pandas)
    nSteps = len(lCs_pandas[0])

    # Panda graphs, distance matrices, and multiscale distance matrices
    for idx in range(nPandas):
        # Panda graph
        fig1, ax1 = plt.subplots(1, 1, figsize=(5, 5))
        nx.draw(Pandas[idx], Pandas_pos[idx], with_labels=True, ax=ax1)
        ax1.set_aspect("equal")

        # Distance matrix
        vmin = 0
        vmax = np.max(dm_pandas[idx])
        fig2, ax2 = plt.subplots(1, 1, figsize=(3, 3))
        im2 = ax2.imshow(dm_pandas[idx], vmin=vmin, vmax=vmax)
        ax2.set_aspect("equal")
        fig2.colorbar(im2, ax=ax2, shrink=0.8)

        # Multiscale networks
        fig3, ax3 = plt.subplots(1, nSteps, figsize=(10, 10 * nSteps))
        for idt in range(nSteps):
            ax3[idt].imshow(lCs_pandas[idx][idt], vmin=vmin, vmax=vmax)

        # --------------
        # Customization
        # --------------
        # Remove whitespace from panda graphs
        ax1.margins(0)

        # Save
        fig1.savefig(
            Path(folder, f"Panda_{idx}.pdf"), bbox_inches="tight", pad_inches=0
        )
        fig2.savefig(
            Path(folder, f"Panda_dm_{idx}.pdf"), bbox_inches="tight", pad_inches=0
        )
        fig3.savefig(
            Path(folder, f"Panda_ms_{idx}.pdf"), bbox_inches="tight", pad_inches=0
        )


# Saving results
def save_couplings(T_ms, Ts_gw, T_gw0, fs=3, folder=""):
    """
    Save visualizations of coupling matrices of GW and parametrized GW distances between pandas.

    Parameters
    ----------
    T_ms : numpy.ndarray
        Parametrized GW distance coupling.
    Ts_gw : list of numpy.ndarray
        List of GW couplings for each parameter.
    T_gw0 : numpy.ndarray
        One GW distance coupling (e.g. a baseline coupling).
    fs : float, optional
        Figure size scaling factor (default is 3).
    folder : str or Path, optional
        Directory to save the resulting figures (default is current folder).

    Saves
    -----
    Panda_GWs.pdf : visualization of the GW coupling at each parameter.
    Panda_MS.pdf : visualization of the pararametrized GW coupling.
    Panda_GW0.pdf : visualization of the baseline GW coupling.
    """
    nSteps = len(Ts_gw)

    vmin = 0
    vmax_gw = np.max([np.max(T) for T in Ts_gw])
    vmax_ms = np.max(T_ms)
    vmax_gw0 = np.max(T_gw0)
    vmax = np.max([vmax_gw, vmax_ms, vmax_gw0])

    # Single level couplings
    fig1, ax1 = plt.subplots(1, nSteps, figsize=(fs * nSteps + 1, fs))
    for idt in range(nSteps):
        im1 = ax1[idt].imshow(Ts_gw[idt], aspect="auto", vmin=vmin, vmax=vmax)
        # axes1[idt].set_title("dGW = %0.2f" % dGWs[idt])
        if idt == nSteps - 1:
            fig1.colorbar(im1, ax=ax1, shrink=1)

    fig2, ax2 = plt.subplots(1, 1, figsize=(fs, fs))
    im2 = ax2.imshow(T_ms, aspect="auto", vmin=vmin, vmax=vmax)
    fig2.colorbar(im2, ax=ax2, shrink=1)
    # axes2.set_title("dMS = %0.2f" % dMS)

    fig3, ax3 = plt.subplots(1, 1, figsize=(fs, fs))
    im3 = ax3.imshow(T_gw0, aspect="auto", vmin=vmin, vmax=vmax)
    fig3.colorbar(im3, ax=ax3, shrink=1)
    # axes3.set_title('dGW_0 = %0.2f' % dGW0)

    # Save
    fig1.savefig(Path(folder, f"Panda_GWs.pdf"), bbox_inches="tight", pad_inches=0)
    fig2.savefig(Path(folder, f"Panda_MS.pdf"), bbox_inches="tight", pad_inches=0)
    fig3.savefig(Path(folder, f"Panda_GW0.pdf"), bbox_inches="tight", pad_inches=0)
