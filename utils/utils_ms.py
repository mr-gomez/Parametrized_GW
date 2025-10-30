import numpy as np
from ot.gromov import gromov_wasserstein
from ot.utils import list_to_array
from ot.lp import emd
from ot.utils import unif
from ot.backend import get_backend, NumpyBackend

from ot.gromov._utils import init_matrix, gwloss, gwggrad

from utils.utils import display_time

# Machine epsilon
from sys import float_info

epsilon_ = float_info.epsilon

from time import time

# -------------------------------------
# Loss functions
# -------------------------------------
# Wrapper for ot.gromov._utils.gwloss.
# Decomposes cost matrices before calling gwloss.
def cost_gw(G0, C1, C2, p=None, q=None, loss_fun="square_loss", nx=None):
    if loss_fun != "square_loss":
        raise NotImplementedError("Must use square_loss")

    # Initialize matrices
    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun, nx=nx)

    # Compute cost
    return gwloss(constC, hC1, hC2, G0, nx)


# Computes multiscale cost of a coupling between two pm-nets
def cost_ms(
    G0,
    lC1,
    lC2,
    p=None,
    q=None,
    nu=None,
    loss_fun="square_loss",
    nx=None,
    all_steps=False,
):
    # Skipping other costs for now
    if loss_fun != "square_loss":
        raise NotImplementedError("Must use square_loss")

    # Decompose matrices
    lconstC, lhC1, lhC2 = init_matrix_list(
        lC1, lC2, p, q, loss_fun=loss_fun, nx=nx, option=0
    )

    # Compute gwloss for every parameter value
    nSteps = len(lC1)
    lS = np.array([gwloss(lconstC[i], lhC1[i], lhC2[i], G0, nx) for i in range(nSteps)])

    # Return the parametrized GW cost
    if not all_steps:
        return np.dot(nu, lS)
    # Also return the cost at every parameter value
    else:
        return np.dot(nu, lS), lS


# Given a set of pmnets lCs, compute the GW distance between the lCs at each time t
def compute_dGWs(nSteps, lCs, verbose=1):
    N = len(lCs)
    dGWs = np.zeros((nSteps, N, N))
    for t in range(nSteps):
        for i in range(N):
            for j in range(i + 1, N):
                time_start = time()

                T, log = gromov_wasserstein(lCs[i][t, :, :], lCs[j][t, :, :], log=True)
                dGWs[t, i, j] = log["gw_dist"]

                time_end = time()
                if verbose:
                    print(
                        f"({t+1},{i+1},{j+1})/({nSteps},{N},{N}): "
                        + display_time(seconds=time_end - time_start)
                    )
    print()

    dGWs = np.maximum(dGWs, dGWs.transpose((0, 2, 1)))
    dGWs = np.maximum(dGWs, 0)

    return dGWs


# -------------------------------------
# Pre-processing of pm-nets (similar to init_matrix)
# -------------------------------------
def add_constants(lconstC1, lconstC2):
    # Reshape to vectorize operations
    # NOTE: axis is different in expand_dims
    # Thanks to Numpy's broadcasting rules, we don't need
    # the extra dot product with np.ones
    d = lconstC1.ndim
    return np.expand_dims(lconstC1, axis=d) + np.expand_dims(lconstC2, axis=d - 1)


def add_constants_batch(lconstC1_bat, lconstC2_bat):
    N = len(lconstC1_bat)
    M = len(lconstC2_bat)

    lConst_bat = np.zeros((N, M)).astype(object)
    for idx in range(N):
        for jdx in range(M):
            lConst_bat[idx, jdx] = add_constants(lconstC1_bat[idx], lconstC2_bat[jdx])

    return lConst_bat


def init_matrix_list(lC1, lC2, p, q, loss_fun="square_loss", nx=None, option=0):
    # Throw error if incorrect option selected
    if option not in [0, 1, 2]:
        raise ValueError(f"option should be 0, 1 or 2. Got: {option}")

    # Decompose matrices, but constants are decoupled
    lfC1_bat, lfC2_bat, lhC1_bat, lhC2_bat = init_matrix_batch(
        [lC1, lC2], [p, q], loss_fun, nx, option=0
    )

    # Option 0: Return vector of combined constants
    if option == 0:
        lConstC = add_constants(lfC1_bat[0], lfC2_bat[1])
        return lConstC, lhC1_bat[0], lhC2_bat[1]
    # Option 1: Return vectors of decoupled constants
    if option == 1:
        return lfC1_bat[0], lfC2_bat[1], lhC1_bat[0], lhC2_bat[1]
    # Option 2: Different parameter spaces. Return matrix of constants
    elif option == 2:
        lconstC = add_constants_batch(lfC1_bat[0], lfC2_bat[1])
        return lconstC, lhC1_bat[0], lhC2_bat[1]


def init_matrix_batch(lCs, Ps, loss_fun="square_loss", nx=None, option=0):
    # option=0: Return decoupled constants
    # option=1: Sum constants, return n*n array of constants

    # Throw error if incorrect option selected
    if option not in [0, 1]:
        raise ValueError(f"option should be 0 or 1. Got: {option}")

    if nx is None:
        lCs = [list_to_array(C) for C in lCs]
        Ps = [list_to_array(p) for p in Ps]
        nx = get_backend(*lCs, *Ps)

    if loss_fun == "square_loss":

        def f1(a):
            return a**2

        def f2(b):
            return b**2

        def h1(a):
            return a

        def h2(b):
            return 2 * b

    elif loss_fun == "kl_loss":

        def f1(a):
            return a * nx.log(a + 1e-15) - a

        def f2(b):
            return b

        def h1(a):
            return a

        def h2(b):
            return nx.log(b + 1e-15)

    else:
        raise ValueError(
            f"Unknown `loss_fun='{loss_fun}'`. Use one of: {'square_loss', 'kl_loss'}."
        )

    # Possible improvement: Implement squareform if using square_loss.
    N = len(lCs)
    # sq_idx = sq_idx_fun(N)

    # Applies init_matrix to each pm-net. They may have different lengths
    lhC1_bat = nx.zeros(N).astype(object)  # entry shape (n_steps_t, n_t, n_t)
    lhC2_bat = nx.zeros(N).astype(object)  # entry shape (n_steps_t, n_t, n_t)
    lfC1_bat = nx.zeros(N).astype(object)  # entry shape (n_steps_t, n_t)
    lfC2_bat = nx.zeros(N).astype(object)  # entry shape (n_steps_t, n_t)
    for idx in range(N):
        lC = lCs[idx]
        p = Ps[idx]

        # Shape: (n_steps_t, n_t, n_t)
        lhC1_bat[idx] = h1(lC)
        lhC2_bat[idx] = h2(lC)

        # Constants
        # Shape: (n_steps_t, n_t)
        lfC1_bat[idx] = nx.dot(f1(lC), p)
        lfC2_bat[idx] = nx.dot(f2(lC), p)

    # Return decoupled constants
    # User will have to call add_constants(lfC1_bat[i], lfC2_bat[j])
    if option == 0:
        return lfC1_bat, lfC2_bat, lhC1_bat, lhC2_bat

    # Add all constants here
    elif option == 1:
        lConst_bat = add_constants_batch(lfC1_bat, lfC2_bat)
        return lConst_bat, lhC1_bat, lhC2_bat


# -------------------------------------
# Tensor products and GW loss and gradients for pm-nets
# -------------------------------------
def tensor_product_list(lconstC, lhC1, lhC2, T, nx=None):
    # Shape lconstC: (T,n1,n2)
    # Shape lhC1: (T, n1, n1)
    # Shape lhC2: (T, nj, n2)
    # Shape Ts: (n1, n2)

    if nx is None:
        constC, hC1, hC2, T = list_to_array(constC, hC1, hC2, T)
        nx = get_backend(constC, hC1, hC2, T)

    lA = -nx.matmul(
        # Shape: (T, n1, n1) @ (n1, n2) = (T, n1, n2)
        nx.matmul(lhC1, T),
        # Shape: (T, n2, n2) (tranpose each (n2,n2) matrix)
        nx.transpose(lhC2, (0, 2, 1)),
    )
    # Result shape: (T, n1, n2)

    # Shape: (T, n1, n2)
    return lconstC + lA


def gwloss_list(lconstC, lhC1, lhC2, T, nx=None):
    if nx is None:
        lconstC, lhC1, lhC2 = list_to_array(lconstC, lhC1, lhC2)
        nx = get_backend(lconstC, lhC1, lhC2)

    # Shape of lTens: (T, n1, n2)
    lTens = tensor_product_list(lconstC, lhC1, lhC2, T, nx)

    # Shape of product: (T, n1, n2) * (n1, n2) = (T, n1, n2)
    # Shape of sum(..., axis=(1,2)): (T,)
    lLoss = nx.sum(lTens * T, axis=(1, 2))

    # Shape: (T,)
    return lLoss


def gwggrad_list(lconstC, lhC1, lhC2, T, nx=None):
    # Shape: (T, n1, n2)
    return 2 * tensor_product_list(
        lconstC, lhC1, lhC2, T, nx
    )  # [12] Prop. 2 misses a 2 factor


def tensor_product_batch(lconstC_bat, lhC1_bat, lhC2_bat, T_bat, nx=None):
    # Shape lconstC: (N,N)
    # Shape lconstC[i,j]: (T,ni,nj)
    # -------------------------------
    # Shape lhC1, lhC2: (N,)
    # Shape lhC1[i], lhC2[i]: (T, ni, ni)
    # -------------------------------
    # Shape Ts: (N,N)
    # Shape Ts[i,j]: (ni, nj)

    if nx is None:
        lconstC_bat, lhC1_bat, lhC2_bat, T_bat = list_to_array(
            lconstC_bat, lhC1_bat, lhC2_bat, T_bat
        )
        nx = get_backend(lconstC_bat, lhC1_bat, lhC2_bat, T_bat)

    N = len(lhC1_bat)
    lA_bat = nx.zeros((N, N)).astype(object)
    for i in range(N):
        # Shape: (T, n1, n1)
        lhC1 = lhC1_bat[i]
        for j in range(N):
            # Shape: (T, n2, n2)
            lhC2 = lhC2_bat[j]

            # Shape of result: (T, n1, n2)
            lA_bat[i, j] = tensor_product_list(0, lhC1, lhC2, T_bat[i, j], nx=nx)

    # Shape: (N,N)
    # Shape of lTens_bat[i,j]: (T, ni, nj)
    lTens_bat = lconstC_bat + lA_bat
    return lTens_bat


def gwloss_batch(lconstC_bat, lhC1_bat, lhC2_bat, T_bat, nx=None):
    # Shape: (N,N)
    # Shape of lTens_bat[i,j]: (T, ni, nj)
    lTens_bat = tensor_product_batch(lconstC_bat, lhC1_bat, lhC2_bat, T_bat, nx)

    if nx is None:
        lTens_bat, T_bat = list_to_array(lTens_bat, T_bat)
        nx = get_backend(lTens_bat, T_bat)

    N = len(lhC1_bat)
    nSteps = len(lhC1_bat[0])
    lLoss_bat = np.zeros((N, N, nSteps))
    for i in range(N):
        for j in range(N):
            # if i==j:
            # continue
            # Shape of product: (T, ni, nj) * (ni, nj) = (T, ni, nj)
            # Shape of sum(..., axis=(1,2)): (T,)
            lLoss_bat[i, j, :] = nx.sum(lTens_bat[i, j] * T_bat[i, j], axis=(1, 2))

    # Shape: (N, N, T)
    return lLoss_bat


def gwggrad_batch(lconstC_bat, lhC1_bat, lhC2_bat, T_bat, nx=None):
    # Shape: (N,N)
    # Shape of result[i,j]: (T, ni, nj)
    return 2 * tensor_product_batch(
        lconstC_bat, lhC1_bat, lhC2_bat, T_bat, nx
    )  # [12] Prop. 2 misses a 2 factor


def emd_batch(Ps, M_bat, **kwargs):
    # def emd(a, b, M, numItermax=100000, log=False, center_dual=True, numThreads=1, check_marginals=True):
    # shape Ps: (N,)
    # shape Ms: (N,N)
    # shape Ps[i]: ni
    # shape Ms[i,j]: (ni,nj)
    N = len(Ps)
    T_bat = np.empty_like(M_bat, dtype=object)
    logs = np.empty_like(M_bat, dtype=object)
    for i in range(N):
        for j in range(i, N):
            T, log = emd(Ps[i], Ps[j], M_bat[i, j], **kwargs)
            T_bat[i, j] = T
            T_bat[j, i] = T.T

            logs[i, j] = log
            logs[j, i] = log

    return T_bat, logs
