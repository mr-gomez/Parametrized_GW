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

# Wrapper for ot.gromov._utils.gwloss.
# Decomposes cost matrices before calling gwloss.
def cost_gw(G0, C1, C2, p=None, q=None, loss_fun="square_loss", nx=None):
    # Obtain backend if it wasn't provided
    if p is None:
        p = unif(C1.shape[0])
    if q is None:
        q = unif(C2.shape[0])
    if nx is None:
        C1, C2, p, q, G0 = list_to_array(C1, C2, p, q, G0)
        nx = get_backend(C1, C2, p, q, G0)

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
    # Initialize variables
    nSteps = len(lC1)
    if p is None:
        p = unif(lC1[0].shape[0])
    if q is None:
        q = unif(lC2[0].shape[0])
    if nu is None:
        nu = unif(nSteps)

    # Obtain backend if it wasn't provided
    if nx is None:
        nx = get_backend(G0, *lC1, *lC2, p, q, nu)

    # Skipping other costs for now
    if loss_fun != "square_loss":
        raise NotImplementedError("Must use square_loss")

    # Decompose matrices
    lconstC, lhC1, lhC2 = init_matrix_list(
        lC1, lC2, p, q, loss_fun=loss_fun, nx=nx, option=0
    )

    # Compute gwloss for every parameter value
    lS = np.array([gwloss(lconstC[i], lhC1[i], lhC2[i], G0, nx) for i in range(nSteps)])

    # Return the parametrized GW cost
    if not all_steps:
        return nx.dot(nu, lS)
    # Also return the cost at every parameter value
    else:
        return nx.dot(nu, lS), lS


def init_matrix_list(lC1, lC2, p, q, loss_fun="square_loss", nx=None, option=0):
    # Throw error if incorrect option selected
    if option not in [0, 1, 2]:
        raise ValueError(f"option should be 0, 1 or 2. Got: {option}")

    # Applies init_matrix to the lists lC1 and lC2, which may have different lengths
    nSteps1 = len(lC1)
    nSteps2 = len(lC2)

    n = lC1[0].shape[0]
    m = lC2[0].shape[0]

    if nx is None:
        lC1 = list_to_array(lC1)
        lC2 = list_to_array(lC2)
        p, q = list_to_array(p, q)
        nx = get_backend(*lC1, *lC2, p, q)

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

    lhC1 = nx.zeros((nSteps1, n, n), type_as=lC1[0])
    lConst1 = nx.zeros((nSteps1, n, m), type_as=lC1[0])
    for idx in range(nSteps1):
        lhC1[idx, :, :] = h1(lC1[idx])

        lConst1[idx, :, :] = nx.dot(
            nx.dot(f1(lC1[idx]), nx.reshape(p, (-1, 1))),
            nx.ones((1, len(q)), type_as=q),
        )

    lhC2 = nx.zeros((nSteps1, m, m), type_as=lC2[0])
    lConst2 = nx.zeros((nSteps1, n, m), type_as=lC2[0])
    for jdx in range(nSteps2):
        lhC2[jdx, :, :] = h2(lC2[jdx])

        lConst2[jdx, :, :] = nx.dot(
            nx.ones((len(p), 1), type_as=p),
            nx.dot(nx.reshape(q, (1, -1)), f2(lC2[jdx]).T),
        )

    # Option 0: Same parameter space. Return sum of constants
    if option == 0:
        lConstC = lConst1 + lConst2
        return lConstC, lhC1, lhC2
    # Option 1: User wants both constants separately, regardless of parameter space
    elif option == 1:
        return lConst1, lConst2, lhC1, lhC2
    # Option 2: Different parameter spaces. Return "matrix" of sums lConst1[i]+lConst2[j]
    elif option == 2:
        lconstC = nx.zeros((nSteps1, nSteps2, n, m))
        for idx in range(nSteps1):
            for jdx in range(nSteps2):
                lconstC[idx, jdx, :, :] = lConst1[idx, :, :] + lConst2[jdx, :, :]

        return lconstC, lhC1, lhC2


def constant_batch(lConst_ind, i, j, loss_fun="square_loss", nx=None):
    if nx is None:
        nx = get_backend(*lConst_ind)

    if loss_fun == "square_loss":
        # f1==f2, so we only stored f1(C)
        lConstC1_bat = lConst_ind[i][0]
        lConstC2_bat = lConst_ind[j][0]
    elif loss_fun == "kl_loss":
        # f1 != f2, so we stored both f1(C) and f2(C)
        lConstC1_bat = lConst_ind[i][0]
        lConstC2_bat = lConst_ind[j][1]
    else:
        raise ValueError(
            f"Unknown `loss_fun='{loss_fun}'`. Use one of: {'square_loss', 'kl_loss'}."
        )

    # Reshape to vectorize operations
    # NOTE: axis is different in expand_dims
    # Thanks to Numpy's broadcasting rules, we don't need
    # the extra dot product with np.ones
    lConstC1_bat = np.expand_dims(lConstC1_bat, axis=2)
    lConstC2_bat = np.expand_dims(lConstC2_bat, axis=1)

    return lConstC1_bat + lConstC2_bat


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
    lhC1_bat = nx.zeros(N).astype(object)
    lhC2_bat = nx.zeros(N).astype(object)
    for idx in range(N):
        lhC1_bat[idx] = h1(lCs[idx])
        lhC2_bat[idx] = h2(lCs[idx])

    # Compute constant terms
    lConst_ind = nx.zeros(N).astype(object)
    for i in range(N):
        lC = lCs[i]
        p = Ps[i]

        if loss_fun == "square_loss":
            # f1 and f2 are the same function
            # We don't store duplicate data
            lfC = np.expand_dims(f1(lC), axis=0)
        elif loss_fun == "kl_loss":
            # f1 and f2 are different functions
            # We need to store both
            lfC = nx.stack((f1(lC), f2(lC)), axis=0)

        # Vectorized dot product for both f1, f2 and all times
        # NOTE:
        # constC.shape == (nSteps, len(lC)) if loss_fun=='square_loss'
        # constC.shape == (2, nSteps, len(lC)) if loss_fun=='kl_loss'
        constC = nx.dot(lfC, p)

        lConst_ind[i] = constC

    # Return decoupled constants
    # User will have to add lConst_ind[i] and the transpose of lConst_ind[j]
    # with constant_batch during runtime
    if option == 0:
        return lConst_ind, lhC1_bat, lhC2_bat

    # Otherwise, we add all constants here
    lConst_bat = nx.zeros((N, N)).astype(object)
    for i in range(N):
        for j in range(N):
            # Add both constants
            lConst_bat[i, j] = constant_batch(
                lConst_ind, i, j, loss_fun=loss_fun, nx=nx
            )

    return lConst_bat, lhC1_bat, lhC2_bat


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
            # if i==j:
            #     continue

            # Shape: (T, n2, n2)
            lhC2 = lhC2_bat[j]

            lA_bat[i, j] = tensor_product_list(0, lhC1, lhC2, T_bat[i, j], nx=nx)
            # Result shape: (T, n1, n2)

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


# -------------------------------------
# Computing GW between sets of pmnets
# -------------------------------------
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
