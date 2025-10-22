# Credit:
# The functions on this file are based on the function ot.gromov.gromov_wasserstein of the POT package
# TODO: Improve citation and documentation

import numpy as np
from scipy.special import binom
from scipy.spatial.distance import squareform
from scipy.stats import entropy
from scipy.optimize import minimize, Bounds

from ot.utils import list_to_array
from ot.optim import cg, line_search_armijo, solve_1d_linesearch_quad
from ot.lp import emd
from ot.utils import unif
from ot.backend import get_backend, NumpyBackend

from ot.gromov._utils import init_matrix, gwloss, gwggrad

from .utils import sq_idx_fun, sq_to_idx_fun

from itertools import combinations, chain

from warnings import warn

# CHECK: Is this the correct machine epsilon or should we change it
#        depending on np's data type?
from sys import float_info
epsilon_ = float_info.epsilon

from time import time

def gromov_wasserstein_ms(
    lC1,
    lC2,
    p=None,
    q=None,
    nu=None,
    loss_fun="square_loss",
    symmetric=None,
    log=False,
    armijo=False,
    G0=None,
    max_iter=1e4,
    tol_rel=1e-9,
    tol_abs=1e-9,
    **kwargs,
):
    nSteps = len(lC1)
    arr = [*lC1, *lC2]

    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(lC1[0].shape[0])
    if q is not None:
        arr.append(list_to_array(q))
    else:
        q = unif(lC2[0].shape[0])
    if nu is not None:
        arr.append(nu)
    else:
        nu = unif(len(lC1))

    if G0 is not None:
        G0_ = G0
        arr.append(G0)

    nx = get_backend(*arr)
    p0, q0, lC10, lC20 = p, q, lC1, lC2

    p = nx.to_numpy(p0)
    q = nx.to_numpy(q0)
    lC1 = nx.to_numpy(lC10)
    lC2 = nx.to_numpy(lC20)
    if symmetric is None:
        symmetric = True
        for i in range(nSteps):
            symmetric = symmetric and np.allclose(lC1[i], lC1[i].T, atol=1e-10)
            symmetric = symmetric and np.allclose(lC2[i], lC2[i].T, atol=1e-10)

    # Initialize an initial guess for a correspondence
    if G0 is None:
        G0 = p[:, None] * q[None, :]
    else:
        G0 = nx.to_numpy(G0_)
        # Check marginals of G0
        np.testing.assert_allclose(G0.sum(axis=1), p, atol=1e-08)
        np.testing.assert_allclose(G0.sum(axis=0), q, atol=1e-08)

    # cg for GW is implemented using numpy on CPU
    np_ = NumpyBackend()

    # ------------------------
    # Decompose cost matrices
    # ------------------------
    lconstC, lhC1, lhC2 = init_matrix_list(lC1, lC2, p, q, loss_fun, np_, option=0)

    def f(G):
        lS = [gwloss(lconstC[i], lhC1[i], lhC2[i], G, np_) for i in range(nSteps)]
        return nx.dot(nu, np.array(lS))

    if symmetric:

        def df(G):
            S = nx.zeros((*G.shape, nSteps))
            for i in range(nSteps):
                S[:, :, i] = gwggrad(lconstC[i], lhC1[i], lhC2[i], G, np_)
            return nx.dot(S, nu)

    else:
        lconstCt, lhC1t, lhC2t = init_matrix_list(
            np.transpose(lC1, axes=(0, 2, 1)),
            np.transpose(lC2, axes=(0, 2, 1)),
            p,
            q,
            loss_fun,
            np_,
            option=0,
        )

        def df(G):
            S = nx.zeros((*G.shape, nSteps))
            for i in range(nSteps):
                S[:, :, i] = gwggrad(lconstC[i], lhC1[i], lhC2[i], G, np_)
                +gwggrad(lconstCt[i], lhC1t[i], lhC2t[i], G, np_)
            return 0.5 * nx.dot(S, nu)

    if loss_fun == "kl_loss":
        raise NotImplementedError
        armijo = True  # there is no closed form line-search with KL

    if armijo:
        raise NotImplementedError

        def line_search(cost, G, deltaG, Mi, cost_G, df_G, **kwargs):
            # ----------------------------------------
            # Check what this function does before implementing
            return line_search_armijo(cost, G, deltaG, Mi, cost_G, nx=np_, **kwargs)
            # ----------------------------------------

    else:

        def line_search(cost, G, deltaG, Mi, cost_G, **kwargs):
            return solve_gromov_linesearch_multiscale(
                G,
                deltaG,
                cost_G,
                lconstC,
                lhC1,
                lhC2,
                nu,
                M=0.0,
                reg=1.0,
                nx=np_,
                **kwargs,
            )

    if log:
        res, log = cg(
            p,
            q,
            0.0,
            1.0,
            f,
            df,
            G0,
            line_search,
            log=True,
            numItermax=max_iter,
            stopThr=tol_rel,
            stopThr2=tol_abs,
            **kwargs,
        )
        log["gw_dist"] = nx.from_numpy(log["loss"][-1], type_as=lC10[0])
        log["u"] = nx.from_numpy(log["u"], type_as=lC10[0])
        log["v"] = nx.from_numpy(log["v"], type_as=lC10[0])
        return nx.from_numpy(res, type_as=lC10[0]), log
    else:
        return nx.from_numpy(
            cg(
                p,
                q,
                0.0,
                1.0,
                f,
                df,
                G0,
                line_search,
                log=False,
                numItermax=max_iter,
                stopThr=tol_rel,
                stopThr2=tol_abs,
                **kwargs,
            ),
            type_as=lC10[0],
        )


def solve_gromov_linesearch_multiscale(
    G,
    deltaG,
    cost_G,
    lconstC,
    lhC1,
    lhC2,
    nu,
    M,
    reg,
    alpha_min=None,
    alpha_max=None,
    nx=None,
    **kwargs,
):
    if nx is None:
        G, deltaG, lconstC, lhC1, lhC2, M = list_to_array(
            G, deltaG, lconstC, lhC1, lhC2, M
        )

        if isinstance(M, int) or isinstance(M, float):
            nx = get_backend(G, deltaG, lconstC, lhC1, lhC2)
        else:
            nx = get_backend(G, deltaG, lconstC, lhC1, lhC2, M)

    nSteps = len(lhC1)
    a_vec = nx.zeros(nSteps)
    b_vec = nx.zeros(nSteps)
    for i in range(nSteps):
        dot_dG = nx.dot(nx.dot(lhC1[i], deltaG), lhC2[i].T)
        dot_G = nx.dot(nx.dot(lhC1[i], G), lhC2[i].T)

        # Note: In contrast to solve_gromov_linesearch, we do not multiply reg
        # by 2 (in b) because lhC2 has an extra factor of 2 compared to lC2.
        # Note 2: it seems the above note is outdated?
        a_vec[i] = -reg * nx.sum(dot_dG * deltaG)
        b_vec[i] = nx.sum(M * deltaG) - reg * (
            nx.sum(dot_dG * G) + nx.sum(dot_G * deltaG)
        )

    a = nx.dot(nu, a_vec)
    b = nx.dot(nu, b_vec)

    alpha = solve_1d_linesearch_quad(a, b)
    if alpha_min is not None or alpha_max is not None:
        alpha = np.clip(alpha, alpha_min, alpha_max)

    # the new cost is deducted from the line search quadratic function
    # print('a,b,alpha,cost,new cost')
    # print(a)
    # print(b)
    # print(alpha)
    # print(cost_G)
    cost_G = cost_G + a * (alpha**2) + b * alpha
    # print(cost_G)
    # print()

    return alpha, 1, cost_G

def cost_gw(G0, C1, C2, p=None, q=None, loss_fun="square_loss", nx=None):
    # Setup
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
    # Setup
    nSteps = len(lC1)
    if p is None:
        p = unif(lC1[0].shape[0])
    if q is None:
        q = unif(lC2[0].shape[0])
    if nu is None:
        nu = unif(nSteps)
    if nx is None:
        nx = get_backend(G0, *lC1, *lC2, p, q, nu)

    if loss_fun != "square_loss":
        raise NotImplementedError("Must use square_loss")

    # Initialize matrices
    lconstC = [None] * nSteps
    lhC1 = [None] * nSteps
    lhC2 = [None] * nSteps
    for i in range(nSteps):
        lconstC[i], lhC1[i], lhC2[i] = init_matrix(
            lC1[i], lC2[i], p, q, loss_fun, nx=nx
        )

    lS = np.array([gwloss(lconstC[i], lhC1[i], lhC2[i], G0, nx) for i in range(nSteps)])

    if not all_steps:
        return nx.dot(nu, lS)
    else:
        return nx.dot(nu, lS), lS


def cost_gw_manual(G0, C1, C2, p=None, q=None):
    # C1 and C2 must be square matrices
    N1 = C1.shape[0]
    N2 = C2.shape[0]

    iter1 = combinations(range(N1), 2)
    iter2 = combinations(range(N2), 2)

    total_cost = 0
    for i, j in iter1:
        for k, l in iter2:
            total_cost += np.abs(C1[i, j] - C2[k, l]) ** 2 * G0[i, k] * G0[j, l]

    return total_cost


def cost_ms_manual(G0, lC1, lC2, nu=None, p=None, q=None, all_steps=False):
    nSteps = len(lC1)
    if nu is None:
        nu = unif(nSteps)

    costs = np.zeros(nSteps)
    for idx in range(nSteps):
        C1 = lC1[idx]
        C2 = lC2[idx]

        # C1 and C2 must be square matrices
        N1 = C1.shape[0]
        N2 = C2.shape[0]

        iter1 = combinations(range(N1), 2)
        iter2 = combinations(range(N2), 2)

        total_cost = 0
        for i, j in iter1:
            for k, l in iter2:
                total_cost += np.abs(C1[i, j] - C2[k, l]) ** 2 * G0[i, k] * G0[j, l]
        costs[idx] = total_cost

    if not all_steps:
        return np.dot(nu, costs)
    else:
        return np.dot(nu, costs), costs


############################
# Matching parameter spaces
############################
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


def gw_ms_couple_nu(
    lC1,
    lC2,
    p=None,
    q=None,
    nu1=None,
    nu2=None,
    loss_fun="square_loss",
    symmetric=None,
    armijo=False,
    E0=None,
    G0=None,
    max_iter=1e4,
    numItermaxEmd=100000,
    tol_rel=1e-9,
    tol_abs=1e-9,
    verbose=False,
    log=False,
    **kwargs,
):
    nSteps1 = len(lC1)
    nSteps2 = len(lC2)
    n1 = lC1[0].shape[0]
    n2 = lC2[0].shape[0]

    arr = [*lC1, *lC2]

    # ----------------
    # Default choices
    # ----------------
    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(lC1[0].shape[0])
        # p = unif(lC1[0].shape[0], type_as=lC1[0])
    if q is not None:
        arr.append(list_to_array(q))
    else:
        q = unif(lC2[0].shape[0])
        # q = unif(lC2[0].shape[0], type_as=lC2[0])
    if nu1 is not None:
        arr.append(nu1)
    else:
        nu1 = unif(nSteps1)
    if nu2 is not None:
        arr.append(nu2)
    else:
        nu2 = unif(nSteps2)

    if E0 is not None:
        E0_ = E0
        arr.append(E0)

    if G0 is not None:
        G0_ = G0
        arr.append(G0)

    nx = get_backend(*arr)
    p0, q0, nu10, nu20, lC10, lC20 = p, q, nu1, nu2, lC1, lC2

    p = nx.to_numpy(p0)
    q = nx.to_numpy(q0)
    nu1 = nx.to_numpy(nu10)
    nu2 = nx.to_numpy(nu20)
    lC1 = nx.to_numpy(lC10)
    lC2 = nx.to_numpy(lC20)
    if symmetric is None:
        symmetric = True
        for i in range(nSteps1):
            symmetric = symmetric and np.allclose(lC1[i], lC1[i].T, atol=1e-10)

        for j in range(nSteps2):
            symmetric = symmetric and np.allclose(lC2[j], lC2[j].T, atol=1e-10)

    # Initialize guesses for correspondences
    # Correspondence on nu
    if E0 is None:
        E0 = nu1[:, None] * nu2[None, :]
    else:
        E0 = nx.to_numpy(E0_)
        # Check marginals of G0
        np.testing.assert_allclose(E0.sum(axis=1), nu1, atol=1e-08)
        np.testing.assert_allclose(E0.sum(axis=0), nu2, atol=1e-08)

    # Correspondence on spaces
    if G0 is None:
        G0 = p[:, None] * q[None, :]
    else:
        G0 = nx.to_numpy(G0_)
        # Check marginals of G0
        np.testing.assert_allclose(G0.sum(axis=1), p, atol=1e-08)
        np.testing.assert_allclose(G0.sum(axis=0), q, atol=1e-08)

    # cg for GW is implemented using numpy on CPU
    np_ = NumpyBackend()

    # ------------------------
    # Decompose cost matrices
    # ------------------------
    lconstC, lhC1, lhC2 = init_matrix_list(lC1, lC2, p, q, loss_fun, np_, option=2)

    # -----------------------------
    # Cost functions and gradients
    # -----------------------------
    def f_mat(G):
        lS = nx.zeros((nSteps1, nSteps2))
        for i in range(nSteps1):
            for j in range(nSteps2):
                lS[i, j] = gwloss(lconstC[i, j], lhC1[i], lhC2[j], G, np_)
        return lS

    def f(E, cost_mat):
        return nx.sum(E * cost_mat)

    if symmetric:

        def df_mat(G):
            S_mat = nx.zeros((nSteps1, nSteps2, n1, n2))
            for i in range(nSteps1):
                for j in range(nSteps2):
                    S_mat[i, j] = gwggrad(lconstC[i, j], lhC1[i], lhC2[j], G, np_)
            return S_mat

        def df(E, grad_mat):
            # I want the sum of E[i,j]*grad_mat[i,j,:,:] over all i, j
            grad_mat_w = nx.reshape(E, (*E.shape, 1, 1)) * grad_mat
            return nx.sum(grad_mat_w, axis=(0, 1))

    else:
        lC1t = [C.T for C in lC1t]
        lC2t = [C.T for C in lC2t]
        lconstCt, lhC1t, lhC2t = init_matrix_list(
            lC1t, lC2t, p, q, loss_fun, np_, option=2
        )

        def df_mat(G):
            S_mat = nx.zeros((nSteps1, nSteps2, n1, n2))
            for i in range(nSteps1):
                for j in range(nSteps2):
                    S_mat[i, j] = gwggrad(lconstC[i, j], lhC1[i], lhC2[j], G, np_)
                    S_mat[i, j] += gwggrad(
                        lconstCt[i, j, :, :], lhC1t[i], lhC2t[j], G, np_
                    )
            return S_mat

        def df(E, grad_mat):
            grad_mat_w = nx.reshape(E, (*E.shape, 1, 1)) * grad_mat
            return 0.5 * nx.sum(grad_mat_w, axis=(0, 1))

    if loss_fun == "kl_loss":
        raise NotImplementedError
        armijo = True  # there is no closed form line-search with KL

    # ----------------------------
    # Define line search function
    # ----------------------------
    if armijo:
        raise NotImplementedError

        def line_search(cost, G, deltaG, Mi, cost_G, df_G, **kwargs):
            # ----------------------------------------
            # Check what this function does before implementing
            return line_search_armijo(cost, G, deltaG, Mi, cost_G, nx=np_, **kwargs)
            # ----------------------------------------

    else:

        def line_search(E, G, deltaG, cost_G, **kwargs):
            return linesearch_multiscale_nu(
                G,
                deltaG,
                cost_G,
                lconstC,
                lhC1,
                lhC2,
                E,
                M=0.0,
                reg=1.0,
                nx=np_,
                **kwargs,
            )

    # -----------------------------
    # Loop preparations
    # -----------------------------
    it = 0

    G = G0.copy()
    E = E0.copy()
    cost_G = f(E, f_mat(G))

    if log:
        log = {"loss": [cost_G]}

    if verbose:
        print(
            "{:5s}|{:12s}|{:8s}|{:8s}".format(
                "It.", "Loss", "Relative loss", "Absolute loss"
            )
            + "\n"
            + "-" * 48
        )
        print("{:5d}|{:8e}|{:8e}|{:8e}".format(it, cost_G, 0, 0))

    # -----------------------------
    # Alternating gradient descent
    # -----------------------------
    loop = 1

    while loop:
        it += 1
        old_cost_G = cost_G

        # Calculate matrix of costs and gradients for the current G
        cost_mat = f_mat(G)
        grad_mat = df_mat(G)

        # ----- Optimize E -----
        E, innerlog_1 = emd(nu1, nu2, cost_mat, numItermax=numItermaxEmd, log=True)
        # print(E)
        # print()

        # Find cost and gradient for the new E and the old G
        cost_G = f(E, cost_mat)
        grad_G = df(E, grad_mat)

        # ----- Optimize G -----
        # Find closest coupling to gradient
        Gc, innerlog_2 = emd(p, q, grad_G, numItermax=numItermaxEmd, log=True)
        deltaG = Gc - G

        # Solve exact line search problem
        alpha, fc, cost_G = line_search(E, G, deltaG, cost_G)
        G = G + alpha * deltaG

        # ----- Test convergence -----
        if it >= max_iter:
            loop = 0

        abs_delta_cost_G = abs(cost_G - old_cost_G)
        relative_delta_cost_G = (
            abs_delta_cost_G / abs(cost_G) if cost_G != 0.0 else np.nan
        )
        if relative_delta_cost_G < tol_rel or abs_delta_cost_G < tol_abs:
            loop = 0

        # ----- show results -----
        if verbose:
            if it % 20 == 0:
                print(
                    "{:5s}|{:12s}|{:8s}|{:8s}".format(
                        "It.", "Loss", "Relative loss", "Absolute loss"
                    )
                    + "\n"
                    + "-" * 48
                )
            print(
                "{:5d}|{:8e}|{:8e}|{:8e}".format(
                    it, cost_G, relative_delta_cost_G, abs_delta_cost_G
                )
            )

        # ----- update logs -----
        if log:
            log["loss"].append(cost_G)

    # -----------------------------
    # Alternating gradient descent
    # -----------------------------
    if log:
        # log.update(innerlog_1)
        # log.update(innerlog_2)
        return E, G, log, innerlog_1, innerlog_2
    else:
        return E, G


def linesearch_multiscale_nu(
    G,
    deltaG,
    cost_G,
    lconstC,
    lhC1,
    lhC2,
    E,
    M,
    reg,
    alpha_min=None,
    alpha_max=None,
    nx=None,
    **kwargs,
):
    if nx is None:
        lhC1 = list_to_array(lhC1)
        lhC2 = list_to_array(lhC2)
        G, deltaG, lconstC, M = list_to_array(G, deltaG, lconstC, M)

        if isinstance(M, int) or isinstance(M, float):
            nx = get_backend(G, deltaG, lconstC, *lhC1, *lhC2)
        else:
            nx = get_backend(G, deltaG, lconstC, *lhC1, *lhC2, M)

    nSteps1 = len(lhC1)
    nSteps2 = len(lhC2)
    a_mat = nx.zeros((nSteps1, nSteps2))
    b_mat = nx.zeros((nSteps1, nSteps2))
    for i in range(nSteps1):
        for j in range(nSteps2):
            dot_dG = nx.dot(nx.dot(lhC1[i], deltaG), lhC2[j].T)
            dot_G = nx.dot(nx.dot(lhC1[i], G), lhC2[j].T)

            # Note: In contrast to solve_gromov_linesearch, we do not multiply reg
            # by 2 (in b) because lhC2 has an extra factor of 2 compared to lC2.
            # Note 2: it seems the above note is outdated?
            a_mat[i, j] = -reg * nx.sum(dot_dG * deltaG)
            b_mat[i, j] = nx.sum(M * deltaG) - reg * (
                nx.sum(dot_dG * G) + nx.sum(dot_G * deltaG)
            )
    a = nx.sum(a_mat * E)
    b = nx.sum(b_mat * E)

    alpha = solve_1d_linesearch_quad(a, b)
    if alpha_min is not None or alpha_max is not None:
        alpha = np.clip(alpha, alpha_min, alpha_max)

    # the new cost is deducted from the line search quadratic function
    cost_G = cost_G + a * (alpha**2) + b * alpha

    return alpha, 1, cost_G


############################
# Metric learning
############################
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

    # Applies init_matrix to the lists lC1 and lC2, which may have different lengths
    N = len(lCs)
    T = lCs[0].shape[0]

    # TODO: Implement squareform if using square_loss.
    # sq_idx = sq_idx_fun(N)

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

def gw_ms_learn_nu(
    lCs,
    Ps=None,
    S=None,  # Score function
    dS=None,  # Gradient of score
    lambda_S=1e-5,  # Regularization for KL divergence of nu
    loss_fun="square_loss",
    option=1,
    symmetric=None,
    armijo=False,
    nu=None,
    G0_bat=None,
    max_iter=1e4,
    numItermaxEmd=1e5,
    tol_rel=1e-9,
    tol_abs=1e-9,
    verbose=False,
    log=False,
    **kwargs,
):
    # NOTE:
    # option=0: Work with decoupled constants. Need to add them during runtime (slower)
    # option=1: Sum constants, return n*n array of constants (more memory)

    # lCs is a list of multiscale networks of the same length
    N = len(lCs)
    N_sq = int(binom(N, 2))

    nSteps = lCs[0].shape[0]

    # Indexing functions
    sq_idx = sq_idx_fun(N)
    sq_to_idx = sq_to_idx_fun(N)

    arr = [*lCs]

    # ----------------
    # Default choices
    # ----------------
    if Ps is not None:
        arr.extend([list_to_array(p) for p in Ps])
    else:
        Ps = [unif(lC.shape[1]) for lC in lCs]
    if nu is not None:
        arr.append(nu)
    else:
        nu = unif(nSteps)
    if G0_bat is not None:
        G0_bat_ = G0_bat

        G_list = chain.from_iterable(G0_bat.tolist())
        arr.extend(G_list)

    nx = get_backend(*arr)
    lCs0, Ps0, nu0 = lCs, Ps, nu

    Ps = [nx.to_numpy(p) for p in Ps0]
    lCs = [nx.to_numpy(lC) for lC in lCs0]
    nu = nx.to_numpy(nu0)

    if symmetric is None:
        symmetric = True
        for idx in range(N):
            for jdx in range(nSteps):
                symmetric = symmetric and np.allclose(
                    lCs[idx][jdx], lCs[idx][jdx].T, atol=1e-10
                )

    # Correspondence on spaces
    if G0_bat is None:
        G0_bat = np.empty((N, N), dtype=object)
        for i in range(N):
            p = Ps[i]
            for j in range(i, N):
                q = Ps[j]

                # idx = sq_idx(i,j)
                # G0_bat[idx] = p[:, None] * q[None, :]
                G0_bat[i, j] = p[:, None] * q[None, :]
                G0_bat[j, i] = q[:, None] * p[None, :]
    else:
        G0_bat = G0_bat_.copy()
        for i in range(N):
            p = Ps[i]
            for j in range(i + 1, N):
                q = Ps[j]
                G0 = G0_bat_[i,j]

                # Check marginals of G0 and save
                # G0 = G0_bat_[i,j]
                np.testing.assert_allclose(G0.sum(axis=1), p, atol=1e-08)
                np.testing.assert_allclose(G0.sum(axis=0), q, atol=1e-08)
                G0_bat[i,j] = G0

    # cg for GW is implemented using numpy on CPU
    np_ = NumpyBackend()

    # ------------------------
    # Decompose cost matrices
    # ------------------------
    lconstC_bat, lhC1_bat, lhC2_bat = init_matrix_batch(
        lCs, Ps, loss_fun, np_, option=option
    )

    if not symmetric:
        lCst = nx.zeros(N, dtype=object)
        for i in range(N):
            lCst[i] = nx.transpose(lCs[i], (0, 2, 1))

        lconstCt_bat, lhC1t_bat, lhC2t_bat = init_matrix_batch(
            lCst, Ps, loss_fun, np_, option=option
        )

    # -----------------------------
    # Cost functions and gradients
    # -----------------------------
    fs = np.zeros((N, N), dtype=object)
    dfs = np.zeros((N, N), dtype=object)
    for i in range(N):
        for j in range(N):
            # Weighted distortion
            def f(G, nu):
                lS = gwloss_list(lconstC_bat[i, j], lhC1_bat[i], lhC2_bat[j], G, np_)
                return nx.dot(lS, nu)

            fs[i, j] = f

            if symmetric:
                # Gradient
                def df(G, nu):
                    # Shape: (T, n1, n2)
                    lgrad = gwggrad_list(
                        lconstC_bat[i, j], lhC1_bat[i], lhC2_bat[j], G, np_
                    )

                    # This gives sum(nu[i]*lgrad[i,:,:])
                    return nx.dot(nx.transpose(lgrad, (1, 2, 0)), nu)

            else:

                def df(G, nu):
                    # Shape: (T, n1, n2)
                    lgrad = gwggrad_list(
                        lconstC_bat[i, j], lhC1_bat[i], lhC2_bat[j], G, np_
                    )
                    lgradt = gwggrad_list(
                        lconstCt_bat[i, j], lhC1t_bat[i], lhC2t_bat[j], G, np_
                    )
                    lgrad_sym = lgrad + lgradt

                    # This gives sum(nu[i]*lgrad[i,:,:])
                    return 0.5 * nx.dot(nx.transpose(lgrad_sym, (1, 2, 0)), nu)

            dfs[i, j] = df

    # ----------------------------
    # Define line search function
    # ----------------------------
    line_searches = np.zeros((N, N), dtype=object)

    for i in range(N):
        for j in range(N):
            if armijo:
                raise NotImplementedError
            else:

                def line_search(cost, G, dG, Mi, cost_G, **kwargs):
                    # shape G: (n1,n2)
                    # shape dG: (n1,n2)
                    # shape cost_G: scalar

                    # The middle return of solve_gromov_linesearch is not used
                    alpha, fs, cost_G = solve_gromov_linesearch_multiscale(
                        G,
                        dG,
                        cost_G,
                        lconstC_bat[i, j],
                        lhC1_bat[i],
                        lhC2_bat[j],
                        nu,
                        M=0.0,
                        reg=1.0,
                        nx=np_,
                        **kwargs,
                    )

                    return alpha, fs, cost_G

            line_searches[i, j] = line_search

    # -----------------------------
    # Gradient descent on nu
    # -----------------------------
    q = unif(nSteps)

    # Define linear constraint (sum(nu)-1=0)
    eq_cons = {
        "type": "eq",
        "fun": lambda x: np.array([np.sum(x) - 1]),
        "jac": lambda x: np.ones_like(x),
    }

    # Define bounds (0 <= nu <= 1)
    bounds = Bounds([0] * nSteps, [1] * nSteps)

    def reg_score(dMS_mat, nu, reg=lambda_S):
        if np.max(dMS_mat) == 0:
            return np.inf

        # Returns the score S with a regularization term for nu
        if reg == 0:
            return S(dMS_mat)
        else:
            return S(dMS_mat) + reg * entropy(nu, q)

    def grad_score(dMS_mat, ldMS_mat, nu, reg=lambda_S):
        # Shape dMS_mat: (N, N)
        # Shape ldMS_mat: (N, N, T)
        # If . is composition, the chain rule states:
        # J_nu(S.D) = J_{D(nu)}(S)*J_nu(D)
        # Shape: 1 x T = 1 x (N*N) * (N*N)xT

        # Compute gradient of score
        JS = np.expand_dims(dS(dMS_mat), 0)  # shape 1 x (N*N)
        JD = ldMS_mat  # grad(dot(distortions, nu)) = distortions

        # Chain rule
        J_comp = np.tensordot(JS, JD, axes=2)

        if reg == 0:
            return J_comp.flatten()
        else:
            # Compute gradient with KL regularization
            # grad_KL = 1 + np.log10(nu + epsilon_*(nu==0))
            grad_KL = 1 + np.log10(nu + 1e-15*(nu==0))
            # grad_KL = 1 + np.log10(nu + 1e-15)
            return J_comp.flatten() + reg * grad_KL

    # Combine score and gradient into a single function
    # min_log tells us to minimize the log instead
    def S_and_dS(nu, ldMS_mat, min_log):
        # Weighted sum of distortions with weight nu
        dMS_mat = np.dot(ldMS_mat, nu)

        # Compute score
        score = reg_score(dMS_mat, nu)

        # Compute gradient
        grad = grad_score(dMS_mat, ldMS_mat, nu)

        if min_log:
            return (nx.log(score), grad / score)
        else:
            return (score, grad)

    # -----------------------------
    # Loop preparations
    # -----------------------------
    it = 0
    loop = 1

    G_bat = G0_bat.copy()  # shape: (N,N), shape G_bat[i,j]=(ni,nj)
    nu = nu.copy()  # (nSteps,)

    # Shape result: (N,N)
    # result[i,j]: nx.sum( dis_G_bat[i,j,:]*nu) = scalar
    dMS_mat = nx.zeros((N, N))
    ldMS_mat = nx.zeros((N, N, nSteps))
    score = 0

    if verbose:
        print(
            "{:11s}|{:15s}|{:15s}|{:15s}".format(
                "(step) It.",
                "Score",
                "Relative delta",
                "Absolute delta",
            )
            + "\n"
            + "-" * 54
        )
        print("{:11d}|{:15e}|{:15e}|{:15e}".format(it, score, 0, 0))

    # -----------------------------
    # Alternating gradient descent
    # -----------------------------
    # Debug
    score_list = []
    abs_delta_list = []
    rel_delta_list = []
    nu_list = [nu]

    while loop:
        it += 1
        # print('it:', it)
        old_score = score

        # Calculate dMS_mat
        for i in range(N):
            for j in range(i, N):

                def f(G):
                    return fs[i, j](G, nu)

                def df(G):
                    return dfs[i, j](G, nu)
                
                if verbose and it==1:
                    time_start = time()

                res_ij, log_ij = cg(
                    Ps[i], # a
                    Ps[j], # b
                    0.0, # M
                    1.0, # reg
                    f, # f
                    df, # df
                    G0_bat[i, j], # G0
                    line_searches[i, j],
                    log=True,
                    numItermax=numItermaxEmd,
                    stopThr=tol_rel,
                    stopThr2=tol_abs,
                    **kwargs,
                )

                if verbose and it==1:
                    time_end = time()
                    print(f'Initial OT ({i+1},{j+1})/({N},{N}):', np.round(time_end-time_start,3))

                # Update couplings
                T_ij = nx.from_numpy(res_ij, type_as=lCs[0])
                G_bat[i, j] = T_ij
                G_bat[j, i] = T_ij.T
            
            if verbose and it==1:
                print()

        # ----- Optimize nu with S -----
        # Recompute distortion matrices
        # Shape: (N, N, T)
        ldMS_mat = gwloss_batch(lconstC_bat, lhC1_bat, lhC2_bat, G_bat, nx=np_)

        # Recompute scores
        dMS_mat = nx.dot(ldMS_mat, nu)
        score_inter = reg_score(dMS_mat, nu)

        abs_delta_score = abs(score_inter - old_score)
        rel_delta_score = abs_delta_score / score_inter if score_inter != 0 else np.nan
        if verbose:
            print(
                    " ( T) {:5d}|{:15e}|{:15e}|{:15e}".format(
                        it, score_inter, rel_delta_score, abs_delta_score
                    )
                )
        
        # Debug
        score_list.append(score_inter)
        abs_delta_list.append(abs_delta_score)
        rel_delta_list.append(rel_delta_score)

        # Perform the optimization
        res = minimize(
            S_and_dS,
            nu,
            method="SLSQP",
            # method="trust-constr",
            jac=True,
            constraints=[eq_cons],
            bounds=bounds,
            args=(ldMS_mat, False),
            # tol=1e-12,
            # options={"disp": False},
            # options={"ftol": 1e-9, "disp": True},
            options={"ftol": 1e-9, "disp": False},
        )
        nu = res.x

        # Recompute scores
        dMS_mat = nx.dot(ldMS_mat, nu)
        score = reg_score(dMS_mat, nu)

        # ----- Test convergence -----
        # abs_delta_score = abs(score - score_inter)
        abs_delta_score = abs(score - old_score)
        rel_delta_score = abs_delta_score / score if score != 0 else np.nan

        # Debug
        score_list.append(score)
        abs_delta_list.append(abs_delta_score)
        rel_delta_list.append(rel_delta_score)
        nu_list.append(nu)

        if rel_delta_score < tol_rel or abs_delta_score < tol_abs:
            loop = 0

        # ----- show results -----
        if verbose:
            if it % 20 == 0:
                print(
                    "{:11s}|{:15s}|{:15s}|{:15s}".format(
                        "(step) It.",
                        "Score",
                        "Relative delta",
                        "Absolute delta",
                    )
                    + "\n"
                    + "-" * 54
                )
            print(
                " (nu) {:5d}|{:15e}|{:15e}|{:15e}".format(
                    it, score, rel_delta_score, abs_delta_score
                )
            )

        # ----- end the loop ----
        # If we exceeded max_iter
        if it >= max_iter:
            loop = 0

    # -----------------------------
    # Gradient descent finished
    # -----------------------------
    # Compute final set of distances and scores
    dMS_mat = nx.dot(ldMS_mat, nu)
    score = S(dMS_mat)

    return dMS_mat, G_bat, nu, score, (score_list, abs_delta_list, rel_delta_list, nu_list)
