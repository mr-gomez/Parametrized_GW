# Credit:
# The functions on this file are based on the function ot.gromov.gromov_wasserstein of the POT package
# TODO: Improve citation and documentation

import numpy as np
from scipy.special import binom
from scipy.stats import entropy
from scipy.optimize import minimize, Bounds

from ot.gromov import gromov_wasserstein
from ot.utils import list_to_array
from ot.optim import cg, line_search_armijo, solve_1d_linesearch_quad
from ot.lp import emd
from ot.utils import unif
from ot.backend import get_backend, NumpyBackend

from ot.gromov._utils import init_matrix, gwloss, gwggrad
from utils.utils_ms import *
from utils.utils import sq_idx_fun, sq_to_idx_fun, display_time

from itertools import combinations, chain

# Machine epsilon
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
    """
    Compute the parametrized GW distance between two pm-nets over a common parameter space.

    The function solves the following optimization problem using Conditional Gradient:

    .. math::
        \mathbf{T}^* \in \mathop{\arg \min}_\mathbf{T} \quad \sum_t \left(\sum_{i,j,k,l}
        L(\mathbf{C_{1,t}}_{i,k}, \mathbf{C_{2,t}}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l} \right) \nu_t

        s.t. \
            \mathbf{T} \mathbf{1} &= \mathbf{p}

            \mathbf{T}^T \mathbf{1} &= \mathbf{q}

            \mathbf{T} &\geq 0

    Where :
    - :math:`\mathbf{C_{1,t}}`: Sequence of cost matrices in the source space
    - :math:`\mathbf{C_{2,t}}`: Sequence of cost matrices in the target space
    - :math:`\mathbf{p}`: distribution in the source space
    - :math:`\mathbf{q}`: distribution in the target space
    - `L`: loss function to account for the misfit between the similarity matrices

    .. note:: This implementation is a generalization of ot.gromov.gromov_wasserstein
              by Erwan Vautier,  Nicolas Courty, Rémi Flamary, Titouan Vayer,
              and Cédric Vincent-Cuaz

    Parameters
    ----------
    lC1 : list of arrays of shape (n,n) or array of shape (nSteps, n, n)
        List of cost matrices in the source pm-net. Alternatively,
        an ndarray where lC1[t,:,:] is a cost matrix for each t.
    lC1 : list of arrays of shape (m,m) or array of shape (nSteps, m, m)
        List of cost matrices in the target pm-net. Alternatively,
        an ndarray where lC1[t,:,:] is a cost matrix for each t.
    p : array-like, optional
        Distribution in the source space.
        If None, we use the uniform distribution on n points.
    q : array-like, optional
        Distribution in the target space.
        If None, we use the uniform distribution on m points.
    nu : array-like, optional
        Distribution on the parameter space (weights over the list of cost matrices).
        If None, we use the uniform distribution on nSteps points.
    loss_fun : str, optional
        loss function used for the solver either 'square_loss' or 'kl_loss'
        NOTE: Only 'square_loss' is currently implemented.
    symmetric : bool, optional
        Assume lC1[t] and lC2[t] are all symmetric or not.
        If symmetric is None, a symmetry test will be conducted.
        Else if set to True (resp. False), each lC1[t] and lC2[t] will be assumed symmetric (resp. asymmetric).
    log : bool, optional
        record log if True. Default is False.
    armijo : bool, optional
        If True the step of the line-search is found via an armijo research. Else closed form is used.
        If there are convergence issues use False. Default is False.
        NOTE: armijo is currently not implmemented.
    G0 : ndarray, optional
        Initial transport plan of the solver. If None, we use pq^T.
        Otherwise G0 must satisfy marginal constraints.
    max_iter : int, optional
        Max number of iterations. Default is 1e4.
    tol_rel : float, optional
        Stop threshold on relative error. Default is 1e-9.
    tol_abs : float, optional
        Stop threshold on absolute error. Default is 1e-9.
    **kwargs : dict
        parameters can be directly passed to the ot.optim.cg solver

    Returns
    -------
    G : ndarray
        Optimal coupling of shape (n, m).
    log : dict, optional
        Convergence information and loss.

    Raises
    ------
    NotImplementedError
        If loss_fun='kl_loss' or armijo=True (not yet supported).

    References
    ----------
    .. Mario Gómez, Guanqun Ma, Tom Needham, and Bei Wang.
        "Metrics for Parametric Families of Networks."
        arxiv:2509.22549. 2025.

    .. Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    .. Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain
        and Courty Nicolas "Optimal Transport for structured data with
        application on graphs", International Conference on Machine Learning
        (ICML). 2019.
    """
    nSteps = len(lC1)

    # Make sure each matrix in the pm-nets is an array
    lC1 = list_to_array(*lC1)
    lC2 = list_to_array(*lC2)

    # Initialize variables, or add the ones we have to arr
    # We convert them to NumPy arrays later
    arr = [*lC1, *lC2]
    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(lC1[0].shape[0], type_as=lC1[0])
    if q is not None:
        arr.append(list_to_array(q))
    else:
        q = unif(lC2[0].shape[0], type_as=lC2[0])
    if G0 is not None:
        G0_ = G0
        arr.append(G0)
    if nu is not None:
        arr.append(nu)
    else:
        nu = unif(len(lC1))

    # Get common backend of the variables
    nx = get_backend(*arr)
    p0, q0, lC10, lC20 = p, q, lC1, lC2

    # Convert variables to Numpy
    p = nx.to_numpy(p0)
    q = nx.to_numpy(q0)
    lC1 = nx.to_numpy(lC10)
    lC2 = nx.to_numpy(lC20)

    # Check symmetry of the elements of lC1 and lC2
    if symmetric is None:
        symmetric = True
        for i in range(nSteps):
            symmetric = symmetric and np.allclose(lC1[i], lC1[i].T, atol=1e-10)
            symmetric = symmetric and np.allclose(lC2[i], lC2[i].T, atol=1e-10)

    # Initialize an initial guess for a correspondence
    if G0 is None:
        G0 = p[:, None] * q[None, :]
    # Otherwise, make sure the provided G0 is a Numpy array and that it has the
    # correct marginals
    else:
        G0 = nx.to_numpy(G0_)
        np.testing.assert_allclose(G0.sum(axis=1), p, atol=1e-08)
        np.testing.assert_allclose(G0.sum(axis=0), q, atol=1e-08)

    # cg for GW is implemented using numpy on CPU
    np_ = NumpyBackend()

    # ------------------------
    # Decompose cost matrices
    # ------------------------
    lconstC, lhC1, lhC2 = init_matrix_list(lC1, lC2, p, q, loss_fun, np_, option=0)

    # f(G) is the parametrized GW cost of a coupling G
    def f(G):
        lS = [gwloss(lconstC[i], lhC1[i], lhC2[i], G, np_) for i in range(nSteps)]
        return nx.dot(nu, np.array(lS))

    # df(G) is the gradient of the parametrized GW cost at G
    if symmetric:

        def df(G):
            S = nx.zeros((*G.shape, nSteps))
            for i in range(nSteps):
                S[:, :, i] = gwggrad(lconstC[i], lhC1[i], lhC2[i], G, np_)
            return nx.dot(S, nu)

    # The gradient is different if the pm-net is not symmetric
    else:
        # We need to decompose the transpose of the cost matrices
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
                S[:, :, i] += gwggrad(lconstCt[i], lhC1t[i], lhC2t[i], G, np_)
            return 0.5 * nx.dot(S, nu)

    # Skip kl_loss for now
    if loss_fun == "kl_loss":
        raise NotImplementedError
        armijo = True  # there is no closed form line-search with KL

    # line_search for gradient descent
    # Skip armijo for now
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

    # Execute gradient descent
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
    # Obtain backend if it wasn't provided
    if nx is None:
        G, deltaG, M = list_to_array(G, deltaG, M)

        # Each element of these lists must be an array
        lconstC = list_to_array(*lconstC)
        lhC1 = list_to_array(*lhC1)
        lhC2 = list_to_array(*lhC2)

        if isinstance(M, int) or isinstance(M, float):
            nx = get_backend(G, deltaG, lconstC, lhC1, lhC2)
        else:
            nx = get_backend(G, deltaG, lconstC, lhC1, lhC2, M)

    # Exact solution of the line search step
    nSteps = len(lhC1)
    a_vec = nx.zeros(nSteps)
    b_vec = nx.zeros(nSteps)
    for i in range(nSteps):
        dot_dG = nx.dot(nx.dot(lhC1[i], deltaG), lhC2[i].T)
        dot_G = nx.dot(nx.dot(lhC1[i], G), lhC2[i].T)

        # NOTE: In contrast to solve_gromov_linesearch, we do not multiply reg
        # by 2 (in b) because lhC2 has an extra factor of 2 compared to lC2.
        # NOTE 2: it seems the above note is outdated?
        a_vec[i] = -reg * nx.sum(dot_dG * deltaG)
        b_vec[i] = nx.sum(M * deltaG) - reg * (
            nx.sum(dot_dG * G) + nx.sum(dot_G * deltaG)
        )

    a = nx.dot(nu, a_vec)
    b = nx.dot(nu, b_vec)

    alpha = solve_1d_linesearch_quad(a, b)
    if alpha_min is not None or alpha_max is not None:
        alpha = np.clip(alpha, alpha_min, alpha_max)

    # The new cost is deducted from the line search quadratic function
    cost_G = cost_G + a * (alpha**2) + b * alpha

    return alpha, 1, cost_G


############################
# Matching parameter spaces
############################
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
    """
    Compute the parametrized GW distance between two pm-nets over different parameter spaces.

    The function solves the two-variable optimization problem using an alternating
    optimization and Conditional Gradient:

    .. math::
        \mathbf{T}^*, \mathbf{S}^* \in \mathop{\arg \min}_{\mathbf{T}, \mathbf{S}} \quad
        \sum_{t,s} \left(\sum_{i,j,k,l} L(\mathbf{C_{1,t}}_{i,k}, \mathbf{C_{2,s}}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l} \right) \mathbf{S}_{t,s}

        s.t. \
            \mathbf{T} \mathbf{1} &= \mathbf{p}

            \mathbf{T}^T \mathbf{1} &= \mathbf{q}

            \mathbf{S} \mathbf{1} &= \mathbf{nu}_1

            \mathbf{S}^T \mathbf{1} &= \mathbf{nu}_2

            \mathbf{T} &\geq 0

            \mathbf{S} &\geq 0

    Where :
    - :math:`(\mathbf{C_{1,t}})_t`: Sequence of cost matrices in the source space
    - :math:`(\mathbf{C_{2,s}})_s`: Sequence of cost matrices in the target space
    - :math:`\mathbf{p}`: distribution in the source space
    - :math:`\mathbf{q}`: distribution in the target space
    - :math:`\nu_1`: distribution in the source parameter space
    - :math:`\nu_2`: distribution in the target parameter space
    - `L`: loss function to account for the misfit between the similarity matrices

    .. note:: This implementation is a generalization of ot.gromov.gromov_wasserstein
              by Erwan Vautier,  Nicolas Courty, Rémi Flamary, Titouan Vayer,
              and Cédric Vincent-Cuaz

    Parameters
    ----------
    lC1 : list of arrays of shape (n1,n1) or array of shape (nSteps1, n1, n1)
        List of cost matrices in the source pm-net. Alternatively,
        an ndarray where lC1[t,:,:] is a cost matrix for each t.
    lC1 : list of arrays of shape (n2,n2) or array of shape (nSteps2, n2, n2)
        List of cost matrices in the target pm-net. Alternatively,
        an ndarray where lC1[t,:,:] is a cost matrix for each t.
    p : array-like, optional
        Distribution in the source space.
        If None, we use the uniform distribution on n1 points.
    q : array-like, optional
        Distribution in the target space.
        If None, we use the uniform distribution on n2 points.
    nu1 : array-like, optional
        Distribution on the source parameter space (weights over the list of cost matrices).
        If None, we use the uniform distribution on nSteps1 points.
    nu2 : array-like, optional
        Distribution on the target parameter space (weights over the list of cost matrices).
        If None, we use the uniform distribution on nSteps2 points.
    loss_fun : str, optional
        loss function used for the solver either 'square_loss' or 'kl_loss'
        NOTE: Only 'square_loss' is currently implemented.
    symmetric : bool, optional
        Assume lC1[t] and lC2[t] are all symmetric or not.
        If symmetric is None, a symmetry test will be conducted.
        Else if set to True (resp. False), each lC1[t] and lC2[t] will be assumed symmetric (resp. asymmetric).
    armijo : bool, optional
        If True the step of the line-search is found via an armijo research. Else closed form is used.
        If there are convergence issues use False. Default is False.
        NOTE: armijo is currently not implmemented.
    E0 : ndarray, optional
        Initial coupling between parameter spaces. If None, we use nu1*nu2^T.
        Otherwise E0 must satisfy marginal constraints.
    G0 : ndarray, optional
        Initial transport plan of the solver. If None, we use p*q^T.
        Otherwise G0 must satisfy marginal constraints.
    max_iter : int, optional
        Max number of iterations. Default is 1e4.
    numItermaxEmd : int, optional
        Max number of iterations in the internal call to emd. Default is 1e5.
    tol_rel : float, optional
        Stop threshold on relative error. Default is 1e-9.
    tol_abs : float, optional
        Stop threshold on absolute error. Default is 1e-9.
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True. Default is False.
    **kwargs : dict
        parameters can be directly passed to the ot.optim.cg solver

    Returns
    -------
    E : ndarray of shape (nSteps1, nSteps2)
        Optimal coupling between parameter spaces.
    G : ndarray of shape (n1, n2)
        Optimal coupling between pm-nets.
    log : dict, optional
        Convergence information and loss.
    innerlog_1 : dict, optional
        Convergence information and loss.
    innerlog_2 : dict, optional
        Convergence information and loss.

    Raises
    ------
    NotImplementedError
        If loss_fun='kl_loss' or armijo=True (not yet supported).

    References
    ----------
    .. Mario Gómez, Guanqun Ma, Tom Needham, and Bei Wang.
        "Metrics for Parametric Families of Networks."
        arxiv:2509.22549. 2025.

    .. Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    .. Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain
        and Courty Nicolas "Optimal Transport for structured data with
        application on graphs", International Conference on Machine Learning
        (ICML). 2019.
    """
    nSteps1 = len(lC1)
    nSteps2 = len(lC2)
    n1 = lC1[0].shape[0]
    n2 = lC2[0].shape[0]

    lC1 = list_to_array(lC1)
    lC2 = list_to_array(lC2)
    arr = [lC1, lC2]

    # ----------------
    # Default choices
    # ----------------
    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(lC1[0].shape[0])
    if q is not None:
        arr.append(list_to_array(q))
    else:
        q = unif(lC2[0].shape[0])
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

        # Calculate matrix of costs
        cost_mat = f_mat(G)

        # ----- Optimize E -----
        E, innerlog_1 = emd(nu1, nu2, cost_mat, numItermax=numItermaxEmd, log=True)
        # print(E)
        # print()

        # ----- Optimize G -----
        # Find cost and gradient with the new E and the old G
        def f_E(G_E):
            return f(E, f_mat(G_E))

        def df_E(G_E):
            return df(E, df_mat(G_E))

        def line_search_E(cost, G_E, deltaG_E, Mi, cost_G_E, **kwargs):
            return line_search(E, G_E, deltaG_E, cost_G_E)

        res, innerlog_2 = cg(
            p,
            q,
            0.0,  # M
            1.0,  # reg
            f_E,  # f
            df_E,  # df
            G,  # G0
            line_search_E,
            log=True,
            numItermax=numItermaxEmd,
            stopThr=tol_rel,
            stopThr2=tol_abs,
            **kwargs,
        )
        G = nx.from_numpy(res, type_as=lC1[0])

        # Recompute cost
        cost_G = f_E(G)

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
        return G, E, log, innerlog_1, innerlog_2
    else:
        return G, E


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
def gw_ms_learn_nu(
    lCs,
    Ps=None,
    S=None,  # Score function
    dS=None,  # Gradient of score
    lambda_S=1e-5,  # Regularization for KL divergence of nu
    loss_fun="square_loss",
    symmetric=None,
    armijo=False,
    nu=None,
    G0_bat=None,
    max_iter=1e4,
    numItermaxEmd=1e5,
    tol_rel=1e-9,
    tol_abs=1e-9,
    verbose=False,
    **kwargs,
):
    """
    Find the weights that minimize a function of the parametrized GW distance between a set of pm-nets.

    This function computes the parametrized GW distance between a set of pm-nets,
    and finds the weights on the common parameter space that minimize a provided
    cost function. Explicitly, we find the matrix D of parametrized-GW
    distance matrices and find the nu that minimizes a cost S(D, nu).

    .. note:: This implementation is a generalization of ot.gromov.gromov_wasserstein
              by Erwan Vautier,  Nicolas Courty, Rémi Flamary, Titouan Vayer,
              and Cédric Vincent-Cuaz

    Parameters
    ----------
    lCs : list of arrays of shape (nSteps, n_t, n_t)
        List of pm-net represented by 3D arrays. Each array must have
        shape (nSteps, n_t, n_t).
    Ps : list of arrays of shape (n_t,)
        List of distributions of the pm-nets. Length must be len(lCs).
        If None, we use the uniform distributions on n_t, for every t.
    S : callable
        Cost function to minimize.
        Must accept a square matrix of shape len(lCs) and return a float.
    dS : callable
        Gradient of the cost function.
        Must accept a square matrix of shape len(lCs) and return an
        array of the same shape.
    lambda_S :
        Regularization constant for the KL divergence of nu.
        Default is 1e-5.
    loss_fun : str, optional
        loss function used for the solver either 'square_loss' or 'kl_loss'
        NOTE: Only 'square_loss' is currently implemented.
    symmetric : bool, optional
        Assume each lCs[idx] is symmetric or not.
        If symmetric is None, a symmetry test will be conducted.
        Else if set to True (resp. False), each lCs[idx] will be assumed symmetric (resp. asymmetric).
    armijo : bool, optional
        If True the step of the line-search is found via an armijo research. Else closed form is used.
        If there are convergence issues use False. Default is False.
        NOTE: armijo is currently not implmemented.
    nu : array of shape (len(lCs),), optional
        Distribution on the parameter space (weights over the list of cost matrices).
        If None, we use the uniform distribution on nSteps points.
    E0 : array of shape (len(lCs), len(lCs)), optional
        Initial coupling between parameter spaces. If None, we use nu1*nu2^T.
        Otherwise E0 must satisfy marginal constraints.
    G0_bat : array of shape (len(lCs), len(lCs))
        Contains the initial transport plan for each pair of pm-nets.
        If None, we use Ps[idx]*Ps[jdx].T.
        Otherwise each G0_bat[idx,jdx] must have shape (len(lCs[idx]), len(lCs[jdx]))
        and satisfy marginal constraints.
    max_iter : int, optional
        Max number of iterations. Default is 1e4.
    numItermaxEmd : int, optional
        Max number of iterations in the internal call to emd. Default is 1e5.
    tol_rel : float, optional
        Stop threshold on relative error. Default is 1e-9.
    tol_abs : float, optional
        Stop threshold on absolute error. Default is 1e-9.
    verbose : bool, optional
        Print information along iterations
    **kwargs : dict
        parameters can be directly passed to the ot.optim.cg solver

    Returns
    -------
    dMS_mat : ndarray of shape (len(lCs), len(lCs))
        Matrix with the parametrized GW distance between each pair of pm-nets
    G_bat : ndarray of shape (len(lCs), len(lCs))
        Array of optimal couplings for each pair of pm-nets
    nu : array of shape (nSteps,)
        Learned weights that minimize the score function S.
    score : float
        The minimum score found.
    log : tuple
        Convergence information and loss with the following variables:
        score_list, abs_delta_list, rel_delta_list, nu_list.

    Raises
    ------
    NotImplementedError
        If loss_fun='kl_loss' or armijo=True (not yet supported).

    References
    ----------
    .. Mario Gómez, Guanqun Ma, Tom Needham, and Bei Wang.
        "Metrics for Parametric Families of Networks."
        arxiv:2509.22549. 2025.

    .. Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    .. Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain
        and Courty Nicolas "Optimal Transport for structured data with
        application on graphs", International Conference on Machine Learning
        (ICML). 2019.
    """
    # NOTE:
    # option=0: Work with decoupled constants. Need to add them during runtime (slower)
    # option=1: Sum constants, return n*n array of constants (more memory)
    option = 1

    # lCs is a list of multiscale networks of the same length
    N = len(lCs)
    nSteps = lCs[0].shape[0]
    arr = [*lCs]

    # ----------------
    # Default choices
    # ----------------
    if Ps is not None:
        arr.extend(list_to_array(*Ps))
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
                G0 = G0_bat_[i, j]

                # Check marginals of G0 and save
                # G0 = G0_bat_[i,j]
                np.testing.assert_allclose(G0.sum(axis=1), p, atol=1e-08)
                np.testing.assert_allclose(G0.sum(axis=0), q, atol=1e-08)
                G0_bat[i, j] = G0

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
            grad_KL = 1 + np.log10(nu + 1e-15 * (nu == 0))
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

                if verbose and it == 1:
                    time_start = time()

                res_ij, log_ij = cg(
                    Ps[i],  # a
                    Ps[j],  # b
                    0.0,  # M
                    1.0,  # reg
                    f,  # f
                    df,  # df
                    G0_bat[i, j],  # G0
                    line_searches[i, j],
                    log=True,
                    numItermax=numItermaxEmd,
                    stopThr=tol_rel,
                    stopThr2=tol_abs,
                    **kwargs,
                )

                if verbose and it == 1:
                    time_end = time()
                    print(
                        f"Initial OT ({i+1},{j+1})/({N},{N}):",
                        np.round(time_end - time_start, 3),
                    )

                # Update couplings
                T_ij = nx.from_numpy(res_ij, type_as=lCs[0])
                G_bat[i, j] = T_ij
                G_bat[j, i] = T_ij.T

            if verbose and it == 1:
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

    return (
        dMS_mat,
        G_bat,
        nu,
        score,
        (score_list, abs_delta_list, rel_delta_list, nu_list),
    )
