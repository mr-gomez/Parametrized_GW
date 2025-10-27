import numpy as np
import ot
import warnings
import random
import itertools
import networkx as nx
from ot.backend import get_backend, NumpyBackend
from ot.utils import list_to_array, unif
from ot.gromov import init_matrix, gwloss, gwggrad, solve_gromov_linesearch
from ot.optim import cg
from ot.bregman import sinkhorn

##################################### Main Function ########################################

def gromov_wasserstein_on_sets(
    S1, S2,
    p, q,
    nu1=None, nu2=None,
    loss_fun="square_loss",
    symmetric=None,
    armijo=False,
    # random‐init options
    random_init=False,
    random_seed=None,
    random_init_iter=10,
    # multi‐start option
    multiple_random_inits=1,
    # inner GW‐solver tolerances
    inner_max_iter=1_000,
    inner_tol_rel=1e-9,
    inner_tol_abs=1e-9,
    # outer alternating loop controls
    outer_max_iter=20,
    tol_outer=1e-6,
    ot_method="sinkhorn",
    sinkhorn_reg=1e-3,
    verbose=False
):
    """
    Returns:
      G : (n_s × n_t) coupling
      Q : (n1 × n2) coupling
      cost : float, the final objective
    """
    n1, n2 = len(S1), len(S2)
    n_s, n_t = S1[0].shape[0], S2[0].shape[0]
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    nu1 = np.asarray(nu1) if nu1 is not None else np.ones(n1)/n1
    nu2 = np.asarray(nu2) if nu2 is not None else np.ones(n2)/n2

    # --- multi‐start wrapper ---
    if multiple_random_inits and multiple_random_inits > 1:
        best_obj = np.inf
        best_G, best_Q = None, None

        # precompute gw_data for cost evaluation
        npb = NumpyBackend()
        gw_data = {
            (k,l): init_matrix(S1[k], S2[l], p, q, loss_fun, npb)
            for k in range(n1) for l in range(n2)
        }

        for run in range(multiple_random_inits):
            seed_i = (random_seed + run) if random_seed is not None else None
            G_i, Q_i, _ = gromov_wasserstein_on_sets(
                S1, S2, p, q,
                nu1=nu1, nu2=nu2,
                loss_fun=loss_fun,
                symmetric=symmetric,
                armijo=armijo,
                random_init=True,
                random_seed=seed_i,
                random_init_iter=random_init_iter,
                multiple_random_inits=1,
                inner_max_iter=inner_max_iter,
                inner_tol_rel=inner_tol_rel,
                inner_tol_abs=inner_tol_abs,
                outer_max_iter=outer_max_iter,
                tol_outer=tol_outer,
                ot_method=ot_method,
                sinkhorn_reg=sinkhorn_reg,
                verbose=False,
            )
            # compute cost_i
            A = np.zeros((n1, n2))
            for (k,l), (constC, hC1, hC2) in gw_data.items():
                A[k,l] = gwloss(constC, hC1, hC2, G_i, npb)
            cost_i = np.sum(A * Q_i)

            if cost_i < best_obj:
                best_obj, best_G, best_Q = cost_i, G_i.copy(), Q_i.copy()

        if verbose:
            print(f"Best cost over {multiple_random_inits} inits: {best_obj:.3e}")
        return best_G, best_Q, best_obj

    # --- single solve ---
    # initialize G
    if random_init:
        rng = np.random.default_rng(random_seed)
        R = rng.random((n_s, n_t))
        for _ in range(random_init_iter):
            R *= (p / R.sum(axis=1))[:, None]
            R *= (q / R.sum(axis=0))[None, :]
        G = R
    else:
        G = np.outer(p, q)

    # initialize Q
    Q = np.outer(nu1, nu2)

    # symmetry test
    if symmetric is None:
        sym1 = all(np.allclose(C, C.T, atol=1e-10) for C in S1)
        sym2 = all(np.allclose(C, C.T, atol=1e-10) for C in S2)
        symmetric_flag = sym1 and sym2
    else:
        symmetric_flag = symmetric

    npb = NumpyBackend()

    # precompute init_matrix
    gw_data   = {}
    gw_data_T = {}
    for k in range(n1):
        for l in range(n2):
            constC, hC1, hC2 = init_matrix(S1[k], S2[l], p, q, loss_fun, npb)
            gw_data[(k,l)] = (constC, hC1, hC2)
            if not symmetric_flag:
                Ct, hC1t, hC2t = init_matrix(S1[k].T, S2[l].T, p, q, loss_fun, npb)
                gw_data_T[(k,l)] = (Ct, hC1t, hC2t)

    def make_ls(h1, h2):
        if armijo:
            from ot.gromov import line_search_armijo
            return lambda cost, G_, dG, Mi, cG, dG_: \
                   line_search_armijo(cost, G_, dG, Mi, cG, nx=npb)
        else:
            return lambda cost, G_, dG, Mi, cG, dG_: \
                   solve_gromov_linesearch(
                       G_, dG, cG, h1, h2, M=0.0, reg=1.0, nx=npb,
                       symmetric=symmetric_flag
                   )

    # outer loop
    for it in range(outer_max_iter):
        # aggregate
        constC_agg = np.zeros_like(gw_data[(0,0)][0])
        hC1_agg    = np.zeros_like(gw_data[(0,0)][1])
        hC2_agg    = np.zeros_like(gw_data[(0,0)][2])
        if not symmetric_flag:
            Ct_agg   = np.zeros_like(gw_data_T[(0,0)][0])
            hC1t_agg = np.zeros_like(gw_data_T[(0,0)][1])
            hC2t_agg = np.zeros_like(gw_data_T[(0,0)][2])

        for k in range(n1):
            for l in range(n2):
                w = Q[k,l]
                cC, h1, h2 = gw_data[(k,l)]
                constC_agg += w * cC
                hC1_agg    += w * h1
                hC2_agg    += w * h2
                if not symmetric_flag:
                    cCt, h1t, h2t = gw_data_T[(k,l)]
                    Ct_agg     += w * cCt
                    hC1t_agg   += w * h1t
                    hC2t_agg   += w * h2t

        def f_agg(Gm): return gwloss(constC_agg, hC1_agg, hC2_agg, Gm, npb)
        if symmetric_flag:
            def df_agg(Gm): return gwggrad(constC_agg, hC1_agg, hC2_agg, Gm, npb)
        else:
            def df_agg(Gm):
                g1 = gwggrad(constC_agg,  hC1_agg,  hC2_agg,  Gm, npb)
                g2 = gwggrad(Ct_agg,       hC1t_agg, hC2t_agg, Gm, npb)
                return 0.5*(g1 + g2)

        ls = make_ls(hC1_agg, hC2_agg)
        G_prev = G.copy()
        G = cg(
            p, q,
            0.0, 1.0,
            f_agg, df_agg,
            G_prev,
            line_search=ls,
            log=False,
            numItermax=inner_max_iter,
            stopThr=inner_tol_rel,
            stopThr2=inner_tol_abs
        )

        # update Q
        A = np.zeros((n1, n2))
        for k in range(n1):
            for l in range(n2):
                cC, h1, h2 = gw_data[(k,l)]
                A[k,l] = gwloss(cC, h1, h2, G, npb)

        Q_prev = Q.copy()
        if ot_method == "sinkhorn":
            Q = sinkhorn(nu1, nu2, A, reg=sinkhorn_reg)
        else:
            Q = ot.emd(nu1, nu2, A)

        # convergence
        if verbose:
            obj = np.sum(A * Q)
            print(f"[outer {it+1}] ΔG={np.linalg.norm(G-G_prev):.2e}, ΔQ={np.linalg.norm(Q-Q_prev):.2e}, obj={obj:.3e}")
        if np.linalg.norm(G-G_prev) < tol_outer and np.linalg.norm(Q-Q_prev) < tol_outer:
            break

    # final cost
    A_final = np.zeros((n1, n2))
    for k in range(n1):
        for l in range(n2):
            cC, h1, h2 = gw_data[(k,l)]
            A_final[k,l] = gwloss(cC, h1, h2, G, npb)
    cost = float(np.sum(A_final * Q))

    return G, Q, cost

##################################### Adaptation of POT GW distance function ########################################

def gromov_wasserstein(
    C1,
    C2,
    p=None,
    q=None,
    loss_fun="square_loss",
    symmetric=None,
    log=False,
    armijo=False,
    # — new random-init args —
    random_init=False,
    random_seed=None,
    random_init_iter=10,
    multiple_random_inits=1,
    # — original args —
    G0=None,
    max_iter=1e4,
    tol_rel=1e-9,
    tol_abs=1e-9,
    **kwargs,
):
    r"""
    (Docstring as before, plus:)

    random_init : bool
        If True, initialize the coupling with a random matrix then IPF‐project to (p,q).
    random_seed : int or None
        Seed for the random initialization.
    random_init_iter : int
        Number of IPF iterations when random_init=True.
    multiple_random_inits : int
        If >1, run that many independent solves (each with its own random init)
        and return the coupling with the lowest final GW loss.
    """
    # 1) Backend & marginals setup (unchanged)
    arr = [C1, C2]
    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(C1.shape[0], type_as=C1)
    if q is not None:
        arr.append(list_to_array(q))
    else:
        q = unif(C2.shape[0], type_as=C1)
    if G0 is not None:
        arr.append(G0)

    nx = get_backend(*arr)
    p0, q0, C10, C20 = p, q, C1, C2

    # Convert to NumPy for the solver
    p_np = nx.to_numpy(p0)
    q_np = nx.to_numpy(q0)
    C1_np = nx.to_numpy(C10)
    C2_np = nx.to_numpy(C20)

    # 2) Symmetry test
    if symmetric is None:
        symmetric_flag = (
            np.allclose(C1_np, C1_np.T, atol=1e-10)
            and np.allclose(C2_np, C2_np.T, atol=1e-10)
        )
    else:
        symmetric_flag = bool(symmetric)

    # 3) Precompute GW ingredients
    npb = NumpyBackend()
    constC, hC1, hC2 = init_matrix(C1_np, C2_np, p_np, q_np, loss_fun, npb)
    if not symmetric_flag:
        constCt, hC1t, hC2t = init_matrix(
            C1_np.T, C2_np.T, p_np, q_np, loss_fun, npb
        )

    # 4) Define loss & gradient
    def f_np(G):
        return gwloss(constC, hC1, hC2, G, npb)

    if symmetric_flag:
        def df_np(G):
            return gwggrad(constC, hC1, hC2, G, npb)
    else:
        def df_np(G):
            g1 = gwggrad(constC,  hC1,   hC2,   G, npb)
            g2 = gwggrad(constCt,   hC1t,  hC2t,  G, npb)
            return 0.5 * (g1 + g2)

    # 5) Line‐search builder
    if armijo:
        from ot.gromov import line_search_armijo
        def make_ls():
            return lambda f, G, dG, M_i, cG, dG_: \
                line_search_armijo(f, G, dG, M_i, cG, nx=npb)
    else:
        def make_ls():
            return lambda f, G, dG, M_i, cG, dG_: \
                solve_gromov_linesearch(
                    G, dG, cG, hC1, hC2,
                    M=0.0, reg=1.0, nx=npb,
                    symmetric=symmetric_flag,
                    **kwargs
                )

    # 6) Helper to IPF‐project a random matrix to (p,q)
    def random_coupling(seed):
        rng = np.random.default_rng(seed)
        R = rng.random((p_np.size, q_np.size))
        for _ in range(random_init_iter):
            R *= (p_np / R.sum(axis=1))[:, None]
            R *= (q_np / R.sum(axis=0))[None, :]
        return R

    # 7) Core GW‐solve given an initial G0_np
    def solve_GW(G0_np, return_log):
        ls = make_ls()
        if return_log:
            res_np, log_dict = cg(
                p_np, q_np,
                0.0, 1.0,
                f_np, df_np,
                G0_np,
                line_search=ls,
                log=True,
                numItermax=int(max_iter),
                stopThr=tol_rel,
                stopThr2=tol_abs,
                **kwargs
            )
            # convert log’s final loss back into the original type
            log_dict["gw_dist"] = nx.from_numpy(log_dict["loss"][-1], type_as=C10)
            log_dict["u"] = nx.from_numpy(log_dict["u"], type_as=C10)
            log_dict["v"] = nx.from_numpy(log_dict["v"], type_as=C10)
            return res_np, log_dict
        else:
            res_np = cg(
                p_np, q_np,
                0.0, 1.0,
                f_np, df_np,
                G0_np,
                line_search=ls,
                log=False,
                numItermax=int(max_iter),
                stopThr=tol_rel,
                stopThr2=tol_abs,
                **kwargs
            )
            return res_np, None

    # 8) Multi‐start logic
    if multiple_random_inits > 1:
        best_cost = np.inf
        best_res = None
        best_log = None

        for run in range(multiple_random_inits):
            seed_run = (random_seed + run) if (random_seed is not None) else None

            # build this run’s init coupling
            if G0 is not None:
                G0_np = nx.to_numpy(G0)
            elif random_init:
                G0_np = random_coupling(seed_run)
            else:
                G0_np = p_np[:, None] * q_np[None, :]

            # solve
            res_np, log_dict = solve_GW(G0_np, return_log=log)
            # extract cost
            if log:
                cost_run = log_dict["loss"][-1]
            else:
                cost_run = f_np(res_np)

            if cost_run < best_cost:
                best_cost, best_res, best_log = cost_run, res_np, log_dict

        # convert best back to original type
        T_best = nx.from_numpy(best_res, type_as=C10)
        if best_log is not None:
            return T_best, best_log
        else:
            return T_best

    # 9) Single‐start branch
    if G0 is not None:
        G0_np = nx.to_numpy(G0)
    elif random_init:
        G0_np = random_coupling(random_seed)
    else:
        G0_np = p_np[:, None] * q_np[None, :]

    res_np, log_dict = solve_GW(G0_np, return_log=log)
    T = nx.from_numpy(res_np, type_as=C10)

    if log:
        return T, log_dict
    else:
        return T

##################################### Code for Generating Random Graphs ########################################

def perturb_graph_edges(G, k: int, m: int, seed: int = None) -> nx.Graph:
    """
    Return a copy of G with k random edges removed and m random non‐existing edges added.

    Parameters
    ----------
    G : networkx.Graph
        The original graph (undirected).
    k : int
        Number of edges to remove. If k > |E(G)|, all edges will be removed.
    m : int
        Number of edges to add. If m > number of possible new edges, as many as possible are added.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    G_new : networkx.Graph
        A new graph with k edges removed and m edges added.
    """
    if seed is not None:
        random.seed(seed)

    # Work on a copy
    G_new = G.copy()

    # 1) Remove k random edges
    existing_edges = list(G_new.edges())
    k_remove = min(k, len(existing_edges))
    edges_to_remove = random.sample(existing_edges, k_remove)
    G_new.remove_edges_from(edges_to_remove)

    # 2) Build list of all possible new edges (undirected, no self‐loops)
    nodes = list(G_new.nodes())
    existing = set(map(lambda e: tuple(sorted(e)), G_new.edges()))
    all_pairs = itertools.combinations(nodes, 2)
    possible_new = [pair for pair in all_pairs if pair not in existing]

    # 3) Add m random new edges
    m_add = min(m, len(possible_new))
    edges_to_add = random.sample(possible_new, m_add)
    G_new.add_edges_from(edges_to_add)

    return G_new
