import warnings
import torch

from .utils import init_matrix, gwloss, gwggrad
from .sinkhorn import sinkhorn


def fgw(
    M,
    C1,
    C2,
    p=None,
    q=None,
    loss_fun="square_loss",
    epsilon=0.1,
    symmetric=None,
    alpha=0.5,
    G0=None,
    max_iter=100,
    tol=1e-5,
    solver="PGD",
    method="sinkhorn_log",
    warmstart=False,
    verbose=False,
    log=False,
    **kwargs,
):
    if solver in ["PGD", "PPA"]:
        return fgw_projected(
            M,
            C1,
            C2,
            p=p,
            q=q,
            loss_fun=loss_fun,
            epsilon=epsilon,
            symmetric=symmetric,
            alpha=alpha,
            G0=G0,
            max_iter=max_iter,
            tol=tol,
            solver=solver,
            method=method,
            warmstart=warmstart,
            verbose=verbose,
            log=log,
            **kwargs,
        )
    elif solver == "BAPG":
        return fgw_bregman(
            M,
            C1,
            C2,
            p=p,
            q=q,
            loss_fun=loss_fun,
            epsilon=epsilon,
            symmetric=symmetric,
            alpha=alpha,
            G0=G0,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            log=log,
        )
    else:
        raise ValueError("Unknown solver '%s'. Pick one in ['PGD', 'PPA', 'BAPG']." % solver)


def fgw_projected(
    M,
    C1,
    C2,
    p=None,
    q=None,
    loss_fun="square_loss",
    epsilon=0.1,
    symmetric=None,
    alpha=0.5,
    G0=None,
    max_iter=100,
    tol=1e-5,
    solver="PGD",
    method="sinkhorn_log",
    warmstart=False,
    verbose=False,
    log=False,
    **kwargs,
):
    if solver not in ["PGD", "PPA"]:
        raise ValueError("Unknown solver '%s'. Pick one in ['PGD', 'PPA']." % solver)

    if loss_fun not in ("square_loss", "kl_loss"):
        raise ValueError(
            f"Unknown `loss_fun='{loss_fun}'`. Use one of: {'square_loss', 'kl_loss'}."
        )

    if G0 is None:
        G0 = torch.outer(p, q)

    T = G0
    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)
    if symmetric is None:
        symmetric = torch.allclose(C1, C1.T, atol=1e-10) and torch.allclose(C2, C2.T, atol=1e-10)
    if not symmetric:
        constCt, hC1t, hC2t = init_matrix(C1.T, C2.T, p, q, loss_fun)
    cpt = 0
    err = 1

    if warmstart:
        # initialize potentials to cope with ot.sinkhorn initialization
        N1, N2 = C1.shape[0], C2.shape[0]
        mu = torch.zeros(N1).to(C1) - torch.log(N1)
        nu = torch.zeros(N2).to(C2) - torch.log(N2)

    if log:
        log = {"err": []}

    while err > tol and cpt < max_iter:

        Tprev = T

        # compute the gradient
        if symmetric:
            tens = alpha * gwggrad(constC, hC1, hC2, T) + (1 - alpha) * M
        else:
            tens = (alpha * 0.5) * (
                gwggrad(constC, hC1, hC2, T) + gwggrad(constCt, hC1t, hC2t, T)
            ) + (1 - alpha) * M

        if solver == "PPA":
            tens = tens - epsilon * torch.log(T)

        if warmstart:
            T, loginn = sinkhorn(
                p, q, tens, epsilon, method=method, log=True, warmstart=(mu, nu), **kwargs
            )
            mu = epsilon * torch.log(loginn["u"])
            nu = epsilon * torch.log(loginn["v"])

        else:
            T = sinkhorn(p, q, tens, epsilon, method=method, **kwargs)

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = torch.norm(T - Tprev)

            if log:
                log["err"].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                print("{:5d}|{:8e}|".format(cpt, err))

        cpt += 1

    if abs(torch.sum(T) - 1) > 1e-5:
        warnings.warn(
            "Solver failed to produce a transport plan. You might want to increase the regularization parameter `epsilon`."
        )
    if log:
        log["fgw_dist"] = (1 - alpha) * torch.sum(M * T) + alpha * gwloss(constC, hC1, hC2, T)
        return T, log
    else:
        return T


def fgw_bregman(
    M,
    C1,
    C2,
    p=None,
    q=None,
    loss_fun="square_loss",
    epsilon=0.1,
    symmetric=None,
    alpha=0.5,
    G0=None,
    max_iter=1000,
    tol=1e-9,
    marginal_loss=False,
    verbose=False,
    log=False,
):
    """Solves the fused Gromov-Wasserstein problem with Bregman projections."""

    if loss_fun not in ("square_loss", "kl_loss"):
        raise ValueError(
            f"Unknown `loss_fun='{loss_fun}'`. Use one of: {'square_loss', 'kl_loss'}."
        )

    if G0 is None:
        G0 = torch.outer(p, q)

    T = G0
    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)
    if symmetric is None:
        symmetric = torch.allclose(C1, C1.T, atol=1e-10) and torch.allclose(C2, C2.T, atol=1e-10)
    if not symmetric:
        constCt, hC1t, hC2t = init_matrix(C1.T, C2.T, p, q, loss_fun)

    # Define gradients
    if marginal_loss:
        if symmetric:

            def df(T):
                return alpha * gwggrad(constC, hC1, hC2, T) + (1 - alpha) * M

        else:

            def df(T):
                return (alpha * 0.5) * (
                    gwggrad(constC, hC1, hC2, T) + gwggrad(constCt, hC1t, hC2t, T)
                ) + (1 - alpha) * M

    else:
        if symmetric:

            def df(T):
                A = -hC1 @ T @ hC2.T
                return 2 * alpha * A + (1 - alpha) * M

        else:

            def df(T):
                A = -hC1 @ T @ hC2.T
                At = -hC1t @ T @ hC2t.T
                return alpha * (A + At) + (1 - alpha) * M

    cpt = 0
    err = 1e15

    if log:
        log = {"err": []}

    while err > tol and cpt < max_iter:

        Tprev = T

        # rows update
        T = T * torch.exp(-df(T) / epsilon)
        row_scaling = p / torch.sum(T, 1)
        T = torch.reshape(row_scaling, (-1, 1)) * T

        # columns update
        T = T * torch.exp(-df(T) / epsilon)
        column_scaling = q / torch.sum(T, 0)
        T = torch.reshape(column_scaling, (1, -1)) * T

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = torch.norm(T - Tprev)

            if log:
                log["err"].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                print("{:5d}|{:8e}|".format(cpt, err))

        cpt += 1

    if torch.any(torch.isnan(T)):
        warnings.warn(
            "Solver failed to produce a transport plan. You might want to increase the regularization parameter `epsilon`."
        )
    if log:
        log["fgw_dist"] = (1 - alpha) * torch.sum(M * T) + alpha * gwloss(constC, hC1, hC2, T)

        if not marginal_loss:
            log["loss"] = log["fgw_dist"] - alpha * torch.sum(constC * T)

        return T, log
    else:
        return T
