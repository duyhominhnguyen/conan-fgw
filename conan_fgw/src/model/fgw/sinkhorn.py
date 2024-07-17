import warnings
import torch
import numpy as np


def sinkhorn(
    a,
    b,
    M,
    reg,
    method="sinkhorn_log",
    numItermax=100,
    stopThr=1e-5,
    verbose=False,
    log=False,
    warn=True,
    warmstart=None,
    **kwargs,
):

    if method.lower() == "sinkhorn":
        return sinkhorn_knopp(
            a,
            b,
            M,
            reg,
            numItermax=numItermax,
            stopThr=stopThr,
            verbose=verbose,
            log=log,
            warn=warn,
            warmstart=warmstart,
            **kwargs,
        )
    elif method.lower() == "sinkhorn_log":
        return sinkhorn_log(
            a,
            b,
            M,
            reg,
            numItermax=numItermax,
            stopThr=stopThr,
            verbose=verbose,
            log=log,
            warn=warn,
            warmstart=warmstart,
            **kwargs,
        )
    elif method.lower() == "greenkhorn":
        return greenkhorn(
            a,
            b,
            M,
            reg,
            numItermax=numItermax,
            stopThr=stopThr,
            verbose=verbose,
            log=log,
            warn=warn,
            warmstart=warmstart,
        )
    elif method.lower() == "sinkhorn_stabilized":
        return sinkhorn_stabilized(
            a,
            b,
            M,
            reg,
            numItermax=numItermax,
            stopThr=stopThr,
            warmstart=warmstart,
            verbose=verbose,
            log=log,
            warn=warn,
            **kwargs,
        )
    elif method.lower() == "sinkhorn_epsilon_scaling":
        return sinkhorn_epsilon_scaling(
            a,
            b,
            M,
            reg,
            numItermax=numItermax,
            stopThr=stopThr,
            warmstart=warmstart,
            verbose=verbose,
            log=log,
            warn=warn,
            **kwargs,
        )
    else:
        raise ValueError("Unknown method '%s'." % method)


def sinkhorn2(
    a,
    b,
    M,
    reg,
    method="sinkhorn",
    numItermax=1000,
    stopThr=1e-9,
    verbose=False,
    log=False,
    warn=False,
    warmstart=None,
    **kwargs,
):

    if len(b.shape) < 2:
        if method.lower() == "sinkhorn":
            res = sinkhorn_knopp(
                a,
                b,
                M,
                reg,
                numItermax=numItermax,
                stopThr=stopThr,
                verbose=verbose,
                log=log,
                warn=warn,
                warmstart=warmstart,
                **kwargs,
            )
        elif method.lower() == "sinkhorn_log":
            res = sinkhorn_log(
                a,
                b,
                M,
                reg,
                numItermax=numItermax,
                stopThr=stopThr,
                verbose=verbose,
                log=log,
                warn=warn,
                warmstart=warmstart,
                **kwargs,
            )
        elif method.lower() == "sinkhorn_stabilized":
            res = sinkhorn_stabilized(
                a,
                b,
                M,
                reg,
                numItermax=numItermax,
                stopThr=stopThr,
                warmstart=warmstart,
                verbose=verbose,
                log=log,
                warn=warn,
                **kwargs,
            )
        else:
            raise ValueError("Unknown method '%s'." % method)
        if log:
            return torch.sum(M * res[0]), res[1]
        else:
            return torch.sum(M * res)

    else:

        if method.lower() == "sinkhorn":
            return sinkhorn_knopp(
                a,
                b,
                M,
                reg,
                numItermax=numItermax,
                stopThr=stopThr,
                verbose=verbose,
                log=log,
                warn=warn,
                warmstart=warmstart,
                **kwargs,
            )
        elif method.lower() == "sinkhorn_log":
            return sinkhorn_log(
                a,
                b,
                M,
                reg,
                numItermax=numItermax,
                stopThr=stopThr,
                verbose=verbose,
                log=log,
                warn=warn,
                warmstart=warmstart,
                **kwargs,
            )
        elif method.lower() == "sinkhorn_stabilized":
            return sinkhorn_stabilized(
                a,
                b,
                M,
                reg,
                numItermax=numItermax,
                stopThr=stopThr,
                warmstart=warmstart,
                verbose=verbose,
                log=log,
                warn=warn,
                **kwargs,
            )
        else:
            raise ValueError("Unknown method '%s'." % method)


def sinkhorn_knopp(
    a,
    b,
    M,
    reg,
    numItermax=1000,
    stopThr=1e-9,
    verbose=False,
    log=False,
    warn=True,
    warmstart=None,
    **kwargs,
):

    if len(a) == 0:
        a = torch.full((M.shape[0],), 1.0 / M.shape[0]).to(M)
    if len(b) == 0:
        b = torch.full((M.shape[1],), 1.0 / M.shape[1]).to(M)

    # init data
    dim_a = len(a)
    dim_b = b.shape[0]

    if len(b.shape) > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    if log:
        log = {"err": []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        if n_hists:
            u = torch.ones((dim_a, n_hists)).to(M) / dim_a
            v = torch.ones((dim_b, n_hists)).to(M) / dim_b
        else:
            u = torch.ones(dim_a).to(M) / dim_a
            v = torch.ones(dim_b).to(M) / dim_b
    else:
        u, v = torch.exp(warmstart[0]), torch.exp(warmstart[1])

    K = torch.exp(M / (-reg))

    Kp = (1 / a).reshape(-1, 1) * K

    err = 1
    for ii in range(numItermax):
        uprev = u
        vprev = v
        KtransposeU = K.T @ u
        v = b / KtransposeU
        u = 1.0 / (Kp @ v)

        if (
            torch.any(KtransposeU == 0)
            or torch.any(torch.isnan(u))
            or torch.any(torch.isnan(v))
            or torch.any(torch.isinf(u))
            or torch.any(torch.isinf(v))
        ):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn("Warning: numerical errors at iteration %d" % ii)
            u = uprev
            v = vprev
            break
        if ii % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            if n_hists:
                tmp2 = torch.einsum("ik,ij,jk->jk", u, K, v)
            else:
                # compute right marginal tmp2= (diag(u)Kdiag(v))^T1
                tmp2 = torch.einsum("i,ij,j->j", u, K, v)
            err = torch.norm(tmp2 - b)  # violation of marginal
            if log:
                log["err"].append(err)

            if err < stopThr:
                break
            if verbose:
                if ii % 200 == 0:
                    print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                print("{:5d}|{:8e}|".format(ii, err))
    else:
        if warn:
            warnings.warn(
                "Sinkhorn did not converge. You might want to increase the number of iterations `numItermax` or the regularization parameter `reg`."
            )
    if log:
        log["niter"] = ii
        log["u"] = u
        log["v"] = v

    if n_hists:  # return only loss
        res = torch.einsum("ik,ij,jk,ij->k", u, K, v, M)
        if log:
            return res, log
        else:
            return res

    else:  # return OT matrix

        if log:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1)), log
        else:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1))


def sinkhorn_log(
    a,
    b,
    M,
    reg,
    numItermax=1000,
    stopThr=1e-9,
    verbose=False,
    log=False,
    warn=True,
    warmstart=None,
    **kwargs,
):

    if len(a) == 0:
        a = torch.full((M.shape[0],), 1.0 / M.shape[0]).to(M)
    if len(b) == 0:
        b = torch.full((M.shape[1],), 1.0 / M.shape[1]).to(M)

    # init data
    dim_a = len(a)
    dim_b = b.shape[0]

    if len(b.shape) > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    # in case of multiple historgrams
    if n_hists > 1 and warmstart is None:
        warmstart = [None] * n_hists

    if n_hists:  # we do not want to use tensors sor we do a loop

        lst_loss = []
        lst_u = []
        lst_v = []

        for k in range(n_hists):
            res = sinkhorn_log(
                a,
                b[:, k],
                M,
                reg,
                numItermax=numItermax,
                stopThr=stopThr,
                verbose=verbose,
                log=log,
                warmstart=warmstart[k],
                **kwargs,
            )

            if log:
                lst_loss.append(torch.sum(M * res[0]))
                lst_u.append(res[1]["log_u"])
                lst_v.append(res[1]["log_v"])
            else:
                lst_loss.append(torch.sum(M * res))
        res = torch.stack(lst_loss)
        if log:
            log = {
                "log_u": torch.stack(lst_u, 1),
                "log_v": torch.stack(lst_v, 1),
            }
            log["u"] = torch.exp(log["log_u"])
            log["v"] = torch.exp(log["log_v"])
            return res, log
        else:
            return res

    else:

        if log:
            log = {"err": []}

        Mr = -M / reg

        # we assume that no distances are null except those of the diagonal of
        # distances
        if warmstart is None:
            u = torch.zeros(dim_a).to(M)
            v = torch.zeros(dim_b).to(M)
        else:
            u, v = warmstart

        def get_logT(u, v):
            if n_hists:
                return Mr[:, :, None] + u + v
            else:
                return Mr + u[:, None] + v[None, :]

        loga = torch.log(a)
        logb = torch.log(b)

        err = 1
        for ii in range(numItermax):

            v = logb - torch.logsumexp(Mr + u[:, None], 0)
            u = loga - torch.logsumexp(Mr + v[None, :], 1)

            if ii % 10 == 0:
                # we can speed up the process by checking for the error only all
                # the 10th iterations

                # compute right marginal tmp2= (diag(u)Kdiag(v))^T1
                tmp2 = torch.sum(torch.exp(get_logT(u, v)), 0)
                err = torch.norm(tmp2 - b)  # violation of marginal
                if log:
                    log["err"].append(err)

                if verbose:
                    if ii % 200 == 0:
                        print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                    print("{:5d}|{:8e}|".format(ii, err))
                if err < stopThr:
                    break
        else:
            if warn:
                warnings.warn(
                    "Sinkhorn did not converge. You might want to increase the number of iterations `numItermax` or the regularization parameter `reg`."
                )

        if log:
            log["niter"] = ii
            log["log_u"] = u
            log["log_v"] = v
            log["u"] = torch.exp(u)
            log["v"] = torch.exp(v)

            return torch.exp(get_logT(u, v)), log

        else:
            return torch.exp(get_logT(u, v))


def greenkhorn(
    a,
    b,
    M,
    reg,
    numItermax=10000,
    stopThr=1e-9,
    verbose=False,
    log=False,
    warn=True,
    warmstart=None,
):

    if len(a) == 0:
        a = torch.full((M.shape[0],), 1.0 / M.shape[0]).to(M)
    if len(b) == 0:
        b = torch.full((M.shape[1],), 1.0 / M.shape[1]).to(M)

    dim_a = a.shape[0]
    dim_b = b.shape[0]

    K = torch.exp(-M / reg)

    if warmstart is None:
        u = torch.full((dim_a,), 1.0 / dim_a).to(M)
        v = torch.full((dim_b,), 1.0 / dim_b).to(M)
    else:
        u, v = torch.exp(warmstart[0]), torch.exp(warmstart[1])
    G = u[:, None] * K * v[None, :]

    viol = torch.sum(G, dim=1) - a
    viol_2 = torch.sum(G, dim=0) - b
    stopThr_val = 1
    if log:
        log = dict()
        log["u"] = u
        log["v"] = v

    for ii in range(numItermax):
        i_1 = torch.argmax(torch.abs(viol))
        i_2 = torch.argmax(torch.abs(viol_2))
        m_viol_1 = torch.abs(viol[i_1])
        m_viol_2 = torch.abs(viol_2[i_2])
        stopThr_val = torch.maximum(m_viol_1, m_viol_2)

        if m_viol_1 > m_viol_2:
            old_u = u[i_1]
            new_u = a[i_1] / torch.dot(K[i_1, :], v)
            G[i_1, :] = new_u * K[i_1, :] * v

            viol[i_1] = torch.dot(new_u * K[i_1, :], v) - a[i_1]
            viol_2 += K[i_1, :].T * (new_u - old_u) * v
            u[i_1] = new_u
        else:
            old_v = v[i_2]
            new_v = b[i_2] / torch.dot(K[:, i_2].T, u)
            G[:, i_2] = u * K[:, i_2] * new_v
            # aviol = (G@one_m - a)
            # aviol_2 = (G.T@one_n - b)
            viol += (-old_v + new_v) * K[:, i_2] * u
            viol_2[i_2] = new_v * torch.dot(K[:, i_2], u) - b[i_2]
            v[i_2] = new_v

        if stopThr_val <= stopThr:
            break
    else:
        if warn:
            warnings.warn(
                "Sinkhorn did not converge. You might want to increase the number of iterations `numItermax` or the regularization parameter `reg`."
            )

    if log:
        log["n_iter"] = ii
        log["u"] = u
        log["v"] = v

    if log:
        return G, log
    else:
        return G


def sinkhorn_stabilized(
    a,
    b,
    M,
    reg,
    numItermax=1000,
    tau=1e3,
    stopThr=1e-9,
    warmstart=None,
    verbose=False,
    print_period=20,
    log=False,
    warn=True,
    **kwargs,
):

    if len(a) == 0:
        a = torch.full((M.shape[0],), 1.0 / M.shape[0]).to(M)
    if len(b) == 0:
        b = torch.full((M.shape[1],), 1.0 / M.shape[1]).to(M)

    # test if multiple target
    if len(b.shape) > 1:
        n_hists = b.shape[1]
        a = a[:, None]
    else:
        n_hists = 0

    # init data
    dim_a = len(a)
    dim_b = len(b)

    if log:
        log = {"err": []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        alpha, beta = torch.zeros(dim_a).to(M), torch.zeros(dim_b).to(M)
    else:
        alpha, beta = warmstart

    if n_hists:
        u = torch.ones((dim_a, n_hists)).to(M) / dim_a
        v = torch.ones((dim_b, n_hists)).to(M) / dim_b
    else:
        u, v = torch.full((dim_a,), 1.0 / dim_a).to(M), torch.full((dim_b,), 1.0 / dim_b).to(M)

    def get_K(alpha, beta):
        """log space computation"""
        return torch.exp(-(M - alpha.reshape((dim_a, 1)) - beta.reshape((1, dim_b))) / reg)

    def get_Gamma(alpha, beta, u, v):
        """log space gamma computation"""
        return torch.exp(
            -(M - alpha.reshape((dim_a, 1)) - beta.reshape((1, dim_b))) / reg
            + torch.log(u.reshape((dim_a, 1)))
            + torch.log(v.reshape((1, dim_b)))
        )

    K = get_K(alpha, beta)
    transp = K
    err = 1
    for ii in range(numItermax):

        uprev = u
        vprev = v

        # sinkhorn update
        v = b / (K.T @ u)
        u = a / (K @ v)

        # remove numerical problems and store them in K
        if torch.max(torch.abs(u)) > tau or torch.max(torch.abs(v)) > tau:
            if n_hists:
                alpha, beta = alpha + reg * torch.max(torch.log(u), 1), beta + reg * torch.max(
                    torch.log(v)
                )
            else:
                alpha, beta = alpha + reg * torch.log(u), beta + reg * torch.log(v)
                if n_hists:
                    u = torch.ones((dim_a, n_hists)).to(M) / dim_a
                    v = torch.ones((dim_b, n_hists)).to(M) / dim_b
                else:
                    u = torch.full((dim_a,), 1.0 / dim_a).to(M)
                    v = torch.full((dim_b,), 1.0 / dim_b).to(M)
            K = get_K(alpha, beta)

        if ii % print_period == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            if n_hists:
                err_u = torch.max(torch.abs(u - uprev))
                err_u /= max(torch.max(torch.abs(u)), torch.max(torch.abs(uprev)), 1.0)
                err_v = torch.max(torch.abs(v - vprev))
                err_v /= max(torch.max(torch.abs(v)), torch.max(torch.abs(vprev)), 1.0)
                err = 0.5 * (err_u + err_v)
            else:
                transp = get_Gamma(alpha, beta, u, v)
                err = torch.norm(torch.sum(transp, dim=0) - b)
            if log:
                log["err"].append(err)

            if verbose:
                if ii % (print_period * 20) == 0:
                    print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                print("{:5d}|{:8e}|".format(ii, err))

        if err <= stopThr:
            break

        if torch.any(torch.isnan(u)) or torch.any(torch.isnan(v)):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn("Numerical errors at iteration %d" % ii)
            u = uprev
            v = vprev
            break
    else:
        if warn:
            warnings.warn(
                "Sinkhorn did not converge. You might want to increase the number of iterations `numItermax` or the regularization parameter `reg`."
            )
    if log:
        if n_hists:
            alpha = alpha[:, None]
            beta = beta[:, None]
        logu = alpha / reg + torch.log(u)
        logv = beta / reg + torch.log(v)
        log["n_iter"] = ii
        log["logu"] = logu
        log["logv"] = logv
        log["alpha"] = alpha + reg * torch.log(u)
        log["beta"] = beta + reg * torch.log(v)
        log["warmstart"] = (log["alpha"], log["beta"])
        if n_hists:
            res = torch.stack(
                [torch.sum(get_Gamma(alpha, beta, u[:, i], v[:, i]) * M) for i in range(n_hists)]
            )
            return res, log

        else:
            return get_Gamma(alpha, beta, u, v), log
    else:
        if n_hists:
            res = torch.stack(
                [torch.sum(get_Gamma(alpha, beta, u[:, i], v[:, i]) * M) for i in range(n_hists)]
            )
            return res
        else:
            return get_Gamma(alpha, beta, u, v)


def sinkhorn_epsilon_scaling(
    a,
    b,
    M,
    reg,
    numItermax=100,
    epsilon0=1e4,
    numInnerItermax=100,
    tau=1e3,
    stopThr=1e-9,
    warmstart=None,
    verbose=False,
    print_period=10,
    log=False,
    warn=True,
    **kwargs,
):

    if len(a) == 0:
        a = torch.full((M.shape[0],), 1.0 / M.shape[0]).to(M)
    if len(b) == 0:
        b = torch.full((M.shape[1],), 1.0 / M.shape[1]).to(M)

    # init data
    dim_a = len(a)
    dim_b = len(b)

    # nrelative umerical precision with 64 bits
    numItermin = 35
    numItermax = max(numItermin, numItermax)  # ensure that last velue is exact

    ii = 0
    if log:
        log = {"err": []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        alpha, beta = torch.zeros(dim_a).to(M), torch.zeros(dim_b).to(M)
    else:
        alpha, beta = warmstart

    # print(np.min(K))
    def get_reg(n):  # exponential decreasing
        return (epsilon0 - reg) * np.exp(-n) + reg

    err = 1
    for ii in range(numItermax):

        regi = get_reg(ii)

        G, logi = sinkhorn_stabilized(
            a,
            b,
            M,
            regi,
            numItermax=numInnerItermax,
            stopThr=stopThr,
            warmstart=(alpha, beta),
            verbose=False,
            print_period=20,
            tau=tau,
            log=True,
        )

        alpha = logi["alpha"]
        beta = logi["beta"]

        if ii % (print_period) == 0:  # spsion nearly converged
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            transp = G
            err = (
                torch.norm(torch.sum(transp, dim=0) - b) ** 2
                + torch.norm(torch.sum(transp, dim=1) - a) ** 2
            )
            if log:
                log["err"].append(err)

            if verbose:
                if ii % (print_period * 10) == 0:
                    print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                print("{:5d}|{:8e}|".format(ii, err))

        if err <= stopThr and ii > numItermin:
            break
    else:
        if warn:
            warnings.warn(
                "Sinkhorn did not converge. You might want to increase the number of iterations `numItermax` or the regularization parameter `reg`."
            )
    if log:
        log["alpha"] = alpha
        log["beta"] = beta
        log["warmstart"] = (log["alpha"], log["beta"])
        log["niter"] = ii
        return G, log
    else:
        return G
