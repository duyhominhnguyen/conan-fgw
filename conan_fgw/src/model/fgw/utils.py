import torch


def init_matrix(C1, C2, p, q, loss_fun="square_loss"):

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
            return a * torch.log(a + 1e-15) - a

        def f2(b):
            return b

        def h1(a):
            return a

        def h2(b):
            return torch.log(b + 1e-15)

    else:
        raise ValueError(
            f"Unknown `loss_fun='{loss_fun}'`. Use one of: {'square_loss', 'kl_loss'}."
        )

    constC1 = f1(C1) @ p.reshape(-1, 1) @ torch.ones((1, q.shape[-1])).to(q)
    constC2 = torch.ones((p.shape[-1], 1)).to(p) @ q.reshape(1, -1) @ f2(C2).T
    constC = constC1 + constC2
    hC1 = h1(C1)
    hC2 = h2(C2)

    return constC, hC1, hC2


def tensor_product(constC, hC1, hC2, T):

    A = -hC1 @ T @ hC2.T
    tens = constC + A
    # tens -= tens.min()
    return tens


def gwloss(constC, hC1, hC2, T):

    tens = tensor_product(constC, hC1, hC2, T)
    return torch.sum(tens * T)


def gwggrad(constC, hC1, hC2, T):

    return 2 * tensor_product(constC, hC1, hC2, T)  # [12] Prop. 2 misses a 2 factor


def update_square_loss(p, lambdas, T, Cs):

    # Correct order mistake in Equation 14 in [12]
    tmpsum = sum([lambdas[s] * T[s] @ Cs[s] @ T[s].T for s in range(len(T))])

    ppt = torch.outer(p, p)
    return tmpsum / ppt


def update_kl_loss(p, lambdas, T, Cs):

    # Correct order mistake in Equation 15 in [12]
    tmpsum = sum(
        [
            lambdas[s] * (T[s] @ torch.log(torch.clamp(Cs[s], min=1e-15)) @ T[s].T)
            for s in range(len(T))
        ]
    )

    ppt = torch.outer(p, p)
    return torch.exp(tmpsum / ppt)


def update_feature_matrix(lambdas, Ys, Ts, p):

    p = 1.0 / p

    tmpsum = sum([lambdas[s] * (Ys[s] @ Ts[s].T) * p[None, :] for s in range(len(Ts))])
    return tmpsum


def init_matrix_semirelaxed(C1, C2, p, loss_fun="square_loss"):

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
            return a * torch.log(a + 1e-15) - a

        def f2(b):
            return b

        def h1(a):
            return a

        def h2(b):
            return torch.log(b + 1e-15)

    else:
        raise ValueError(
            f"Unknown `loss_fun='{loss_fun}'`. Use one of: {'square_loss', 'kl_loss'}."
        )

    constC = f1(C1) @ p.reshape(-1, 1) @ torch.ones((1, C2.shape[-1])).to(C2)

    hC1 = h1(C1)
    hC2 = h2(C2)
    fC2t = f2(C2).T
    return constC, hC1, hC2, fC2t


def dist(x1, x2=None, metric="sqeuclidean", p=2, w=None):

    if x2 is None:
        x2 = x1
    if metric == "sqeuclidean":
        return euclidean_distances(x1, x2, squared=True)
    elif metric == "euclidean":
        return euclidean_distances(x1, x2, squared=False)
    else:
        # TODO: implement other metric with torch
        raise NotImplementedError()


def euclidean_distances(X, Y, squared=False):

    a2 = torch.einsum("ij,ij->i", X, X)
    b2 = torch.einsum("ij,ij->i", Y, Y)

    c = -2 * (X @ Y.T)
    c += a2[:, None]
    c += b2[None, :]

    c = torch.clamp(c, min=0)

    if not squared:
        c = torch.sqrt(c)

    if X is Y:
        c = c * (1 - torch.eye(X.shape[0]).to(c))

    return c
