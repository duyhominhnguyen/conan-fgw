import torch
import numpy as np
from .bregman import fgw
from .utils import dist, update_feature_matrix, update_square_loss, update_kl_loss
from lpmp_gm import gm_solver


def fgw_barycenters(
    N,
    Ys,
    Cs,
    ps=None,
    p=None,
    lambdas=None,
    loss_fun="square_loss",
    epsilon=0.1,
    symmetric=True,
    alpha=0.5,
    max_iter=100,
    tol=1e-9,
    solver="PGD",
    stop_criterion="barycenter",
    warmstartT=False,
    verbose=False,
    log=False,
    init_C=None,
    init_Y=None,
    fixed_structure=False,
    fixed_features=False,
    seed=0,
    **kwargs,
):

    if loss_fun not in ("square_loss", "kl_loss"):
        raise ValueError(
            f"Unknown `loss_fun='{loss_fun}'`. Use one of: {'square_loss', 'kl_loss'}."
        )

    if stop_criterion not in ["barycenter", "loss"]:
        raise ValueError(
            f"Unknown `stop_criterion='{stop_criterion}'`. Use one of: {'barycenter', 'loss'}."
        )

    if solver not in ["PGD", "PPA", "BAPG"]:
        raise ValueError("Unknown solver '%s'. Pick one in ['PGD', 'PPA', 'BAPG']." % solver)

    S = len(Cs)
    if lambdas is None:
        lambdas = [1.0 / S] * S

    if p is None:
        p = torch.ones(N).to(Cs[0]) / N

    d = Ys[0].shape[1]  # dimension on the node features

    # Initialization of C : random euclidean distance matrix (if not provided by user)
    if fixed_structure:
        if init_C is None:
            raise ValueError("If C is fixed it must be initialized")
        else:
            C = init_C
    else:
        if init_C is None:
            torch.manual_seed(seed)
            xalea = torch.randn(N, 2).to(Cs[0])
            C = dist(xalea, xalea)
        else:
            C = init_C

    # Initialization of Y
    if fixed_features:
        if init_Y is None:
            raise ValueError("If Y is fixed it must be initialized")
        else:
            Y = init_Y
    else:
        if init_Y is None:
            Y = torch.zeros((N, d)).to(ps[0])

        else:
            Y = init_Y

    Ms = [dist(Y, Ys[s]) for s in range(len(Ys))]

    if warmstartT:
        T = [None] * S

    cpt = 0

    if stop_criterion == "barycenter":
        inner_log = False
        err_feature = 1e15
        err_structure = 1e15
        err_rel_loss = 0.0

    else:
        inner_log = True
        err_feature = 0.0
        err_structure = 0.0
        curr_loss = 1e15
        err_rel_loss = 1e15

    if log:
        log_ = {}
        if stop_criterion == "barycenter":
            log_["err_feature"] = []
            log_["err_structure"] = []
            log_["Ts_iter"] = []
        else:
            log_["loss"] = []
            log_["err_rel_loss"] = []

    while (err_feature > tol or err_structure > tol or err_rel_loss > tol) and cpt < max_iter:
        if stop_criterion == "barycenter":
            Cprev = C
            Yprev = Y
        else:
            prev_loss = curr_loss

        # get transport plans
        with torch.no_grad():  # comment this line for no gradient through mapping T
            if warmstartT:
                res = [
                    fgw(
                        Ms[s],
                        C,
                        Cs[s],
                        p,
                        ps[s],
                        loss_fun,
                        epsilon,
                        symmetric,
                        alpha,
                        T[s],
                        max_iter,
                        1e-4,
                        solver=solver,
                        verbose=False,
                        log=inner_log,
                        **kwargs,
                    )
                    for s in range(S)
                ]

            else:
                res = [
                    fgw(
                        Ms[s],
                        C,
                        Cs[s],
                        p,
                        ps[s],
                        loss_fun,
                        epsilon,
                        symmetric,
                        alpha,
                        None,
                        max_iter,
                        1e-4,
                        solver=solver,
                        verbose=False,
                        log=inner_log,
                        **kwargs,
                    )
                    for s in range(S)
                ]

        if stop_criterion == "barycenter":
            T = res
        else:
            T = [output[0] for output in res]
            curr_loss = torch.sum([output[1]["fgw_dist"] for output in res])

        # update barycenters
        if not fixed_features:
            Ys_temp = [y.T for y in Ys]
            Y = update_feature_matrix(lambdas, Ys_temp, T, p).T
            Ms = [dist(Y, Ys[s]) for s in range(len(Ys))]

        if not fixed_structure:
            if loss_fun == "square_loss":
                C = update_square_loss(p, lambdas, T, Cs)

            elif loss_fun == "kl_loss":
                C = update_kl_loss(p, lambdas, T, Cs)

        # update convergence criterion
        if stop_criterion == "barycenter":
            err_feature, err_structure = 0.0, 0.0
            if not fixed_features:
                err_feature = torch.norm(Y - Yprev)
            if not fixed_structure:
                err_structure = torch.norm(C - Cprev)
            if log:
                log_["err_feature"].append(err_feature)
                log_["err_structure"].append(err_structure)
                log_["Ts_iter"].append(T)

            if verbose:
                if cpt % 200 == 0:
                    print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                print("{:5d}|{:8e}|".format(cpt, err_structure))
                print("{:5d}|{:8e}|".format(cpt, err_feature))
        else:
            err_rel_loss = (
                abs(curr_loss - prev_loss) / prev_loss if prev_loss != 0.0 else torch.nan
            )
            if log:
                log_["loss"].append(curr_loss)
                log_["err_rel_loss"].append(err_rel_loss)

            if verbose:
                if cpt % 200 == 0:
                    print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                print("{:5d}|{:8e}|".format(cpt, err_rel_loss))

        cpt += 1

    if log:
        log_["T"] = T
        log_["p"] = p
        log_["Ms"] = Ms

        return Y, C, log_
    else:
        return Y, C


def fused_ACC_torch(M, A, B, a=None, b=None, X=None, alpha=0, epoch=200, eps=1e-5, rho=1e-1):
    if a is None:
        a = torch.full((A.shape[0],), 1.0 / A.shape[0]).to(A)
    else:
        a = a[:, None].to(A)

    if b is None:
        b = torch.full((B.shape[0],), 1.0 / B.shape[0]).to(B)
    else:
        b = b[:, None].to(B)

    if X is None:
        X = a @ b.T
    obj_list = []
    for ii in range(epoch):
        X = X + 1e-10
        grad = 4 * alpha * A @ X @ B - (1 - alpha) * M
        X = torch.exp(grad / rho) * X
        X = X * (a / (X @ torch.ones_like(b)))
        grad = 4 * alpha * A @ X @ B - (1 - alpha) * M
        X = torch.exp(grad / rho) * X
        X = X * (b.T / (X.T @ torch.ones_like(a)).T)
        if ii > 0 and ii % 10 == 0:
            objective = torch.trace(((1 - alpha) * M - 2 * alpha * A @ X @ B) @ X.T)
            if len(obj_list) > 0 and torch.abs((objective - obj_list[-1]) / obj_list[-1]) < eps:
                # print('iter:{}, smaller than eps'.format(ii))
                break
            obj_list.append(objective)
    return X, obj_list


def fgw_barycenters_BAPG(
    N,
    Ys,
    Cs,
    ps=None,
    p=None,
    lambdas=None,
    loss_fun="square_loss",
    alpha=0.5,
    max_iter=100,
    tol=1e-9,
    rho=1.0,
    verbose=False,
    log=False,
    init_C=None,
    init_Y=None,
    fixed_structure=False,
    fixed_features=False,
    seed=0,
    **kwargs,
):
    """https://github.com/ArthurLeoM/FGWMixup/blob/main/src/FGW_barycenter.py"""

    if loss_fun not in ("square_loss", "kl_loss"):
        raise ValueError(
            f"Unknown `loss_fun='{loss_fun}'`. Use one of: {'square_loss', 'kl_loss'}."
        )

    S = len(Cs)
    if lambdas is None:
        lambdas = [1.0 / S] * S

    if p is None:
        p = torch.ones(N).to(Cs[0]) / N

    d = Ys[0].shape[1]  # dimension on the node features

    # Initialization of C : random euclidean distance matrix (if not provided by user)
    if fixed_structure:
        if init_C is None:
            raise ValueError("If C is fixed it must be initialized")
        else:
            C = init_C
    else:
        if init_C is None:
            torch.manual_seed(seed)
            xalea = torch.randn(N, 2).to(Cs[0])
            C = dist(xalea, xalea)
        else:
            C = init_C

    # Initialization of Y
    if fixed_features:
        if init_Y is None:
            raise ValueError("If Y is fixed it must be initialized")
        else:
            Y = init_Y
    else:
        if init_Y is None:
            Y = torch.zeros((N, d)).to(ps[0])

        else:
            Y = init_Y

    Ms = [dist(Y, Ys[s]) for s in range(len(Ys))]

    cpt = 0
    inner_log = False
    err_feature = 1e15
    err_structure = 1e15
    err_rel_loss = 0.0

    if log:
        log_ = {}
        log_["err_feature"] = []
        log_["err_structure"] = []
        log_["Ts_iter"] = []

    while (err_feature > tol or err_structure > tol) and cpt < max_iter:
        Cprev = C
        Yprev = Y

        T = []
        # dists = []
        for s in range(S):
            cur_T, _ = fused_ACC_torch(
                Ms[s], C, Cs[s], p, ps[s], alpha=alpha, epoch=100, eps=1e-5, rho=rho
            )
            T.append(cur_T)
            # c1 = np.dot(C*C, np.outer(p, np.ones_like(ps[s]))) + np.dot(np.outer(np.ones_like(p), ps[s]), Cs[s]*Cs[s])
            # res = np.trace(np.dot(c1.T, cur_T))
            # dists.append(cur_dist[-1] + alpha * res)

        if not fixed_features:
            Ys_temp = [y.T for y in Ys]
            Y = update_feature_matrix(lambdas, Ys_temp, T, p).T
            Ms = [dist(Y, Ys[s]) for s in range(len(Ys))]

        if not fixed_structure:
            if loss_fun == "square_loss":
                C = update_square_loss(p, lambdas, T, Cs)

            elif loss_fun == "kl_loss":
                C = update_kl_loss(p, lambdas, T, Cs)

        # update convergence criterion
        err_feature, err_structure = 0.0, 0.0
        if not fixed_features:
            err_feature = torch.norm(Y - Yprev)
        if not fixed_structure:
            err_structure = torch.norm(C - Cprev)
        if log:
            log_["err_feature"].append(err_feature)
            log_["err_structure"].append(err_structure)
            log_["Ts_iter"].append(T)

        if verbose:
            if cpt % 200 == 0:
                print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
            print("{:5d}|{:8e}|".format(cpt, err_structure))
            print("{:5d}|{:8e}|".format(cpt, err_feature))

        cpt += 1

    if log:
        log_["T"] = T
        log_["p"] = p
        log_["Ms"] = Ms

        return Y, C, log_
    else:
        return Y, C


def normalize_tensor(tensor, a, b):
    min_value = tensor.min()
    max_value = tensor.max()

    normalized_tensor = a + (tensor - min_value) * (b - a) / (max_value - min_value)

    return normalized_tensor


def batch_fused_ACC_torch(M, A, B, a=None, b=None, X=None, alpha=0, epoch=200, eps=1e-5, rho=1e-1):
    assert B.shape[0] == M.shape[0]
    batch_size = B.shape[0]
    a_size, b_size = A.shape[0], B.shape[1]
    if a is None:
        a = torch.full((batch_size, a_size), 1.0 / a_size).to(A)
    else:
        a = a.to(A)
        if a.ndim == 1:
            a = a[None, :].repeat(batch_size, 1)

    if b is None:
        b = torch.full((batch_size, b_size), 1.0 / b_size).to(B)
    else:
        b = b.to(B)

    if X is None:
        X = torch.einsum('bi,bj->bij', a, b)

    prev_obj = 0
    for ii in range(epoch):
        X = X + 1e-10
        # grad = 4 * alpha * A @ X @ B - (1 - alpha) * M
        grad = 4 * alpha * torch.einsum('ij,bjk->bik', A, torch.einsum('bij,bjk->bik', X, B)) - (1 - alpha) * M
        X = torch.exp(grad / rho) * X
        # X = X * (a / (X @  torch.ones_like(b)))
        X = X * (a / X.sum(dim=2))[:, :, None]
        # grad = 4 * alpha * A @ X @ B - (1 - alpha) * M
        grad = 4 * alpha * torch.einsum('ij,bjk->bik', A, torch.einsum('bij,bjk->bik', X, B)) - (1 - alpha) * M
        X = torch.exp(grad / rho) * X
        # X = X * (b.T / (X.T @ torch.ones_like(a)).T)
        X = X * (b / X.sum(dim=1))[:, None, :]
        if ii > 0 and ii % 10 == 0:
            # objective = torch.trace(((1 - alpha) * M - 2 * alpha * A @ X @ B) @ X.T)
            objective = torch.einsum('bij,bjk->bik', (1 - alpha) * M - 2 * alpha * torch.einsum('ij,bjk->bik', A, torch.einsum('bij,bjk->bik', X, B)), X)
            objective = objective.sum()
            if torch.abs((objective - prev_obj) / prev_obj) < eps:
                # print('iter:{}, smaller than eps'.format(ii))
                break
            prev_obj = objective
    return X


def batch_update_kl_loss(p, lambdas, T, Cs):

    # Correct order mistake in Equation 15 in [12]
    tmpsum = torch.einsum('sij,sjk->sik', T, torch.log(torch.clamp(Cs, min=1e-15)))
    tmpsum = torch.einsum('sij,sjk->sik', tmpsum, T.transpose(1, 2))
    tmpsum = (lambdas[:, None, None] * tmpsum).sum(dim=0)

    ppt = torch.outer(p, p)
    return torch.exp(tmpsum / ppt)


def batch_update_kl_loss(p, lambdas, T, Cs):

    # Correct order mistake in Equation 15 in [12]
    tmpsum = torch.einsum('sij,sjk->sik', T, torch.log(torch.clamp(Cs, min=1e-15)))
    tmpsum = torch.einsum('sij,sjk->sik', tmpsum, T.transpose(1, 2))
    tmpsum = (lambdas[:, None, None] * tmpsum).sum(dim=0)

    ppt = torch.outer(p, p)
    return torch.exp(tmpsum / ppt)


def batch_update_square_loss(p, lambdas, T, Cs):

    # Correct order mistake in Equation 14 in [12]
    tmpsum = torch.einsum('sij,sjk->sik', T, Cs)
    tmpsum = torch.einsum('sij,sjk->sik', tmpsum, T.transpose(1, 2))
    tmpsum = (lambdas[:, None, None] * tmpsum).sum(dim=0)

    ppt = torch.outer(p, p)
    return tmpsum / ppt


def batch_update_feature_matrix(lambdas, Ys, Ts, p):
    tmpsum = lambdas[:, None, None] * torch.einsum('sij,sjk->sik', Ts, Ys)
    tmpsum = tmpsum.sum(dim=0) / p[:, None]

    return tmpsum


def batch_fgw_barycenters_BAPG(
    N, Ys, Cs, ps=None, p=None, lambdas=None, loss_fun='square_loss',
    alpha=0.5, max_iter=100, toly=1e-9, tolc=1e-9, rho=1., verbose=False,
    log=False, init_C=None, init_Y=None, fixed_structure=False,
    fixed_features=False, seed=0, **kwargs
):

    if loss_fun not in ('square_loss', 'kl_loss'):
        raise ValueError(f"Unknown `loss_fun='{loss_fun}'`. Use one of: {'square_loss', 'kl_loss'}.")

    S = Ys.shape[0]
    if lambdas is None:
        lambdas = torch.full((S,), 1.0 / S).to(Ys)

    if p is None:
        p = torch.full((N,), 1.0 / N).to(Ys)
    
    if ps is None:
        ps = torch.full((S, N), 1.0 / N).to(Ys)

    d = Ys.shape[-1]  # dimension on the node features

    # Initialization of C : random euclidean distance matrix (if not provided by user)
    if fixed_structure:
        if init_C is None:
            raise ValueError('If C is fixed it must be initialized')
        else:
            C = init_C
    else:
        if init_C is None:
            torch.manual_seed(seed)
            xalea = torch.randn(N, 2).to(Cs)
            C = dist(xalea, xalea)
        else:
            C = init_C

    # Initialization of Y
    if fixed_features:
        if init_Y is None:
            raise ValueError('If Y is fixed it must be initialized')
        else:
            Y = init_Y
    else:
        if init_Y is None:
            Y = torch.zeros((N, d)).to(ps[0])

        else:
            Y = init_Y

    # Ms = [dist(Y, Ys[s]) for s in range(len(Ys))]
    Ms = torch.vmap(lambda y: dist(Y, y))(Ys)

    cpt = 0
    inner_log = False
    err_feature = 1e15
    err_structure = 1e15
    err_rel_loss = 0.

    if log:
        log_ = {}
        log_['err_feature'] = []
        log_['err_structure'] = []
        log_['Ts_iter'] = []

    while((err_feature > toly or err_structure > tolc) and cpt < max_iter):
        Cprev = C
        Yprev = Y

        with torch.no_grad():
            rho = rho.at(cpt) if isinstance(rho, Epsilon) else rho
            T = batch_fused_ACC_torch(Ms, C, Cs, p, ps, alpha=alpha, epoch=100, eps=1e-5, rho=rho)

        if not fixed_features:
            Y = batch_update_feature_matrix(lambdas, Ys, T, p)
            Ms = torch.vmap(lambda y: dist(Y, y))(Ys)

        if not fixed_structure:
            if loss_fun == 'square_loss':
                C = batch_update_square_loss(p, lambdas, T, Cs)

            elif loss_fun == 'kl_loss':
                C = batch_update_kl_loss(p, lambdas, T, Cs)

        # update convergence criterion
        err_feature, err_structure = 0., 0.
        if not fixed_features:
            err_feature = torch.norm(Y - Yprev)
        if not fixed_structure:
            err_structure = torch.norm(C - Cprev)
        if log:
            log_['err_feature'].append(err_feature)
            log_['err_structure'].append(err_structure)
            log_['Ts_iter'].append(T)

        if verbose:
            if cpt % 200 == 0:
                print('{:5s}|{:12s}'.format(
                    'It.', 'Err') + '\n' + '-' * 19)
            print('{:5d}|{:8e}|'.format(cpt, err_structure))
            print('{:5d}|{:8e}|'.format(cpt, err_feature))

        cpt += 1

    if log:
        log_['T'] = T
        log_['p'] = p
        log_['Ms'] = Ms

        return Y, C, log_
    else:
        return Y, C


def gm_barycenters(
    N, Ys, Cs, ps=None, p=None, lambdas=None, loss_fun='square_loss',
    max_iter=100, toly=1e-9, tolc=1e-9, verbose=False,
    log=False, init_C=None, init_Y=None, fixed_structure=False,
    fixed_features=False, seed=0, **kwargs
):

    if loss_fun not in ('square_loss', 'kl_loss'):
        raise ValueError(f"Unknown `loss_fun='{loss_fun}'`. Use one of: {'square_loss', 'kl_loss'}.")

    S = Ys.shape[0]
    if lambdas is None:
        lambdas = torch.full((S,), 1.0 / S).to(Ys)

    if p is None:
        p = torch.full((N,), 1.0 / N).to(Ys)
    
    if ps is None:
        ps = torch.full((S, N), 1.0 / N).to(Ys)

    d = Ys.shape[-1]  # dimension on the node features

    # Initialization of C : random euclidean distance matrix (if not provided by user)
    if fixed_structure:
        if init_C is None:
            raise ValueError('If C is fixed it must be initialized')
        else:
            C = init_C
    else:
        if init_C is None:
            torch.manual_seed(seed)
            xalea = torch.randn(N, 2).to(Cs[0])
            C = dist(xalea, xalea)
        else:
            C = init_C

    # Initialization of Y
    if fixed_features:
        if init_Y is None:
            raise ValueError('If Y is fixed it must be initialized')
        else:
            Y = init_Y
    else:
        if init_Y is None:
            Y = torch.zeros((N, d)).to(Cs[0])

        else:
            Y = init_Y

    # Ms = [dist(Y, Ys[s]) for s in range(len(Ys))]
    Ms = torch.vmap(lambda y: dist(Y, y))(Ys)
    Ms_cpu = Ms.cpu().numpy()
    Cs_cpu = [c.cpu().numpy() for c in Cs]
    edge_indices = [c.nonzero().contiguous().cpu().numpy() for c in Cs]

    cpt = 0
    inner_log = False
    err_feature = 1e15
    err_structure = 1e15
    err_rel_loss = 0.

    if log:
        log_ = {}
        log_['err_feature'] = []
        log_['err_structure'] = []
        log_['Ts_iter'] = []
    
    lambda_val = 80.0
    solver_params = {
        "maxIter": 100,
        "primalComputationInterval": 10,
        "timeout": 1000
    }

    while((err_feature > toly or err_structure > tolc) and cpt < max_iter):
        Cprev = C
        Yprev = Y

        with torch.no_grad():
            T = []
            for s in range(S):
                quadratic_costs = np.zeros((edge_indices[s].shape[0], edge_indices[s].shape[0]))
                Ts, _ = gm_solver(Ms_cpu[s], quadratic_costs, edge_indices[s], edge_indices[s], solver_params, verbose=False)
                Ts = torch.from_numpy(Ts).to(Ys)
                T.append(Ts)
            T = torch.stack(T, dim=0)

        if not fixed_features:
            Y = batch_update_feature_matrix(lambdas, Ys, T, p)
            Ms = torch.vmap(lambda y: dist(Y, y))(Ys)

        if not fixed_structure:
            if loss_fun == 'square_loss':
                C = batch_update_square_loss(p, lambdas, T, Cs)

            elif loss_fun == 'kl_loss':
                C = batch_update_kl_loss(p, lambdas, T, Cs)

        # update convergence criterion
        err_feature, err_structure = 0., 0.
        if not fixed_features:
            err_feature = torch.norm(Y - Yprev)
        if not fixed_structure:
            err_structure = torch.norm(C - Cprev)
        if log:
            log_['err_feature'].append(err_feature)
            log_['err_structure'].append(err_structure)
            log_['Ts_iter'].append(T)

        if verbose:
            if cpt % 200 == 0:
                print('{:5s}|{:12s}'.format(
                    'It.', 'Err') + '\n' + '-' * 19)
            print('{:5d}|{:8e}|'.format(cpt, err_structure))
            print('{:5d}|{:8e}|'.format(cpt, err_feature))

        cpt += 1

    if log:
        log_['T'] = T
        log_['p'] = p
        log_['Ms'] = Ms

        return Y, C, log_
    else:
        return Y, C


class Epsilon:
    """Epsilon scheduler for Sinkhorn and Sinkhorn Step."""

    def __init__(
        self,
        target: float = None,
        scale_epsilon: float = None,
        init: float = 1.0,
        decay: float = 1.0
    ):
        self._target_init = target
        self._scale_epsilon = scale_epsilon
        self._init = init
        self._decay = decay

    @property
    def target(self) -> float:
        """Return the final regularizer value of scheduler."""
        target = 5e-2 if self._target_init is None else self._target_init
        scale = 1.0 if self._scale_epsilon is None else self._scale_epsilon
        return scale * target

    def at(self, iteration: int = 1) -> float:
        """Return (intermediate) regularizer value at a given iteration."""
        if iteration is None:
            return self.target
        # check the decay is smaller than 1.0.
        decay = min(self._decay, 1.0)
        # the multiple is either 1.0 or a larger init value that is decayed.
        multiple = max(self._init * (decay ** iteration), 1.0)
        return multiple * self.target

    def done(self, eps: float) -> bool:
        """Return whether the scheduler is done at a given value."""
        return eps == self.target

    def done_at(self, iteration: int) -> bool:
        """Return whether the scheduler is done at a given iteration."""
        return self.done(self.at(iteration))
