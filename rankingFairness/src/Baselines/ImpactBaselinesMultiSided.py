import cvxpy as cvx
import numpy as np
from scipy.stats import rankdata
from sklearn.utils import check_random_state

"""
Reference: Saito et al 2022, Paper:"Fair Impact as Fair Division"
"""

def exam_func(n_doc: int, K: int, shape: str = "inv") -> np.ndarray:
    assert shape in ["inv", "exp", "log"]
    if shape == "inv":
        v = np.ones(K) / np.arange(1, K + 1)
    elif shape == "exp":
        v = 1.0 / np.exp(np.arange(K))

    return v[:, np.newaxis]


def evaluate_pi(pi: np.ndarray, rel_mat: np.ndarray, v: np.ndarray):
    n_query, n_doc = rel_mat.shape
    expo_mat = (pi * v.T).sum(2)
    click_mat = rel_mat * expo_mat
    user_util = click_mat.sum() / n_query
    item_utils = click_mat.sum(0) / n_query
    nsw = np.power(item_utils.prod(), 1 / n_doc)

    max_envies = np.zeros(n_doc)
    for i in range(n_doc):
        u_d_swap = (expo_mat * rel_mat[:, [i] * n_doc]).sum(0)
        d_envies = u_d_swap - u_d_swap[i]
        max_envies[i] = d_envies.max() / n_query

    return user_util, item_utils, max_envies, nsw


def compute_pi_max(
    rel_mat: np.ndarray,
    v: np.ndarray,) -> np.ndarray:
    n_query, n_doc = rel_mat.shape
    K = v.shape[0]

    pi = np.zeros((n_query, n_doc, K))
    for k in np.arange(K):
        pi_at_k = np.zeros_like(rel_mat)
        pi_at_k[rankdata(-rel_mat, axis=1, method="ordinal") == k + 1] = 1
        pi[:, :, k] = pi_at_k

    return pi


def compute_pi_expo_fair(
    rel_mat: np.ndarray,
    v: np.ndarray,) -> np.ndarray:
    n_query, n_doc = rel_mat.shape
    K = v.shape[0]
    query_basis = np.ones((n_query, 1))
    am_rel = rel_mat.sum(0)[:, np.newaxis]
    am_expo = v.sum() * n_query * am_rel / rel_mat.sum()

    pi = cvx.Variable((n_query, n_doc * K))
    obj = 0.0
    constraints = []
    for d in np.arange(n_doc):
        pi_d = pi[:, K * d : K * (d + 1)]
        obj += rel_mat[:, d] @ pi_d @ v
        # feasible allocation
        basis_ = np.zeros((n_doc * K, 1))
        basis_[K * d : K * (d + 1)] = 1
        constraints += [pi @ basis_ <= query_basis]
        # amortized exposure
        constraints += [query_basis.T @ pi_d @ v <= am_expo[d]]
    # feasible allocation
    for k in np.arange(K):
        basis_ = np.zeros((n_doc * K, 1))
        basis_[np.arange(n_doc) * K + k] = 1
        constraints += [pi @ basis_ <= query_basis]
    constraints += [pi <= 1.0]
    constraints += [0.0 <= pi]

    prob = cvx.Problem(cvx.Maximize(obj), constraints)
    prob.solve(solver=cvx.SCS)

    pi = pi.value.reshape((n_query, n_doc, K))
    pi = np.clip(pi, 0.0, 1.0)

    return pi


def compute_pi_nsw(
    rel_mat: np.ndarray,
    v: np.ndarray,
    alpha: float = 0.0,) -> np.ndarray:
    n_query, n_doc = rel_mat.shape
    K = v.shape[0]
    query_basis = np.ones((n_query, 1))
    am_rel = rel_mat.sum(0) ** alpha

    pi = cvx.Variable((n_query, n_doc * K))
    obj = 0.0
    constraints = []
    for d in np.arange(n_doc):
        obj += am_rel[d] * cvx.log(rel_mat[:, d] @ pi[:, K * d : K * (d + 1)] @ v)
        # feasible allocation
        basis_ = np.zeros((n_doc * K, 1))
        basis_[K * d : K * (d + 1)] = 1
        constraints += [pi @ basis_ <= query_basis]
    # feasible allocation
    for k in np.arange(K):
        basis_ = np.zeros((n_doc * K, 1))
        basis_[np.arange(n_doc) * K + k] = 1
        constraints += [pi @ basis_ <= query_basis]
    constraints += [pi <= 1.0]
    constraints += [0.0 <= pi]

    prob = cvx.Problem(cvx.Maximize(obj), constraints)
    prob.solve(solver=cvx.SCS)

    pi = pi.value.reshape((n_query, n_doc, K))
    pi = np.clip(pi, 0.0, 1.0)

    return pi


def compute_pi_unif(rel_mat: np.ndarray, v: np.ndarray) -> np.ndarray:
    n_query, n_doc = rel_mat.shape
    K = v.shape[0]

    return np.ones((n_query, n_doc, K)) / n_doc