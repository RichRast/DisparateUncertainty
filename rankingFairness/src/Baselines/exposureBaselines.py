import cvxpy as cp
import numpy as np 
from copy import deepcopy
import itertools
import matplotlib.pyplot as plt


"""
Reference: https://github.com/MilkaLichtblau/BA_Laura/tree/master/src/algorithms/FOEIR,
https://github.com/dmh43/fair-ranking/blob/master/fairness_of_exposure.py"
"""


def getPiMatrix(rel: np.ndarray,
        grp_arr: np.ndarray,
        grp_rel: dict,
        verbose=True
    ) -> (np.ndarray, float):
    """
    params rel: relevance array of n items (n, 1)
    return 
    P: doubly stochastic matrix items x ranks (n,n)
    objetcive value: DCG, float
    """
    num_items = rel.shape[0]
    v=np.array([1.0 / np.log2(2 + i) for i in range(num_items)])[:,None]
    am_rel = v.sum()/rel.sum()
    sum_basis_ = np.ones((num_items,1))
    grps=np.array(list(grp_rel.keys()))
    # max_iters=100000
    exp_thresh=1.0
    decrement=0.01
    feasible_flag=False

    while(not feasible_flag):
        P = cp.Variable((num_items, num_items))
        obj = rel.T @ P @ v
        exp_by_items = P @ v

        # pairwise constraints
        exp_by_group = {}
        for i in grps:
            basis_ = np.zeros((num_items,1))
            basis_[grp_arr==i,:]=1
            exp_by_group[i] = basis_.T @ exp_by_items / grp_rel[i]

        constraints=[P>=0.0,
        P<=1.0,  
        sum_basis_.T @ P == sum_basis_.T,
        P @ sum_basis_ == sum_basis_ 
        ]

        for (i,j) in itertools.combinations(grps, r=2):
            constraints += [exp_by_group[i] >= exp_thresh * exp_by_group[j]]
            constraints += [exp_by_group[j] >= exp_thresh * exp_by_group[i]]

        prob = cp.Problem(cp.Maximize(obj), constraints)

        result = prob.solve(verbose=False, solver=cp.SCS)
        if prob.status in ["infeasible", "unbounded"]:
            exp_thresh -= decrement
        else:
            feasible_flag=True
            if verbose:
                print(f"found feasible solution at exp_thresh:{exp_thresh}")

    return P.value, prob.value

def getPiMatrix_DP_exp(rel: np.ndarray,
        grp_arr: np.ndarray,
        grp_rel: dict,
        verbose=True,
    ) -> (np.ndarray, float):
    """
    params rel: relevance array of n items (n, 1)
    return 
    P: doubly stochastic matrix items x ranks (n,n)
    objetcive value: DCG, float
    """
    num_items = rel.shape[0]
    v=np.array([1.0 / np.log2(2 + i) for i in range(num_items)])[:,None]
    sum_basis_ = np.ones((num_items,1))
    grps=np.array(list(grp_rel.keys()))
    max_iters=100000
    exp_thresh=1.0
    decrement=0.01
    feasible_flag=False

    while(not feasible_flag):
        P = cp.Variable((num_items, num_items))
        obj = rel.T @ P @ v
        exp_by_items = P @ v

        # pairwise constraints
        exp_by_group = {}
        for i in grps:
            basis_ = np.zeros((num_items,1))
            basis_[grp_arr==i,:]=1
            exp_by_group[i] = basis_.T @ exp_by_items 

        constraints=[P>=0.0,
        P<=1.0,  
        sum_basis_.T @ P == sum_basis_.T,
        P @ sum_basis_ == sum_basis_ 
        ]

        for (i,j) in itertools.combinations(grps, r=2):
            constraints += [exp_by_group[i] >= exp_thresh * exp_by_group[j]]
            constraints += [exp_by_group[j] >= exp_thresh * exp_by_group[i]]

        prob = cp.Problem(cp.Maximize(obj), constraints)

        result = prob.solve(verbose=False, solver=cp.SCS, max_iters=max_iters)
        if prob.status in ["infeasible", "unbounded"]:
            exp_thresh -= decrement
        else:
            feasible_flag=True
            if verbose:
                print(f"found feasible solution at exp_thresh:{exp_thresh}")

    return P.value, prob.value

def getExposureMetrics(rel: np.ndarray,
        grp_arr: np.ndarray,
        grp_rel: dict,
        merits=None,
        dp=False,
        verbose=True) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    params rel: relevance array of n items (n, 1)
    params grp_arr: group membership of n items (n,)
    params grp_rel: dict with keys as groups and values as sum of relevance by grps
    params merits: merits array of n items (n, 1)
    
    return 
    EOR: (n,),
    total_cost: (n,),
    group_cost:(G, n) for {1,..,G} groups
    DCG: (n,)
    """
    assert rel.shape[1]==1
    num_items = rel.shape[0]
    if dp:
        P, obj = getPiMatrix_DP_exp(rel, grp_arr, grp_rel, verbose=verbose)
    else:
        P, obj = getPiMatrix(rel, grp_arr, grp_rel, verbose=verbose)
    # P = np.clip(P, 0.0, 1.0)
    if verbose:
        plt.imshow(P)
        print(f"objective:{obj}")
    if merits is None:
        merits=rel
    else:
        assert merits.shape[1]==1

    grp_merits={}
    grps=np.array(list(grp_rel.keys()))
    
    for i in grps:
        grp_merits[i]=merits[grp_arr==i].sum()
    

    total_cost_d = np.full((num_items,1), 1/merits.sum())

    group_d=np.zeros((num_items,1))
    for i,g in enumerate(grp_arr):
        group_d[i,:] = 1/grp_merits[g]

    v = np.array([1.0 / np.log2(2 + i) for i in range(num_items)])[:,None]
    EOR_g = np.full((len(grps), num_items), np.inf)

    group_cost = np.full((len(grps),num_items), np.inf)
    DCG = np.full((num_items), np.inf)
    total_cost= np.full((num_items), np.inf)
    
    # calculate total, group cost, EOR for each group.Then you can add, subtract max - min for EOR
    # If P is items x ranks, then mask will be on items. Mask will be 1 where items in group G
    grp_mask = np.zeros((num_items, 1))
    pos_mask = np.zeros((num_items, 1))
    
    upper_bound=np.ones((num_items)).astype(float)

    for k in range(num_items):
        pos_mask[:k+1,:]=1
        for g in grps:
            grp_mask_copy = deepcopy(grp_mask)
            grp_mask_copy[grp_arr==g]=1
            group_denom = np.multiply(group_d, grp_mask_copy)
            EOR_g[g,k] =  np.multiply(merits, group_denom).T @ P @  pos_mask
            assert np.any(EOR_g <= upper_bound)
            group_cost[g,k] =  np.multiply(merits, group_denom).T @ P @  pos_mask
        total_cost[k] = np.multiply(merits, total_cost_d).T @ P @  pos_mask
        DCG[k] = merits.T @ P @ np.multiply(v, pos_mask)

    group_cost = 1-group_cost
    total_cost=1-total_cost
    if len(grps)==2:
        EOR = EOR_g[0,:]-EOR_g[1,:]
    else:
        max_EOR_group = np.max(EOR_g, axis=0)
        min_EOR_group = np.min(EOR_g, axis=0)
        EOR = max_EOR_group - min_EOR_group
    
    assert EOR.shape==(num_items,)
    EOR_abs = np.abs(EOR)
    return EOR, total_cost, group_cost, DCG, EOR_abs
    