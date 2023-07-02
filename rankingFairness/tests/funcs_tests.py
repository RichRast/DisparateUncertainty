import numpy as np
import pytest 
from rankingFairness.src.tradeoff import UtilityCost
from rankingFairness.src.distributions import Bernoulli
from hypothesis import given, event, target, note
from strategies import small_K
import pdb

# Checks
# Whether the draws of relevances means converge to the mean of distribution it is drawn from. It checks whether the sampling of relevances is done currently
# task 1_1 Check the EO_constraint calculation for a given ranking
# check the majority, moniority and total cost for a given ranking and a given d number of draws of relevances
# check the utility (DCG) given a particular ranking
# check that for uniform ranking - given the variance of cost, the majority and minority cost are somewhat equal
# check that for EOR - given the variance of cost, the majority and minority cost are somewhat equal
# check that PRP always has the highest utility for a given dataset

class UtilCostRanking1(UtilityCost):
    def __init__(self, top_k) -> None:
        self.ranking = [[0,4,5,1,6,7]]
        self.top_k = top_k
        self.num_docs = 8
        self.n_majority, self.n_minority = None, None


class BernoulliDist():
    def __init__(self) -> None:
        self.dist_A = [Bernoulli(p) for p in [0.9,0.9,0.1,0.1]] 
        self.dist_B = [Bernoulli(p) for p in [0.5,0.5,0.5,0.5]] 

@pytest.mark.task1_1
def test_EO():
    "check proper calculation of EO for a small ranking"
    ranking_obj = UtilCostRanking1(top_k=3)
    dist_obj = BernoulliDist()
    
    np.testing.assert_almost_equal(ranking_obj.EOR_constraint(start_minority_idx=4, dist=[dist_obj.dist_A, dist_obj.dist_B]), 0.05)


@given(small_K)
@pytest.mark.task1_2
def test_EO_max(small_K):
    "check that for every k, EO is within delta_max"
    ranking_obj = UtilCostRanking1(top_k=small_K)
    dist_obj = BernoulliDist()
    assert ranking_obj.EOR_constraint(start_minority_idx=4, dist=[dist_obj.dist_A, dist_obj.dist_B]) <= 0.45


@pytest.mark.task2_1
def test_getDCG():
    "check the calculation of DCG for a specific k and for any k"
    pass