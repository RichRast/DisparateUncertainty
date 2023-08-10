import numpy as np
import math
import pdb
from rankingFairness.src.decorators import timer
EPSILON=1e-12


def getProbsDict(num_docs, dist):
    prob_by_groups ={}
    n_group={}
    ids = [i for i in range(num_docs)]
    for i in range(len(dist)):
        prob_by_groups[i] = [d.getMean() for d in dist[i]] #probs[0], probs[1]
        n_group[i] = sum(prob_by_groups[i]) #n_A, n_B as n[0], n[1]
    
    probs_all = np.hstack(([prob_by_groups[i] for i in range(len(dist))]))
    probs_all = {i:j for (i,j) in zip(ids, probs_all)}
    return probs_all, n_group


class UtilityCost():
    def __init__(self, ranking, num_docs, top_k, simulations, groups=2, n_labels=False, v=None) ->None:
        super().__init__()
        self.ranking = ranking
        self.v = v
        self.top_k = top_k
        self.num_docs = num_docs
        self.groups = groups
        self.cost_groups = np.full(self.groups, np.inf)
        self.mask_total = None
        self.n_labels=n_labels
        self.total_cost_std=None
        self.group_cost_std=np.full(self.groups, np.inf)

    def setV(self) -> None:
        self.v = [1.0 / np.log2(2 + i) if i < self.top_k else 0 for i in range(self.num_docs)]
    

    def getUtil(self, dist, merit_all, type='dcg') -> np.ndarray:
        if self.v is None:
            self.setV()
        ranking_ls=[]
        util_ls = []
        for i in np.arange(self.ranking.shape[0]):
            if self.n_labels and (merit_all is not None):
                util_ls.append(np.dot(merit_all[:,self.ranking[i]].astype(float), np.array(self.v)).sum()/merit_all.shape[0])
            else:
                if self.mask_total is None:
                    self.getExpectedRelevance(dist, merit_all)
                util_ls.append(np.dot(self.probs_items[self.ranking[i]].astype(float), np.array(self.v)).sum())
        return sum(util_ls)/self.ranking.shape[0]

    def square(x):
        return np.power(x,2)/25
    
    def cube(x):
        return np.power(x,3)/math.pow(5,3)
    
    def exponential(x):
        return np.exp(x)/math.exp(5)
    
    def identity(x):
        return x


    def getCostArms(self, dist, merit_all, funcApply=identity) -> None:
        assert self.num_docs==self.ranking.shape[1]
        if self.mask_total is None: 
            self.getExpectedRelevance(dist, merit_all)
        total_cost = 0.0
        cost_groups = np.zeros((self.groups))
        cost_groups_full={}
                
        if self.n_labels and (merit_all is not None):
            total_cost = np.dot(self.mask_total, merit_all.T).sum(axis=-1)/merit_all.shape[0] #RXn dot nXS -> (RXS) ->R/S
        else:
            total_cost = np.dot(self.mask_total, self.probs_items) #RXn dot n -> R

        for g in range(self.groups):
            if self.n_labels and (merit_all is not None):
                cost_groups[g] = np.dot(self.group_mask[:,g], merit_all.T).sum()/merit_all.shape[0]
                cost_groups_full[g] = np.dot(self.group_mask[:,g], merit_all.T).sum(axis=-1)/merit_all.shape[0]
            else:
                cost_groups[g] = np.dot(self.group_mask[:,g], self.probs_items).sum() #SXn dot n -> S
                cost_groups_full[g] = np.dot(self.group_mask[:,g], self.probs_items)
        
        
        self.total_cost_std=np.std(1-(total_cost/sum([self.n_group[g] for g in range(self.groups)])), axis=0)
        total_cost = total_cost.sum()/self.ranking.shape[0]
        for g in range(self.groups):
            cost_groups[g] /= self.ranking.shape[0]
            self.cost_groups[g] = 1-(cost_groups[g]/(self.n_group[g]))
            self.group_cost_std[g]=np.std(1-(cost_groups_full[g]/(self.n_group[g])), axis=0)
        
        self.cost = 1-(total_cost/sum([self.n_group[g] for g in range(self.groups)]))

 
    def getExpectedRelevance(self, dist, merit_all=None, funcApply=identity):
        """
        get n_g for all groups
        """
        self.num_ids = 0
        for g in range(self.groups):
            self.num_ids += len(dist[g]) 
        self.rel_groups = {}
        
        self.group_ids = []
        for g, d in enumerate(dist):
            for _ in range(len(d)):
                self.group_ids.append(g)

        self.getMaskRanking()

        if self.n_labels and (merit_all is not None):
            # assert merit_all.shape[0]==1
            self.n_group = np.full(self.groups, 0.0)
            assert merit_all.shape[1]==self.num_docs
            for g in range(self.groups):
                group_ids_mask = np.array(self.group_ids)==g
                self.rel_groups[g] = np.mean((group_ids_mask* merit_all), axis=0)
                assert self.rel_groups[g].shape[0]==self.num_docs
                self.n_group[g]=sum(self.rel_groups[g])

        else:
            self.probs_all, self.n_group = getProbsDict(self.num_ids, dist)
            self.probs_items = np.array(list(self.probs_all.values()))
            for g in range(self.groups):
                self.rel_groups[g]=np.array([d.getMean(funcApply=funcApply) for d in dist[g]])
        
        if self.groups==2:      
            self.delta_max = sum([np.max(self.rel_groups[g], axis=0)/self.n_group[g] for g in range(self.groups)])/self.groups
        else:
            self.delta_max = max([np.max(self.rel_groups[g], axis=0)/self.n_group[g] for g in range(self.groups)])


    def getMaskRanking(self):
        positions = np.zeros_like(self.ranking)
        self.group_mask = np.zeros((self.ranking.shape[0], self.groups, self.num_docs))
        self.mask_total = np.zeros((self.ranking.shape[0], self.num_docs))

        for i, r in enumerate(self.ranking):
            # To create a matrix where in row i, columns show the ranking of the item at index j
            for j in np.arange(self.ranking.shape[1]):
                positions[i, r[j]]=j
        # To create a mask where in row i, columns have True for items j which are in top_k ranking
        self.mask_total = (positions < self.top_k)
        for g in range(self.groups):
            # Copy the mask to group mask, currently has all groups
            self.group_mask[:, g, :] = self.mask_total.copy()
            # Identify which items have group = g
            idxs_group = (np.array(self.group_ids)==g)
            # Make mask = 0 for all items not in group g
            self.group_mask[:, g, ~idxs_group] = 0


    def EOR_constraint(self, dist, merit_all):
        if self.mask_total is None:
            self.getExpectedRelevance(dist, merit_all)

        n_group_ranking = np.full((self.groups, self.ranking.shape[0]), 0.0)
        for g in range(self.groups):
            if self.n_labels and (merit_all is not None):
                n_group_ranking[g,:] = np.dot(self.group_mask[:,g], merit_all.T).sum(axis=1)/(self.n_group[g]*merit_all.shape[0])
            else:
                n_group_ranking[g,:] = np.dot(self.group_mask[:,g], self.probs_items)/self.n_group[g]
        # for Two Group EOR
        if self.groups==2:
            EOR = n_group_ranking[0,:]-n_group_ranking[1,:]
        else:
            EOR = np.max(n_group_ranking, axis=0)-np.min(n_group_ranking, axis=0) #always +ve, absolute value of delta in 2 group case
        EOR_avg = np.mean(EOR)
        
        return EOR_avg, self.delta_max, np.std(EOR, axis=0)
    
