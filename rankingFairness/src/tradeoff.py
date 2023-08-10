import numpy as np
import math
import pdb
import collections.abc
EPSILON=1e-12


def getProbsDict(num_docs, dist):
    #ToDo generalize this to any number of groups
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
    def __init__(self, ranking, num_docs, top_k, simulations, start_minority_idx, n_labels=False, v=None) ->None:
        super().__init__()
        self.ranking=ranking
        self.v=v
        self.top_k = top_k
        self.num_docs = num_docs
        self.n_majority, self.n_minority = None, None
        self.cost_majority=[]
        self.cost_minority=[]
        self.cost = []
        self.draws=simulations
        self.n_labels=n_labels
        self.start_minority_idx=start_minority_idx


    def setV(self) -> None:
        self.v = [1.0 / np.log2(2 + i) if i < self.top_k else 0 for i in range(self.num_docs)]
    
    
    def getUtil(self, dist, merit_all, type='dcg') -> np.ndarray:
        if self.v is None:
            self.setV()
        ranking_ls=[]
        util_ls = []
        for i in np.arange(self.ranking.shape[0]):
            if self.n_labels and (merit_all is not None):
                util_ls.append(np.dot(merit_all[:,self.ranking[i]].astype(float),np.array(self.v)).sum()/merit_all.shape[0])
            else:
                if self.n_majority is None:
                    self.getExpectedRelevance(dist, merit_all)
                util_ls.append(np.dot(self.probs_all[self.ranking[i]].astype(float), np.array(self.v)).sum())
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
        merit_all = funcApply(merit_all)
        # assert self.draws==self.ranking.shape[0]
        assert self.num_docs==self.ranking.shape[1]

        if self.n_majority is None: 
            self.getExpectedRelevance(dist, merit_all)
        positions = np.zeros_like(self.ranking)
        cost_majority, cost_minority, total_cost = 0.0, 0.0, 0.0
        for i in np.arange(self.ranking.shape[0]):
            for j in np.arange(self.ranking.shape[1]):
                positions[i,self.ranking[i,j]]=j
            mask_total = (positions[i,:] >= self.top_k)

            if self.n_labels and (merit_all is not None):
                total_cost += np.dot(mask_total, merit_all.T).sum()/merit_all.shape[0]
            else:
                total_cost += np.dot(mask_total, self.probs_all).sum()

            majority_mask = mask_total.copy()
            majority_mask[self.start_minority_idx:] = 0
            
            minority_mask = mask_total.copy()
            minority_mask[:self.start_minority_idx] = 0
            if self.n_labels and (merit_all is not None):
                cost_majority += np.dot(majority_mask, merit_all.T).sum()/merit_all.shape[0]
                cost_minority += np.dot(minority_mask, merit_all.T).sum()/merit_all.shape[0]
            else:
                cost_majority += np.dot(majority_mask, self.probs_all).sum()
                cost_minority += np.dot(minority_mask, self.probs_all).sum()
        
        total_cost /= self.ranking.shape[0]
        cost_majority /= self.ranking.shape[0]
        cost_minority /= self.ranking.shape[0]
        self.cost_majority = cost_majority/(self.n_majority)
        self.cost_minority = cost_minority/(self.n_minority)
        self.cost = total_cost/(self.n_minority + self.n_majority)


    def getExpectedRelevance(self, dist, merit_all=None, funcApply=identity):
        """
        get n_A and n_B
        """
        if self.n_labels and (merit_all is not None):
            assert merit_all.shape[0]==1
            assert merit_all.shape[1]==self.num_docs
            self.majority_rel = merit_all[:,:self.start_minority_idx].squeeze(0)
            self.minority_rel = merit_all[:,self.start_minority_idx:].squeeze(0)
        else:
            self.majority_rel = np.array([d.getMean(funcApply=funcApply) for d in dist[0]])
            self.minority_rel = np.array([d.getMean(funcApply=funcApply) for d in dist[1]])
            self.probs_all=np.hstack((self.majority_rel, self.minority_rel))
            assert len(self.probs_all)==self.num_docs
        self.n_majority = np.sum(self.majority_rel) #n_A
        self.n_minority = np.sum(self.minority_rel) #n_B
        self.n_group=[self.n_majority, self.n_minority]

        self.delta_max = (np.max(self.majority_rel, axis=0)/self.n_group[0] + np.max(self.minority_rel, axis=0)/self.n_group[1])/2
        assert not isinstance(self.delta_max, collections.abc.Sequence)

    def EOR_constraint(self, dist):
        EOR=[]
        if self.n_majority is None: self.getExpectedRelevance(dist)
        for r in self.ranking:
            ids_majority_ranking = np.intersect1d(r[:self.top_k], np.arange(self.start_minority_idx))
            ids_minority_ranking = np.intersect1d(r[:self.top_k], np.arange(self.start_minority_idx, self.num_docs))
            if len(ids_minority_ranking)>0: 
                ids_minority_ranking -= self.start_minority_idx 
            n_majority_ranking = np.sum(self.majority_rel[ids_majority_ranking]) if len(ids_majority_ranking)>0 else 0
            n_minority_ranking = np.sum(self.minority_rel[ids_minority_ranking]) if len(ids_minority_ranking)>0 else 0
            EOR.append((n_majority_ranking/self.n_majority) - (n_minority_ranking/self.n_minority))
        
        return sum([np.abs(i) for i in EOR])/len(EOR), sum(EOR)/len(EOR), self.delta_max
    
