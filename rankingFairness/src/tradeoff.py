import numpy as np
import math
import pdb
EPSILON=1e-6


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
    def __init__(self, ranking, num_docs, top_k, simulations, v=None) ->None:
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


    def setV(self) -> None:
        self.v = [1.0 / np.log2(2 + i) if i < self.top_k else 0 for i in range(self.num_docs)]
    
    
    def getUtil(self, merit_all, type='dcg') -> np.ndarray:
        if self.v is None:
            self.setV()
        ranking_ls=[]
        if len(self.ranking[0])<merit_all.shape[1]:
            for d, r in enumerate(self.ranking):
                idx_not_top_k = np.setdiff1d(np.arange(merit_all.shape[1]), r)
                ranking_ls.append(np.hstack((r, idx_not_top_k)))
        else:
            ranking_ls=self.ranking

        util_ls = [np.dot(merit_all[i,r].astype(float), self.v) for i,r in enumerate(ranking_ls)]
        assert len(util_ls)==len(self.ranking)
        assert isinstance(util_ls[0], float), f"{util_ls[0]}"
        return sum(util_ls)/len(self.ranking)

    def square(x):
        return np.power(x,2)/25
    
    def cube(x):
        return np.power(x,3)/math.pow(5,3)
    
    def exponential(x):
        return np.exp(x)/math.exp(5)
    
    def identity(x):
        return x

    def getCostArmsOld(self, start_minority_idx, merit_all, dist, funcApply=identity) -> None:
        merit_all = funcApply(merit_all)
        tmp_var=0.0
        assert self.draws==len(self.ranking)
        for d, r in enumerate(self.ranking):
            cost_majority, cost_minority, total_cost = 0.0, 0.0, 0.0
            ranked_ids = r[:self.top_k]
            not_selected_ids = np.setdiff1d(np.arange(self.num_docs), ranked_ids)
            if self.n_majority is None: 
                self.getExpectedRelevance(dist)
            
            for idx in not_selected_ids:
                if idx >= start_minority_idx:
                    cost_minority +=(merit_all[d, idx]/self.n_minority)
                else:
                    cost_majority +=(merit_all[d, idx]/self.n_majority)
                    tmp_var +=merit_all[d, idx]
                total_cost += (merit_all[d, idx]/(self.n_majority+self.n_minority))
            self.cost_majority.append(cost_majority)
            self.cost_minority.append(cost_minority)
            self.cost.append(total_cost)
        
        # normalize the cost with respect to expected number of relevant candidates
        assert len(self.cost_majority)==len(self.ranking)
        assert len(self.cost_minority)==len(self.ranking)
        assert len(self.cost)==len(self.ranking)
        self.cost_majority=sum(self.cost_majority)/len(self.cost_majority)
        self.cost_minority=sum(self.cost_minority)/len(self.cost_minority)
        self.cost=sum(self.cost)/len(self.cost)
    
    def getCostArms(self, start_minority_idx, merit_all, dist, funcApply=identity) -> None:
        merit_all = funcApply(merit_all)
        assert self.draws==self.ranking.shape[0]
        for d, r in enumerate(self.ranking):
            cost_majority, cost_minority, total_cost = 0.0, 0.0, 0.0
            ranked_ids = r[:self.top_k]
            not_selected_ids = np.setdiff1d(np.arange(self.num_docs), ranked_ids)
            if self.n_majority is None: 
                self.getExpectedRelevance(dist)
            
            for idx in not_selected_ids:
                if idx >= start_minority_idx:
                    cost_minority +=self.probs_all[idx]
                else:
                    cost_majority +=self.probs_all[idx]
                total_cost += self.probs_all[idx]
            self.cost_majority.append(cost_majority)
            self.cost_minority.append(cost_minority)
            self.cost.append(total_cost)
        # normalize the cost with respect to expected number of relevant candidates
        assert len(self.cost_majority)==self.draws
        assert len(self.cost_minority)==self.draws
        assert len(self.cost)==self.draws
        self.cost_majority = sum(self.cost_majority)/(self.draws*self.n_majority)
        self.cost_minority = sum(self.cost_minority)/(self.draws*self.n_minority)
        self.cost = sum(self.cost)/(self.draws*(self.n_minority + self.n_majority))


    def getExpectedRelevance(self, dist, funcApply=identity):
        """
        get n_A and n_B
        """
        majority_rel = np.array([d.getMean(funcApply=funcApply) for d in dist[0]])[:,None]
        minority_rel = np.array([d.getMean(funcApply=funcApply) for d in dist[1]])[:,None]
        self.n_majority = np.sum(majority_rel) #n_A
        self.n_minority = np.sum(minority_rel) #n_B
        self.n_group=[self.n_majority, self.n_minority]
        self.probs_all=np.vstack((majority_rel, minority_rel))
        assert len(self.probs_all)==self.num_docs

        # self.delta_max = max([self.probs_all[i]/self.n_group[j] for i,j in zip(np.arange(self.num_docs), [0]*len(dist[0])+ [1]*len(dist[1]))])
        self.delta_max = (np.sort(majority_rel, axis=0)[-1]/self.n_group[0] + np.sort(minority_rel, axis=0)[-1]/self.n_group[1])/2

    def EOR_constraint(self, start_minority_idx, dist):
        EOR=[]
        if self.n_majority is None: self.getExpectedRelevance(dist)
        for r in self.ranking:
            ids_majority_ranking = np.intersect1d(r[:self.top_k], np.arange(start_minority_idx))
            ids_minority_ranking = np.intersect1d(r[:self.top_k], np.arange(start_minority_idx, self.num_docs))
            if len(ids_minority_ranking)>0: 
                ids_minority_ranking -= start_minority_idx 
            n_majority_ranking = sum([d.getMean() for d in np.array(dist[0])[ids_majority_ranking]]) if len(ids_majority_ranking)>0 else 0
            n_minority_ranking = sum([d.getMean() for d in np.array(dist[1])[ids_minority_ranking]]) if len(ids_minority_ranking)>0 else 0
            EOR.append((n_majority_ranking/self.n_majority) - (n_minority_ranking/self.n_minority))
        
        
        return sum([np.abs(i) for i in EOR])/len(EOR), sum(EOR)/len(EOR), self.delta_max
    
