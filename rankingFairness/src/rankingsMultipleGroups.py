import numpy as np
import random
from abc import ABC, abstractmethod
import math
import itertools
from scipy.optimize import linprog
import pdb
from rankingFairness.src.tradeoff import getProbsDict
from rankingFairness.src.utils import MaxPriorityQueue
from rankingFairness.src.decorators import timer
from tqdm import tqdm

EPSILON=1e-12

class Ranker(ABC):
    def __init__(self, dist, ucb=None, distType=None, switch_start=False) -> None:
        super().__init__()
        self.dist = dist
        self.ucb=ucb
        self.distType=distType
        self.switch_start=switch_start

    @abstractmethod
    def rank(self) -> np.ndarray:
        pass


class PRP_Ranker(Ranker):
    def __init__(self, dist, ucb=None, distType=None, switch_start=False) -> None:
        super().__init__(dist, ucb, distType)
        self.queue = MaxPriorityQueue()
        self.formQueue()

    @staticmethod
    def name():
        return 'PRP'
    
    def formQueue(self):
        pMeans_ls=[]
        for i in range(len(self.dist)):
            pMeans_ls.append([p.getMean() for p in self.dist[i]])
        pMeans = np.concatenate((pMeans_ls))
        for ids, val in zip(np.arange(len(pMeans)), pMeans):
            self.queue.add(val, ids)
    
    def rank(self, top_k) -> np.ndarray:

        self.ranking = np.array([self.queue.pop_max()[1] for _ in range(top_k)])
        
class Uniform_Ranker(Ranker):
    def __init__(self, dist, ucb=None , distType=None, switch_start=False) -> None:
        super().__init__(dist, ucb, distType)
        

    @staticmethod
    def name():
        return 'Uniform'
    
    def rank(self, top_k) -> np.ndarray:
        self.ranking = np.random.choice(np.arange(top_k), top_k, replace=False)

class TS_RankerII(Ranker):
    def __init__(self, dist, ucb=None , distType=None, switch_start=False) -> None:
        super().__init__(dist, ucb, distType)
        """
        This class takes the drawn merits and randomizes for all the students with
        merit=1
        """

    @staticmethod
    def name():
        return 'TS'
    
    def sampleMerits(self):
        meritObj = self.distType(self.dist)
        return meritObj.sample()
    
    def rank(self, top_k, simulations) -> np.ndarray:
        self.ranking = np.zeros((simulations, top_k))
        merits = np.full((simulations, top_k), np.inf)
        for e in tqdm(range(simulations)):
            merits[e,:] = self.sampleMerits()
        b=np.random.random((simulations, top_k))
        self.ranking=np.lexsort((b,merits), axis=1)[:,::-1]
        

class UCB_Ranker(Ranker):
    pass

class EO_Ranker(Ranker):
    def __init__(self, dist, ucb=None, distType=None, switch_start=False) -> None:
        super().__init__(dist, ucb, distType)
        self.num_ids = len(dist[0]) + len(dist[1])
        self.ids = np.array([i for i in range(self.num_ids)])
        self.start_minority_idx = len(dist[0])
        probs_all, n_group = self.getExpectedRel(dist)
        self.n_group = n_group
        self.probs_all = probs_all
        # self.delta = max([self.probs_all[i]/self.n_group[j] for i,j in zip(self.ids, [0]*len(dist[0])+ [1]*len(dist[1]))])
        majority_rel = np.array([d.getMean() for d in dist[0]])[:,None]
        minority_rel = np.array([d.getMean() for d in dist[1]])[:,None]
        self.delta = ((np.sort(majority_rel, axis=0)[-1]/self.n_group[0]) + (np.sort(minority_rel, axis=0)[-1]/self.n_group[1]))/2
        self.queue=[]
        for i in range(len(dist)):
            self.queue.append(MaxPriorityQueue())
            self.formQueue(i)
        print(self.delta)
        self.arm_a=0.0
        self.arm_b=0.0

    @staticmethod
    def name():
        return "EORI"
    
    def formQueue(self, g):
        if self.ucb is not None:
            pMeans=self.ucb[g]
        else:
            pMeans=np.array([p.getMean() for p in self.dist[g]])
        for ids, val in zip(np.arange(len(pMeans)), pMeans):
            self.queue[g].add(val, ids)

    def getExpectedRel(self, dist):
        probs_all, n_group = getProbsDict(self.num_ids, dist)
        return probs_all, n_group

    @timer
    def rank(self, top_k=None) -> np.ndarray:
        EOR_monitor =[]
        self.ranking = []
        j = 0
        
        while j < top_k:
            EOR_compare=[]
            a,b=None, None
            assert len(self.queue[0]) + len(self.queue[1]) > 0 
            if len(self.queue[0]) > 0 : 
                a = self.queue[0].peek_max()[1]
                EOR_compare.append(self.getEOR(self.ranking + [a]))
            if len(self.queue[1]) > 0 : 
                b = self.start_minority_idx + self.queue[1].peek_max()[1]
                EOR_compare.append(self.getEOR(self.ranking + [b]))
            
            EO_satisfy_indices = np.argwhere(EOR_compare<=self.delta)[:,0]
            assert len(EO_satisfy_indices)>0, f"EOR:{EOR_compare}"
            if (len(EO_satisfy_indices)==2):
                if (abs(EOR_compare[0]-EOR_compare[1])<=EPSILON):
                    selected_idx = np.random.choice(np.arange(len(EO_satisfy_indices)), 1).item()
                    print(selected_idx)
                elif self.probs_all[a]>self.probs_all[b]:
                    selected_idx=0
                else:
                    selected_idx=1
                if selected_idx==0:
                    selected_arm = a
                    self.queue[0].pop_max()
                    self.arm_a=self.probs_all[a]
                else:
                    selected_arm = b
                    self.queue[1].pop_max()
                    self.arm_b=self.probs_all[b]
            elif (EO_satisfy_indices[0]==0) and (a is not None) and (len(EO_satisfy_indices)==1):
                selected_arm = a
                self.queue[0].pop_max()
                self.arm_a=self.probs_all[a]
            else:
                selected_arm = b
                self.queue[1].pop_max()
                self.arm_b=self.probs_all[b]

            self.ranking.append(selected_arm)
            j+=1
            EOR_monitor.append(self.getEOR(self.ranking))

        return EOR_monitor

    def getEOR(self, ranking):
        # add unit test for EOR calculation
        ids_majority_ranking = np.intersect1d(ranking, self.ids[:self.start_minority_idx])
        ids_minority_ranking = np.intersect1d(ranking, self.ids[self.start_minority_idx:])
        if len(ids_minority_ranking)>0: 
            ids_minority_ranking -= self.start_minority_idx
        n_majority_ranking = sum([d.getMean() for d in np.array(self.dist[0])[ids_majority_ranking]]) if len(ids_majority_ranking)>0 else 0
        n_minority_ranking = sum([d.getMean() for d in np.array(self.dist[1])[ids_minority_ranking]]) if len(ids_minority_ranking)>0 else 0
        EOR = (n_majority_ranking/self.n_group[0]) - (n_minority_ranking/self.n_group[1])
        return np.abs(EOR)

class EO_RankerII(Ranker):
    def __init__(self, dist, ucb=None, distType=None, switch_start=False) -> None:
        super().__init__(dist, ucb, distType)
        self.groups = len(dist)
        self.num_ids = 0
        for g in range(self.groups):
            self.num_ids += len(dist[g]) 
        self.ids = np.array([i for i in range(self.num_ids)])
        self.probs_all, self.n_group = self.getExpectedRel(dist)

        rel_groups = []
        for g in range(self.groups):
            rel_groups.append(np.array([d.getMean() for d in dist[g]])[:,None])
        
        self.switch_start = switch_start

        self.group_ids = []
        for g, d in enumerate(dist):
            for _ in range(len(d)):
                self.group_ids.append(g)
        
        self.queue=[]
        for i in range(self.groups):
            self.queue.append(MaxPriorityQueue())
            self.formQueue(i)
        

    @staticmethod
    def name():
        return "EOR"
    
    def formQueue(self, g):
        if self.ucb is not None:
            pMeans=self.ucb[g]
        else:
            pMeans=np.array([p.getMean() for p in self.dist[g]])

        g_idx = np.argwhere(np.array(self.group_ids) == g).squeeze(1)
        idx = self.ids[g_idx]
        
        for ids, val in zip(idx, pMeans):
            self.queue[g].add(val, ids)

    def getExpectedRel(self, dist):
        probs_all, n_group = getProbsDict(self.num_ids, dist)
        return probs_all, n_group

    @timer
    def rank(self, top_k=None) -> np.ndarray:
        
        self.ranking = []
        j = 0
        self.n_group_ranking = np.full(self.groups, 0.0)
        
        self.last_element = np.full(self.groups, 0.0)
        #fill up the last element with the first element of that group
        for g in range(self.groups):
            self.last_element[g] = self.queue[g].peek_max()[0]
        
        while j < top_k:
            arm = np.full(self.groups, np.inf)
            EOR_compare = np.full(self.groups, np.inf)
            
            assert any([len(self.queue[g]) for g in range(self.groups)]) > 0 
            
            for g in range(self.groups):
                if len(self.queue[g]) > 0 : 
                    arm[g] = self.queue[g].peek_max()[1]
                    EOR_compare[g] = self.getEOR(arm[g].item())

            # print(f"EOR_compare:{EOR_compare}, k:{j}")
            selected_groups = np.where(EOR_compare==np.min(EOR_compare))[0]
            assert len(selected_groups)>=1
            if len(selected_groups)>1:
                selected_group=np.random.choice(selected_groups,1).item()
                # print(f"randomly selected group:{selected_group}, when these groups had same val:{selected_groups}")
            else:
                selected_group = selected_groups.item()
            
            # print(f"selected_group:{selected_group}")
            selected_arm = self.queue[selected_group].peek_max()[1]
            # print(f"selected_arm:{selected_arm}")
            self.n_group_ranking[selected_group] += self.probs_all[selected_arm]/self.n_group[selected_group]
            self.queue[selected_group].pop_max()
            self.ranking.append(selected_arm)
            self.last_element[selected_group] = self.probs_all[selected_arm]
            
            j+=1
        

    def getEOR(self, i):
        # add unit test for EOR calculation
        n_group_ranking = self.n_group_ranking.copy()
        g = self.group_ids[int(i)]
        n_group_ranking[g] += self.probs_all[i]/self.n_group[g]
        EOR = np.max(n_group_ranking, axis=0)-np.min(n_group_ranking, axis=0)
        return EOR.item()

class DP_Ranker(Ranker):
    def __init__(self, dist, ucb=None, distType=None, switch_start=False) -> None:
        super().__init__(dist, ucb, distType)
        self.groups = len(dist)
        self.num_ids = 0
        for g in range(self.groups):
            self.num_ids += len(dist[g]) 
        self.ids = np.array([i for i in range(self.num_ids)])

        self.group_ids = []
        self.s_group=np.full(self.groups,0)
        for g, d in enumerate(dist):
            for _ in range(len(d)):
                self.group_ids.append(g)
            self.s_group[g]=len(d)
        
        self.queue=[]
        for i in range(self.groups):
            self.queue.append(MaxPriorityQueue())
            self.formQueue(i)

    @staticmethod
    def name():
        return "DP"

    def formQueue(self, g):
        if self.ucb is not None:
            pMeans=self.ucb[g]
        else:
            pMeans=np.array([p.getMean() for p in self.dist[g]])

        g_idx = np.argwhere(np.array(self.group_ids) == g).squeeze(1)
        idx = self.ids[g_idx]
        
        for ids, val in zip(idx, pMeans):
            self.queue[g].add(val, ids)


    def rank(self, top_k=None) -> np.ndarray:
        self.ranking = []
        j = 0
        self.s_group_ranking = np.full(self.groups, 0.0)
        while j < top_k:
            arm = np.full(self.groups, np.inf)
            DPR_compare = np.full(self.groups, np.inf)
            
            assert any([len(self.queue[g]) for g in range(self.groups)]) > 0 
            
            for g in range(self.groups):
                if len(self.queue[g]) > 0 : 
                    DPR_compare[g] = self.getDPR(g)

            # print(f"DPR_compare:{DPR_compare}, k:{j}")
            selected_groups = np.where(DPR_compare==np.min(DPR_compare))[0]
            assert len(selected_groups)>=1
            if len(selected_groups)>1:
                selected_group=np.random.choice(selected_groups,1).item()
                # print(f"randomly selected group:{selected_group}, when these groups had same val:{selected_groups}")
            else:
                selected_group = selected_groups.item()
            # print(f"selected_group:{selected_group}")
            selected_arm = self.queue[selected_group].peek_max()[1]
            # print(f"selected_arm:{selected_arm}")
            
            self.s_group_ranking[selected_group] += 1/self.s_group[selected_group]
            self.queue[selected_group].pop_max()
            self.ranking.append(selected_arm)
            j+=1

            # print(f"k:{j}, ranking:{self.ranking}")
            # print(f" s_group_ranking:{self.s_group_ranking}")


    def getDPR(self, g):
        # add unit test for EOR calculation
        s_group_ranking = self.s_group_ranking.copy()
        s_group_ranking[g] += 1/self.s_group[g]
        DPR = np.max(s_group_ranking, axis=0)-np.min(s_group_ranking, axis=0)
        return DPR.item()


