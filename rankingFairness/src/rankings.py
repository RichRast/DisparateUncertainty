import numpy as np
import random
from abc import ABC, abstractmethod
import math
import itertools
from scipy.optimize import linprog
import pdb
from rankingFairness.src.tradeoff import getProbsDict
from rankingFairness.src.utils import MaxPriorityQueue
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
        self.ranking = np.random.choice(np.arange(len(self.dist[0])+len(self.dist[1])),len(self.dist[0])+len(self.dist[1]), replace=False)

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
        return "EORI policy"
    
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
        self.groups=len(dist)
        self.num_ids=0
        for g in range(self.groups):
            self.num_ids += len(dist[g]) 
        self.ids = np.array([i for i in range(self.num_ids)])
        self.start_minority_idx = len(dist[0])
        probs_all, n_group = self.getExpectedRel(dist)
        self.n_group = n_group
        self.probs_all = probs_all
        majority_rel = np.array([d.getMean() for d in dist[0]])[:,None]
        minority_rel = np.array([d.getMean() for d in dist[1]])[:,None]
        self.switch_start=switch_start
        self.queue=[]
        for i in range(len(dist)):
            self.queue.append(MaxPriorityQueue())
            self.formQueue(i)
        self.arm_a=0.0
        self.arm_b=0.0

    @staticmethod
    def name():
        return "EOR"
    
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

    def rank(self, top_k=None) -> np.ndarray:
        EOR_monitor =[]
        self.ranking = []
        j = 0
        prev_a = self.queue[0].peek_max()[0]
        prev_b = self.queue[1].peek_max()[0]
        
        while j < top_k:
            EOR_compare=[]
            a,b=None, None
            assert len(self.queue[0]) + len(self.queue[1]) > 0 
            if len(self.queue[0]) > 0 : 
                a = self.queue[0].peek_max()[1]
                EOR_compare.append(self.getEOR(self.ranking + [a])[0])
            if len(self.queue[1]) > 0 : 
                b = self.start_minority_idx + self.queue[1].peek_max()[1]
                EOR_compare.append(self.getEOR(self.ranking + [b])[0])
            
            if (len(EOR_compare)==2):
                if (abs(EOR_compare[0]-EOR_compare[1])<=EPSILON):
                    selected_idx = np.random.choice(np.arange(len(EOR_compare)), 1).item()
                elif EOR_compare[0]<EOR_compare[1]:
                    selected_idx=0
                else:
                    selected_idx=1
                if selected_idx==0:
                    selected_arm = a
                    prev_a=self.probs_all[a]
                    self.queue[0].pop_max()

                else:
                    selected_arm = b
                    prev_b=self.probs_all[b]
                    self.queue[1].pop_max()

            elif (a is not None) and (len(EOR_compare)==1):
                selected_arm = a
                prev_a=self.probs_all[a]
                self.queue[0].pop_max()


            else:
                selected_arm = b
                prev_b=self.probs_all[b]
                self.queue[1].pop_max()


            self.arm_a=prev_a
            self.arm_b=prev_b
            self.ranking.append(selected_arm)
            j+=1


    def getEOR(self, ranking):
        # add unit test for EOR calculation
        ids_majority_ranking = np.intersect1d(ranking, self.ids[:self.start_minority_idx])
        ids_minority_ranking = np.intersect1d(ranking, self.ids[self.start_minority_idx:])
        if len(ids_minority_ranking)>0: 
            ids_minority_ranking -= self.start_minority_idx
        n_majority_ranking = sum([d.getMean() for d in np.array(self.dist[0])[ids_majority_ranking]]) if len(ids_majority_ranking)>0 else 0
        n_minority_ranking = sum([d.getMean() for d in np.array(self.dist[1])[ids_minority_ranking]]) if len(ids_minority_ranking)>0 else 0
        EOR = (n_majority_ranking/self.n_group[0]) - (n_minority_ranking/self.n_group[1])
        return np.abs(EOR), EOR

class DP_Ranker(Ranker):
    def __init__(self, dist, ucb=None, distType=None, switch_start=False) -> None:
        super().__init__(dist, ucb, distType)
        self.num_ids = len(dist[0]) + len(dist[1])
        self.ids = [i for i in range(self.num_ids)]
        self.start_minority_idx = len(dist[0])
        probs_all, n_group = self.getExpectedRel(dist)
        self.n_group = n_group
        self.queue=[]
        for i in range(len(dist)):
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
        for ids, val in zip(np.arange(len(pMeans)), pMeans):
            self.queue[g].add(val, ids)

    def getExpectedRel(self, dist):
        probs_all, n_group = getProbsDict(self.num_ids, dist)
        return probs_all, n_group

    def rank(self, top_k=None) -> np.ndarray:
        DPR_monitor =[]
        self.ranking = []
        j = 0
        
        while j < top_k:
            DPR_compare=[]
            a,b=None, None
            assert len(self.queue[0]) + len(self.queue[1]) > 0 
            if len(self.queue[0]) > 0 : 
                a = self.queue[0].peek_max()[1]
                DPR_compare.append(self.getDPR(self.ranking + [a])[0])
            if len(self.queue[1]) > 0 : 
                b = self.start_minority_idx + self.queue[1].peek_max()[1]
                DPR_compare.append(self.getDPR(self.ranking + [b])[0])
            
            if (len(DPR_compare)==2):
                if (abs(DPR_compare[0]-DPR_compare[1])<=EPSILON) or (self.switch_start and (self.getDPR(self.ranking)[0]<=EPSILON)):
                    selected_idx = np.random.choice(np.arange(len(DPR_compare)), 1).item()
                elif DPR_compare[0]<DPR_compare[1]:
                    selected_idx=0
                else:
                    selected_idx=1
                if selected_idx==0:
                    selected_arm = a
                    self.queue[0].pop_max()
                else:
                    selected_arm = b
                    self.queue[1].pop_max()
            elif (a is not None) and (len(DPR_compare)==1):
                selected_arm = a
                self.queue[0].pop_max()
            else:
                selected_arm = b
                self.queue[1].pop_max()

            self.ranking.append(selected_arm)
            j+=1

    def getDPR(self, ranking):
        # add unit test for EOR calculation
        ids_majority_ranking = np.intersect1d(ranking, self.ids[:self.start_minority_idx])
        ids_minority_ranking = np.intersect1d(ranking, self.ids[self.start_minority_idx:])
        if len(ids_minority_ranking)>0: 
            ids_minority_ranking -= self.start_minority_idx
        n_majority_ranking = len(ids_majority_ranking) if len(ids_majority_ranking)>0 else 0
        n_minority_ranking = len(ids_minority_ranking) if len(ids_minority_ranking)>0 else 0
        DPR = (n_majority_ranking/len(self.dist[0])) - (n_minority_ranking/len(self.dist[1]))
        return np.abs(DPR), DPR


class DP_RankerII(Ranker):
    def __init__(self, dist, ucb=None, distType=None, switch_start=False) -> None:
        super().__init__(dist, ucb, distType)
        self.num_ids = len(dist[0]) + len(dist[1])
        self.ids = [i for i in range(self.num_ids)]
        self.start_minority_idx = len(dist[0])
        probs_all, n_group = self.getExpectedRel(dist)
        self.n_group = n_group
        self.queue=[]
        for i in range(len(dist)):
            self.queue.append(MaxPriorityQueue())
            self.formQueue(i)

    @staticmethod
    def name():
        return "others"

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

    def rank(self, top_k=None) -> np.ndarray:
        DPR_monitor =[]
        self.ranking = []
        j = 0
        
        while j < top_k:
            DPR_compare=[]
            a,b=None, None
            assert len(self.queue[0]) + len(self.queue[1]) > 0 
            if len(self.queue[0]) > 0 : 
                a = self.queue[0].peek_max()[1]
                DPR_compare.append(self.getDPR(self.ranking + [a])[0])
            if len(self.queue[1]) > 0 : 
                b = self.start_minority_idx + self.queue[1].peek_max()[1]
                DPR_compare.append(self.getDPR(self.ranking + [b])[0])

            if len(DPR_compare)==2:
                if (abs(DPR_compare[0]-DPR_compare[1])<=EPSILON) or (self.switch_start and (self.getDPR(self.ranking)[0]<=EPSILON)):
                    selected_idx = np.random.choice(np.arange(len(DPR_compare)), 1).item()
                elif DPR_compare[0]<DPR_compare[1]:
                    selected_idx=0
                else:
                    selected_idx=1
                if selected_idx==0:
                    selected_arm = a
                    self.queue[0].pop_max()
                else:
                    selected_arm = b
                    self.queue[1].pop_max()
            elif (a is not None) and (len(DPR_compare)==1):
                selected_arm = a
                self.queue[0].pop_max()
            else:
                selected_arm = b
                self.queue[1].pop_max()

            self.ranking.append(selected_arm)
            j+=1

    def getDPR(self, ranking):
        # add unit test for EOR calculation
        ids_majority_ranking = np.intersect1d(ranking, self.ids[:self.start_minority_idx])
        ids_minority_ranking = np.intersect1d(ranking, self.ids[self.start_minority_idx:])
        if len(ids_minority_ranking)>0: 
            ids_minority_ranking -= self.start_minority_idx
        n_majority_ranking = sum([d.getMean() for d in np.array(self.dist[0])[ids_majority_ranking]]) if len(ids_majority_ranking)>0 else 0
        n_minority_ranking = sum([d.getMean() for d in np.array(self.dist[1])[ids_minority_ranking]]) if len(ids_minority_ranking)>0 else 0
        DPR = (n_majority_ranking/len(self.dist[0])) - (n_minority_ranking/len(self.dist[1]))
        return np.abs(DPR), DPR

class DP_RankerIII(Ranker):
    def __init__(self, dist, ucb=None, distType=None, switch_start=False) -> None:
        super().__init__(dist, ucb, distType)
        self.num_ids = len(dist[0]) + len(dist[1])
        self.ids = [i for i in range(self.num_ids)]
        self.start_minority_idx = len(dist[0])
        probs_all, n_group = self.getExpectedRel(dist)
        self.n_group = n_group
        self.queue=[]
        for i in range(len(dist)):
            self.queue.append(MaxPriorityQueue())
            self.formQueue(i)

    @staticmethod
    def name():
        return "RR"

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

    def rank(self, top_k=None) -> np.ndarray:
        self.ranking = []
        i,j=0,0
        while (i+j)<top_k:
            if (j/(i+j) if (i+j)!=0 else 0) < ((self.num_ids - self.start_minority_idx)/self.num_ids):
                selected_idx=1
                j +=1
            else:
                selected_idx=0
                i +=1
            selected_arm = self.queue[selected_idx].peek_max()[1]
            if selected_idx==1:
                selected_arm += self.start_minority_idx
            self.queue[selected_idx].pop_max()
            self.ranking.append(selected_arm)

class DP_RankerIV(Ranker):
    def __init__(self, dist, ucb=None, distType=None, switch_start=False) -> None:
        super().__init__(dist, ucb, distType)
        self.num_ids = len(dist[0]) + len(dist[1])
        self.ids = [i for i in range(self.num_ids)]
        self.start_minority_idx = len(dist[0])
        probs_all, n_group = self.getExpectedRel(dist)
        self.n_group = n_group
        self.queue=[]
        for i in range(len(dist)):
            self.queue.append(MaxPriorityQueue())
            self.formQueue(i)

    @staticmethod
    def name():
        return "DC"

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

    def rank(self, top_k=None) -> np.ndarray:
        self.ranking = []
        i,j=0,0
        
        while (i+j)<top_k:
            val_a, val_b = -np.inf,-np.inf
            if len(self.queue[0])>0:
                val_a=self.queue[0].peek_max()[0]
            if len(self.queue[1])>0:
                val_b=self.queue[1].peek_max()[0]
            assert len(self.queue[0])+len(self.queue[1])>0
            if ((j/(i+j) if (i+j)!=0 else 0) < ((self.num_ids - self.start_minority_idx)/self.num_ids)) or (val_b>val_a):
                selected_idx=1
                j +=1
            else:
                selected_idx=0
                i +=1
            selected_arm = self.queue[selected_idx].peek_max()[1]
            if selected_idx==1:
                selected_arm += self.start_minority_idx
            self.queue[selected_idx].pop_max()
            self.ranking.append(selected_arm)

