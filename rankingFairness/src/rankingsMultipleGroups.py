import numpy as np
import random
from abc import ABC, abstractmethod
import math
import itertools
from rankingFairness.src.utils import MaxPriorityQueue, getProbsDict
from rankingFairness.src.decorators import timer
from rankingFairness.src.Baselines.rankAggregationBaselines import epiRA, calc_exposure_ratio
from rankingFairness.src.Baselines.exposureBaselines import getExposureMetrics
from rankingFairness.src.Baselines.fairsearch import getFSMetrics
from tqdm import tqdm

EPSILON=1e-12
SEED=42

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
        self.groups = len(dist)
        self.num_ids = 0
        for g in range(self.groups):
            self.num_ids += len(dist[g]) 
        self.ids = np.array([i for i in range(self.num_ids)])
        self.probs_all, self.n_group = getProbsDict(self.num_ids, dist)
        
        self.switch_start = switch_start

        self.group_ids = []
        for g, d in enumerate(dist):
            for _ in range(len(d)):
                self.group_ids.append(g)

    @staticmethod
    def name():
        return 'Uniform'
    
    def rank_sim(self, top_k, simulations) -> np.ndarray:
        np.random.seed(SEED)
        random.seed(SEED)
        self.ranking = np.zeros((simulations, top_k), dtype=int)
        for e in tqdm(range(simulations)):
            self.ranking[e,:] = np.random.choice(np.arange(top_k), top_k, replace=False)

    def rank(self, top_k, simulations) -> np.ndarray:
        np.random.seed(SEED)
        random.seed(SEED)
        abs_EOR_summary = np.full((simulations,), np.inf)
        ranking_tmp = np.zeros((simulations, top_k), dtype=int)
        for e in tqdm(range(simulations)):
            ranking_tmp[e,:] = np.random.choice(np.arange(top_k), top_k, replace=False)
            abs_EOR_summary[e:,]= self.getEORSummary(ranking_tmp[e,:])
        idx = (np.abs(abs_EOR_summary - np.median(abs_EOR_summary))).argmin()
        # idx = np.argwhere(np.isclose(abs_EOR_summary, np.median(abs_EOR_summary), rtol=EPSILON, atol=EPSILON))[0].item()
        self.ranking = ranking_tmp[idx,:]
    
    def getEORSummary(self, ranking):
        n_group_ranking = np.full((self.groups,ranking.shape[0]), 0.0, dtype=float) # shape is number of groups by rank (col 0 is rank 1)
        for k, r in enumerate(ranking):
            if k >0:
                n_group_ranking[:,k] = n_group_ranking[:,k-1]
            n_group_ranking[self.group_ids[r],k] += self.probs_all[r]/self.n_group[self.group_ids[r]]
        EOR = np.max(n_group_ranking, axis=0)-np.min(n_group_ranking, axis=0)
        return EOR.sum()


class TS_RankerII(Ranker):
    def __init__(self, dist, ucb=None , distType=None, switch_start=False) -> None:
        super().__init__(dist, ucb, distType)
        """
        This class takes the drawn merits and randomizes for all the students with
        merit=1
        """
        self.groups = len(dist)
        self.num_ids = 0
        for g in range(self.groups):
            self.num_ids += len(dist[g]) 
        self.ids = np.array([i for i in range(self.num_ids)])
        self.probs_all, self.n_group = getProbsDict(self.num_ids, dist)
        
        self.switch_start = switch_start

        self.group_ids = []
        for g, d in enumerate(dist):
            for _ in range(len(d)):
                self.group_ids.append(g)

    @staticmethod
    def name():
        return 'TS'
    
    def sampleMerits(self):
        meritObj = self.distType(self.dist)
        return meritObj.sample()
    
    # def rank(self, top_k, simulations) -> np.ndarray:
    #     ranking_tmp = np.zeros((simulations, top_k))
    #     merits = np.full((simulations, top_k), np.inf)
    #     for e in tqdm(range(simulations)):
    #         merits[e,:] = self.sampleMerits()
    #     b=np.random.random((simulations, top_k))
    #     ranking_tmp = np.lexsort((b,merits), axis=1)[:,::-1]
    #     abs_EOR_summary = self.getEORSummary(ranking_tmp, simulations)
    #     idx = np.argwhere(np.isclose(abs_EOR_summary, np.median(abs_EOR_summary), rtol=EPSILON, atol=EPSILON))[0].item()
    #     self.ranking = ranking_tmp[idx,:]
    
    def rank_sim(self, top_k, simulations) -> np.ndarray:
        merits = np.full((simulations, top_k), np.inf)
        for e in tqdm(range(simulations)):
            merits[e,:] = self.sampleMerits()
        b=np.random.random((simulations, top_k))
        self.ranking = np.lexsort((b,merits), axis=1)[:,::-1]
        

    def rank(self, top_k, simulations) -> np.ndarray:
        """
        Ashudeep et al. Fairness in Ranking under Uncertainty
        """
        np.random.seed(SEED)
        random.seed(SEED)
        abs_EOR_summary=np.full((simulations,), np.inf)
        ranking_tmp = np.zeros((simulations, top_k), dtype=int)
        merits = np.full((simulations, top_k), np.inf)
        for e in tqdm(range(simulations)):
            merits[e,:] = self.sampleMerits()
            merits[e,:] += np.random.random(top_k) * EPSILON
            ranking_tmp[e,:] = np.argsort(-merits[e,:])
            abs_EOR_summary[e] = self.getEORSummary(ranking_tmp[e,:])
        idx = (np.abs(abs_EOR_summary - np.median(abs_EOR_summary))).argmin()
        self.ranking = ranking_tmp[idx,:]
    
    def getEORSummary(self, ranking):
        n_group_ranking = np.full((self.groups,ranking.shape[0]), 0.0, dtype=float) 
        for k, r in enumerate(ranking):
            if k >0:
                n_group_ranking[:,k] = n_group_ranking[:,k-1]
            n_group_ranking[self.group_ids[r],k] += self.probs_all[r]/self.n_group[self.group_ids[r]]
        EOR = np.max(n_group_ranking, axis=0)-np.min(n_group_ranking, axis=0)
        return EOR.sum()
        
class EO_Ranker(Ranker):
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

class DP_RankerIII(Ranker):
    def __init__(self, dist, ucb=None, distType=None, switch_start=False) -> None:
        super().__init__(dist, ucb, distType)
        assert len(dist)==2, f" provide 2 groups "
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
        return "PRR"

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


class epiRAnker(Ranker):
    def __init__(self, dist, ucb=None, distType=None, switch_start=False, verbose=False) -> None:
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
        
        self.queue=MaxPriorityQueue()
        self.formQueue()
        self.lmbd=0.95
        self.verbose=verbose
        if self.verbose:
            print(f" with exposure threshold as :{self.lmbd}")

    @staticmethod
    def name():
        return "RA"

    def formQueue(self):
        pMeans_ls=[]
        for i in range(len(self.dist)):
            pMeans_ls.append([p.getMean() for p in self.dist[i]])
        pMeans = np.concatenate((pMeans_ls))
        for ids, val in zip(np.arange(len(pMeans)), pMeans):
            self.queue.add(val, ids)


    def rank(self, top_k=None) -> np.ndarray:
        PRP_ranking = np.array([self.queue.pop_max()[1] for _ in range(top_k)])
        PRP_group_ids=np.array([self.group_ids[i] for i in PRP_ranking])
        current_ranking, current_group_ids, cur_exp = epiRA(None, item_ids=self.ids, group_ids=self.group_ids, bnd=self.lmbd,
         grporder=True, current_ranking=PRP_ranking, current_group_ids=PRP_group_ids, verbose=self.verbose)
        self.ranking = list(current_ranking)
        self.exp_achieved = cur_exp

class exposure(Ranker):
    def __init__(self, dist, ucb=None, distType=None, switch_start=False, verbose=True) -> None:
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
        
        self.exp_thresh=1.0
        self.verbose=verbose

    @staticmethod
    def name():
        return "EXP"
    
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

    def rank(self, top_k=None, merits=None) -> np.ndarray:
        rel=np.array(list(self.probs_all.values()))[:,None]
        if merits is not None:
            merits = merits.T
        EOR, total_cost, group_cost, DCG, EOR_abs = getExposureMetrics(rel, np.array(self.group_ids), self.n_group, merits, verbose=self.verbose)
        return EOR, total_cost, group_cost, DCG, EOR_abs
        
class exposure_DP(Ranker):
    def __init__(self, dist, ucb=None, distType=None, switch_start=False, verbose=True) -> None:
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
        
        self.exp_thresh=1.0
        self.verbose=verbose

    @staticmethod
    def name():
        return "DPE"
    
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

    def rank(self, top_k=None, merits=None) -> np.ndarray:
        rel=np.array(list(self.probs_all.values()))[:,None]
        if merits is not None:
            merits = merits.T
        EOR, total_cost, group_cost, DCG, EOR_abs = getExposureMetrics(rel, np.array(self.group_ids), self.n_group, merits=merits, dp=True, verbose=self.verbose)
        return EOR, total_cost, group_cost, DCG, EOR_abs

class fairSearch(Ranker):
    def __init__(self, dist, ucb=None, distType=None, alpha=0.1, p=0.5, switch_start=False) -> None:
        super().__init__(dist, ucb, distType)
        self.groups = len(dist)
        self.num_ids = 0
        self.alpha=alpha
        self.p = p
        for g in range(self.groups):
            self.num_ids += len(dist[g]) 
        self.ids = np.array([i for i in range(self.num_ids)])

        self.group_ids = []
        self.s_group=np.full(self.groups,0)
        for g, d in enumerate(dist):
            for _ in range(len(d)):
                self.group_ids.append(g)
            self.s_group[g]=len(d)
        
        self.queue=MaxPriorityQueue()
        self.formQueue()

    @staticmethod
    def name():
        return "FS"

    def formQueue(self):
        pMeans_ls=[]
        for i in range(len(self.dist)):
            pMeans_ls.append([p.getMean() for p in self.dist[i]])
        pMeans = np.concatenate((pMeans_ls))
        for ids, val in zip(np.arange(len(pMeans)), pMeans):
            self.queue.add(val, ids)


    def rank(self, top_k=None) -> np.ndarray:
        PRP_ranking = np.full((top_k,),0, dtype=int)
        PRP_rel = np.full((top_k,),0.0)
        for i in range(top_k):
            rel, rel_id = self.queue.pop_max()
            PRP_ranking[i]=rel_id
            PRP_rel[i]=rel
        # PRP_ranking = np.array([self.queue.pop_max()[1] for _ in range(top_k)])
        PRP_group_ids=np.array([self.group_ids[i] for i in PRP_ranking])
        # PRP_rel = np.array([self.queue.pop_max()[0] for _ in range(top_k)])
        current_ranking = getFSMetrics(PRP_rel=PRP_rel, PRP_group_ids=PRP_group_ids, PRP_ranking=PRP_ranking, alpha=self.alpha, p=self.p)
        self.ranking = list(current_ranking)