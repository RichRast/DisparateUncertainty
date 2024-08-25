import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from rankingFairness.src.distributions import Bernoulli
from rankingFairness.src.tradeoffMultipleGroups import UtilityCost
from rankingFairness.src.utils import getGroupNames
from rankingFairness.src.decorators import timer
from rankingFairness.src.rankingsMultipleGroups import Uniform_Ranker, TS_RankerII, EO_RankerII, DP_Ranker, parallelRanker, epiRAnker, exposure, exposure_DP, fairSearch

import random
from copy import deepcopy
from matplotlib.lines import Line2D
from scipy.stats import beta as beta_dist
import seaborn as sns
from tqdm import tqdm

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'svg')
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
import matplotlib
matplotlib.style.use('classic')
import os
import os.path as osp

MARKERS = ['s', 'o', 'X', 'P', 'D']
COLORMAP = {'PRP':'tab:blue', 
                        'TS':'tab:orange', 
                        'DP':'tab:grey',
                        'Uniform':'tab:green', 
                        'EOR':'tab:red',
                        'PRR':'tab:olive',
                        'RA':'tab:cyan',
                        'EXP':'tab:purple',
                        'DPE': 'tab:pink',
                        'FS': 'tab:brown'}

class GeneralExperiment(ABC):
    def __init__(self, num_groups) -> None:
        super().__init__()
        self.num_groups=num_groups

    @abstractmethod
    def setGroups(self): 
        pass


    @abstractmethod
    def visualize_EO(self):
        pass


class simpleOfflineMultipleGroups(GeneralExperiment):
    def __init__(self, num_groups, num_docs, predfined_ls, distType, online=False, plot_median=True, 
    merits=None, plot=True, saveFig=None, offset=0.05, verbose=False) -> None:
        super().__init__(num_groups)
        self.num_docs = num_docs
        self.online=online
        self.experimentSetup='Offline'
        self.predfined_ls=predfined_ls
        self.groups=len(self.predfined_ls)
        self.colorMap=COLORMAP
        self.markers = MARKERS*self.groups
        self.lineMap=["dashed", "dotted"]*self.groups
        self.distType = distType
        self.delta_max=None
        self.merits=merits
        self.n_labels=False
        if self.merits is not None:
            self.n_labels=True
        self.plot=plot
        self.offset=offset
        self.verbose=verbose
        self.saveFig=saveFig
        if (self.saveFig is not None) and (not osp.exists(self.saveFig)):
                os.makedirs(self.saveFig)
        self.plot_median=plot_median

    def setGroups(self):
        self.majority_ids=np.arange(self.start_minority_idx)
        self.minority_ids=np.arange(self.start_minority_idx, self.num_docs)
        
    def getGroupNames(self, ranking):
        groupNames=[0 if r<self.start_minority_idx else 1 for r in ranking]
        return groupNames

    def sampleMerits(self):
        meritObj = self.distType(self.predfined_ls)
        return meritObj.sample()


    def posteriorPredictiveAlg(self, simulations, rankingAlgos):

        a_EOR=None
        rankingAlgoInst=[]
        for a, rankingAlg in enumerate(rankingAlgos):
            ranker = rankingAlg(self.predfined_ls, distType=self.distType)
            rankingAlgoInst.append(ranker)
            if isinstance(ranker, Uniform_Ranker) or isinstance(ranker, TS_RankerII):
                if self.plot_median:
                    ranker.rank(self.num_docs, simulations)
                    self.ranking[a]=np.array(ranker.ranking)[None,:]
                else:
                    ranker.rank_sim(self.num_docs, simulations)
                    self.ranking[a]=ranker.ranking
            elif isinstance(ranker, exposure) or isinstance(ranker, exposure_DP):
                pass
            else:
                ranker.rank(self.num_docs)
                self.ranking[a]=np.array(ranker.ranking)[None,:]
                if isinstance(ranker, EO_RankerII):
                    a_EOR=a
                elif isinstance(ranker, epiRAnker):
                    self.RA_exp_achieved=ranker.exp_achieved
                    
        for top_k in range(self.num_docs):    
            for a, rankingAlg in enumerate(rankingAlgoInst):
                if isinstance(rankingAlg, exposure) or isinstance(rankingAlg, exposure_DP):
                    pass
                else:
                    utilCostObj= UtilityCost(self.ranking[a], self.num_docs, top_k+1, simulations, self.groups, n_labels=self.n_labels)
                    self.dcgUtil[a, top_k] = utilCostObj.getUtil(self.predfined_ls, self.merits)
                    utilCostObj.getCostArms(self.predfined_ls, self.merits)
                    self.cost_groups[a, :, top_k] =utilCostObj.cost_groups
                    self.total_cost[a, top_k]=utilCostObj.cost
                    EOR_obj=utilCostObj.EOR_constraint(self.predfined_ls, self.merits)
                    self.EO_constraint[a, top_k]=EOR_obj[0]
                    self.EOR_abs_avg[a, top_k]=EOR_obj[2]
                    if self.verbose:
                        self.total_cost_std[a, top_k]=utilCostObj.total_cost_std
                        self.group_cost_std[a, :, top_k]=utilCostObj.group_cost_std
        
                    self.delta_max=EOR_obj[1]
        for a, rankingAlg in enumerate(rankingAlgoInst):
            if isinstance(rankingAlg, exposure):
                ranker_exp = exposure(self.predfined_ls, distType=self.distType, verbose=self.verbose)
                self.EO_constraint[a, :], self.total_cost[a, :], self.cost_groups[a, :, :], self.dcgUtil[a, :], self.EOR_abs_avg[a, :] = ranker_exp.rank(self.num_docs, merits=self.merits)
            if isinstance(rankingAlg, exposure_DP):
                ranker_exp_dp = exposure_DP(self.predfined_ls, distType=self.distType, verbose=self.verbose)
                self.EO_constraint[a, :], self.total_cost[a, :], self.cost_groups[a, :, :], self.dcgUtil[a, :], self.EOR_abs_avg[a, :] = ranker_exp_dp.rank(self.num_docs, merits=self.merits)
        
        if a_EOR is not None and (self.merits is None):
            assert np.all(self.EO_constraint[a_EOR, :]<=self.delta_max), f"{np.max(self.EO_constraint[a_EOR, :])}, delta_max: {self.delta_max}"
                            
            
        if self.plot:
            self.visualize_Cost(rankingAlgos)
            self.visualize_EO(rankingAlgos)
            self.visualize_Util(rankingAlgos)


    def visualize_Util(self, rankingAlgos):
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(5,5))
        for a, rankingAlg in enumerate(rankingAlgos):
            ax.plot(np.arange(self.num_docs)+1, self.dcgUtil[a, :], label=f"{rankingAlg.name()}", c=self.colorMap[rankingAlg.name()], marker = '.',linewidth=1, markersize=1)


        plt.xlabel("Length of Ranking (k)", fontsize=20)
        plt.ylabel(r'DCG $U[\pi_k]$', fontsize=20)

        plt.ylim(-self.offset, np.max(self.dcgUtil) + self.offset)
        plt.xlim(1-self.offset, self.num_docs+self.offset)
        plt.tight_layout() 
        if self.saveFig is not None:
            plt.savefig(f"{osp.join(self.saveFig,'DCG_Util.pdf')}")
        plt.show()
        plt.close()
    
    def visualize_Cost(self, rankingAlgos):
        fig, ax = plt.subplots(figsize=(10,5), ncols=2)
        plt.rc('font', family='serif')
        start_index=int(self.num_docs/3)
        end_index=int(self.num_docs/3)
        for a, rankingAlg in enumerate(rankingAlgos):
            for g in range(self.groups):
                ax[0].plot(np.arange(self.num_docs)+1, self.cost_groups[a, g, :], linestyle=self.lineMap[g], c=self.colorMap[rankingAlg.name()], linewidth=3)
            ax[1].plot(np.arange(self.num_docs)+1, self.total_cost[a, :], linestyle='solid', c=self.colorMap[rankingAlg.name()], linewidth=1)    
            
        handles, labels = plt.gca().get_legend_handles_labels()

        
        # fig.supxlabel("Length of Ranking (k)", fontsize=20)
        # fig.supylabel(f"Costs ", fontsize=20)
        for axis in ax.ravel():
            axis.set_ylim(-self.offset, 1 + self.offset)
            axis.set_xlim(1-self.offset, self.num_docs+self.offset)
        
        fig.tight_layout() 
        if self.saveFig is not None:
            plt.savefig(f"{osp.join(self.saveFig,'Cost.pdf')}")
        plt.show()
        plt.close()

    def visualize_EO(self, rankingAlgos):
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(6,6))
        for a, rankingAlg in enumerate(rankingAlgos):
            ax.plot(np.arange(self.num_docs)+1, self.EO_constraint[a, :], label=str(rankingAlg.name()), c=self.colorMap[rankingAlg.name()])
        
            
        if len(self.predfined_ls)==2:
            plt.ylabel(r'$\bf{\delta(\sigma_k) = \frac{n(A|\sigma_k) }{n(A)}- \frac{n(B|\sigma_k)}{n(B)}}$', fontsize=20)
        else:
            plt.ylabel(r'$\bf{\delta(\sigma_k) = \max_{g} \left(\frac{n(g|\sigma_k) }{n(g)}\right)- \min_{g} \left(\frac{n(g|\sigma_k)}{n(g)}\right)}$', fontsize=20)
        plt.xlabel("Length of Ranking (k)", fontsize=20)
        
        if self.delta_max is not None:
            plt.hlines(y=self.delta_max, xmin=1, xmax=self.num_docs, color='black', linestyle='dashed')
            plt.hlines(y=-self.delta_max, xmin=1, xmax=self.num_docs, color='black', linestyle='dashed')
            plt.text(self.num_docs/2, self.delta_max+self.offset, r'$\delta_{max}$', c='black', fontsize=20)
            plt.text(self.num_docs/2, -self.delta_max-self.offset, r'$-\delta_{max}$', c='black', fontsize=20)
        
        n_majority=sum([p.getMean() for p in self.predfined_ls[0]])
        n_minority=sum([p.getMean() for p in self.predfined_ls[1]])
        
        handles, labels = ax.get_legend_handles_labels()
        majority_lines= Line2D([], [], color='black', linestyle='dashed', linewidth=3, label=r'Majority (group A) Cost')
        minority_lines= Line2D([], [], color='black', linestyle='dotted',  linewidth=3, label=r'Minority (group B) Cost')
        total_lines = Line2D([], [], color='black', linestyle='solid',  linewidth=3, label=r'Total Cost')
        handles.extend([majority_lines, minority_lines, total_lines])

        plt.xlim(1-self.offset, self.num_docs+self.offset)
        plt.tight_layout()
        if self.saveFig is not None:
            plt.savefig(f"{osp.join(self.saveFig,'EO.pdf')}", bbox_inches='tight')
            
        plt.show()
        plt.close()

    def visualize_ranking(self, ranking, true_thetas, rankingAlgoName ):
        gName_ranking=getGroupNames(self.start_minority_idx, ranking)
        color_map={0:'r', 1:'b'}
        k=self.num_docs
        plt.scatter(np.arange(len(gName_ranking[:k])), [true_thetas[i] for i in ranking[:k]], 
                    c=[color_map[i] for i in gName_ranking[:k]], s=5)

        plt.title(f"{rankingAlgoName} ranking of items from two groups", fontsize=15)
        plt.scatter([], [], c='r', label="A", s=5)
        plt.scatter([], [], c='b', label="B", s=5)
        plt.ylabel(r'$\theta^*$', fontsize=15)
        plt.xlabel(f"top k", fontsize=15)
        plt.legend(fontsize=15)
        plt.show()
        plt.close()

    def experiment(self, rankingAlgos, simulations, figsize=(20,5)):
        """
        Args:
            timesteps: (int) how many steps for the algo to learn the bandit
            simulations: (int) number of epochs
        """
        names=[]
        self.dcgUtil = np.zeros((len(rankingAlgos), self.num_docs))
        self.cost_groups = np.zeros((len(rankingAlgos), self.groups, self.num_docs))
        self.total_cost = np.zeros((len(rankingAlgos), self.num_docs))
        self.EO_constraint = np.zeros((len(rankingAlgos), self.num_docs))
        
        self.ranking={}
        self.EOR_abs_avg=np.zeros((len(rankingAlgos), self.num_docs))
        if self.verbose:
            self.total_cost_std=np.zeros((len(rankingAlgos), self.num_docs))
            self.group_cost_std=np.zeros((len(rankingAlgos), self.groups, self.num_docs))

        plt.rcParams["figure.figsize"] = figsize
        self.posteriorPredictiveAlg(simulations, rankingAlgos)


    
