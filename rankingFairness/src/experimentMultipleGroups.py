import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from rankingFairness.src.distributions import Bandit, Bernoulli
from rankingFairness.src.tradeoffMultipleGroups import UtilityCost
from rankingFairness.src.utils import getGroupNames
from rankingFairness.src.decorators import timer
from rankingFairness.src.rankingsMultipleGroups import Uniform_Ranker, TS_RankerII, EO_RankerII, DP_Ranker
import pdb
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
    def __init__(self, num_groups, num_docs, predfined_ls, distType, online=False, 
    merits=None, plot=True, saveFig=None, offset=0.05, verbose=False, switch_start=False) -> None:
        super().__init__(num_groups)
        self.num_docs = num_docs
        self.online=online
        # self.setGroups()
        self.predfined_ls=predfined_ls
        self.groups=len(self.predfined_ls)
        self.colorMap={'PRP':'tab:blue', 
                        'TS':'tab:orange', 
                        'DP':'tab:grey',
                        'Uniform':'tab:green', 
                        'EOR':'tab:red',
                        'others':'tab:brown',
                        'RR':'tab:olive',
                        'DC':'tab:pink'}
        self.markers=['s', 'X', 'o', 'P']*self.groups
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
        self.switch_start=switch_start

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
        for a, rankingAlg in enumerate(rankingAlgos):
            ranker = rankingAlg(self.predfined_ls, distType=self.distType)
        
            if isinstance(ranker, Uniform_Ranker):
                ranking_simulations = np.ones((simulations, self.num_docs), dtype=int)
                for e in tqdm(range(simulations)):
                    ranker.rank(self.num_docs)
                    ranking_simulations[e]=ranker.ranking
                self.ranking[a]=ranking_simulations
            elif isinstance(ranker, TS_RankerII):
                ranker.rank(self.num_docs, simulations)
                self.ranking[a]=ranker.ranking
            else:
                ranker.rank(self.num_docs)
                self.ranking[a]=np.array(ranker.ranking)[None,:]
                if isinstance(ranker, EO_RankerII):
                    a_EOR=a
                    
        for top_k in range(self.num_docs):    
            for a, rankingAlg in enumerate(rankingAlgos):
                utilCostObj= UtilityCost(self.ranking[a], self.num_docs, top_k+1, simulations, self.groups, n_labels=self.n_labels)
                self.dcgUtil[a, top_k] = utilCostObj.getUtil(self.predfined_ls, self.merits)
                utilCostObj.getCostArms(self.predfined_ls, self.merits)
                self.cost_groups[a, :, top_k] =utilCostObj.cost_groups
                self.total_cost[a, top_k]=utilCostObj.cost
                EOR_obj=utilCostObj.EOR_constraint(self.predfined_ls, self.merits)
                self.EO_constraint[a, top_k]=EOR_obj[0]
                if self.verbose:
                    self.EOR_std[a, top_k]=EOR_obj[2]
                    self.total_cost_std[a, top_k]=utilCostObj.total_cost_std
                    self.group_cost_std[a, :, top_k]=utilCostObj.group_cost_std
        
        self.delta_max=EOR_obj[1]
        if a_EOR is not None and (self.merits is None):
            assert np.all(self.EO_constraint[a_EOR, :]<=self.delta_max), f"{np.max(self.EO_constraint[a_EOR, :])}, delta_max: {self.delta_max}"
                            
        # if (a_EOR is not None) and (self.merits is None):
        #     assert np.allclose(np.abs(self.EO_constraint[a_EOR, :])<=self.delta_max, np.full((self.num_docs), True, dtype=bool)), f"{self.EO_constraint[a_EOR, self.EO_constraint[a_EOR,:]>self.delta_max]}, {self.delta_max[self.EO_constraint[a_EOR,:]>self.delta_max]}, k:{np.where(self.EO_constraint[a_EOR,:]>self.delta_max)[0]}"
            
        if self.plot:
            self.visualize_Cost(rankingAlgos)
            self.visualize_EO(rankingAlgos)
            self.visualize_Util(rankingAlgos)


    def visualize_Util(self, rankingAlgos):
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(5,5))
        for a, rankingAlg in enumerate(rankingAlgos):
            ax.plot(np.arange(self.num_docs)+1, self.dcgUtil[a, :], label=f"{rankingAlg.name()}", c=self.colorMap[rankingAlg.name()], marker = '.',linewidth=1, markersize=1)

        # plt.grid()   
        plt.xlabel("Length of Ranking (k)", fontsize=20)
        plt.ylabel(r'DCG $U[\pi_k]$', fontsize=20)
        # plt.legend(fontsize=12, loc='upper center', 
        # bbox_to_anchor=(0.5, 1.35),
        # ncol=3)
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
                # ax[0].plot(np.arange(self.num_docs)+1, self.cost_groups[a, g, :], linestyle=self.lineMap[g], marker=self.markers[g], markerfacecolor='w', markersize=12, markevery=(start_index, end_index), c=self.colorMap[rankingAlg.name()], markeredgecolor=self.colorMap[rankingAlg.name()], linewidth=3)
                ax[0].plot(np.arange(self.num_docs)+1, self.cost_groups[a, g, :], linestyle=self.lineMap[g], c=self.colorMap[rankingAlg.name()], linewidth=3)
            ax[1].plot(np.arange(self.num_docs)+1, self.total_cost[a, :], linestyle='solid', c=self.colorMap[rankingAlg.name()], linewidth=1)    
            
        handles, labels = plt.gca().get_legend_handles_labels()
        # for g in range(self.groups):
        #     group_lines= Line2D([], [], color='black', linestyle=self.lineMap[g], label=f'Group {g} Cost')
        #     handles.extend([group_lines])
        
        
        fig.supxlabel("Length of Ranking (k)", fontsize=20)
        fig.supylabel(f"Costs ", fontsize=20)
        # pos = ax.get_position()
        # ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.7])
        # plt.legend(handles=handles, fontsize=12, loc='upper center', 
        # bbox_to_anchor=(0.5, 1.35),
        # ncol=3)
        for axis in ax.ravel():
            axis.set_ylim(-self.offset, 1 + self.offset)
            axis.set_xlim(1-self.offset, self.num_docs+self.offset)
        # plt.grid()
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
        # if self.verbose:
        #     relevance_handle = ax.plot([], [], ' ', label=f'$n_A={{{n_majority}}}, n_B={{{n_minority}}}$')
        #     size_handle = ax.plot([], [], ' ', label=f'$\vert A \vert={{{self.start_minority_idx}}}, \vert B \vert={{{self.num_docs-self.start_minority_idx}}}$')
        handles, labels = ax.get_legend_handles_labels()
        majority_lines= Line2D([], [], color='black', linestyle='dashed', linewidth=3, label=r'Majority (group A) Cost')
        minority_lines= Line2D([], [], color='black', linestyle='dotted',  linewidth=3, label=r'Minority (group B) Cost')
        total_lines = Line2D([], [], color='black', linestyle='solid',  linewidth=3, label=r'Total Cost')
        handles.extend([majority_lines, minority_lines, total_lines])
        # legend=plt.legend(handles=handles, fontsize=12, loc='upper center', 
        # bbox_to_anchor=(0.5, 1.35),
        # ncol=8)
        plt.xlim(1-self.offset, self.num_docs+self.offset)
        plt.tight_layout()
        if self.saveFig is not None:
            plt.savefig(f"{osp.join(self.saveFig,'EO.pdf')}", bbox_inches='tight')
            # bbox=legend.get_window_extent()
            # expand=[-5,-5,5,5]
            # bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
            # bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
            # plt.savefig(f"{osp.join(self.saveFig,'legend.pdf')}", bbox_inches=bbox)
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
        if self.verbose:
            self.EOR_std=np.zeros((len(rankingAlgos), self.num_docs))
            self.total_cost_std=np.zeros((len(rankingAlgos), self.num_docs))
            self.group_cost_std=np.zeros((len(rankingAlgos), self.groups, self.num_docs))

        plt.rcParams["figure.figsize"] = figsize
        self.posteriorPredictiveAlg(simulations, rankingAlgos)

