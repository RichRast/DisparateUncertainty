import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from rankingFairness.src.distributions import Bernoulli
from rankingFairness.src.tradeoffMultipleGroups import UtilityCost
from rankingFairness.src.utils import getGroupNames
from rankingFairness.src.decorators import timer
from rankingFairness.src.rankingsMultipleGroups import Uniform_Ranker, TS_RankerII, EO_RankerII, DP_Ranker, parallelRanker, epiRAnker, exposure, exposure_DP, fairSearch

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
                    if self.verbose:
                        self.EOR_abs_avg[a, top_k]=EOR_obj[2]
                        self.total_cost_std[a, top_k]=utilCostObj.total_cost_std
                        self.group_cost_std[a, :, top_k]=utilCostObj.group_cost_std
        
                    self.delta_max=EOR_obj[1]
        for a, rankingAlg in enumerate(rankingAlgoInst):
            if isinstance(rankingAlg, exposure):
                ranker_exp = exposure(self.predfined_ls, distType=self.distType)
                self.EO_constraint[a, :], self.total_cost[a, :], self.cost_groups[a, :, :], self.dcgUtil[a, :], self.EOR_abs_avg[a, :] = ranker_exp.rank(self.num_docs, merits=self.merits)
            if isinstance(rankingAlg, exposure_DP):
                ranker_exp_dp = exposure_DP(self.predfined_ls, distType=self.distType)
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
        if self.verbose:
            self.EOR_abs_avg=np.zeros((len(rankingAlgos), self.num_docs))
            self.total_cost_std=np.zeros((len(rankingAlgos), self.num_docs))
            self.group_cost_std=np.zeros((len(rankingAlgos), self.groups, self.num_docs))

        plt.rcParams["figure.figsize"] = figsize
        self.posteriorPredictiveAlg(simulations, rankingAlgos)

class BayesianSetup(GeneralExperiment):
    def __init__(self, num_groups, num_docs, groupLen=None, predfined_ls=None, distType=None, online=False, 
    merits=None, plot=True, saveFig=None, offset=0.05, verbose=False) -> None:
        super().__init__(num_groups)
        self.num_docs = num_docs
        self.online=online
        self.experimentSetup='Bayesian'
        self.predfined_ls=predfined_ls
        self.groupLen=groupLen
        if self.predfined_ls is not None:
            self.groups=len(self.predfined_ls)
        else:
            self.groups=len(self.groupLen)
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


    def setGroups(self):
        ...
        
    def getGroupNames(self, ranking):
        groupNames=[0 if r<self.start_minority_idx else 1 for r in ranking]
        return groupNames

    def sampleMerits(self):
        ...

    def updateDist(self):
        ...

    def get_regret(self):
        ...

    def get_thetaDiff(self):
        ...
    def createAssymety(self, algo, majority_rounds=50, minority_rounds=0):
        for _ in range(majority_rounds):
            merits_drawn = algo.get_merits()
            algo.update_params(np.repeat(np.arange(self.groupLen[0])[:,None], self.draws, axis=1), merits_drawn )
        for _ in range(minority_rounds):
            merits_drawn = algo.get_merits()
            algo.update_params(np.repeat(np.arange(self.groupLen[0], self.groupLen[0]+self.groupLen[1])[:,None], self.draws, axis=1),
            merits_drawn )
    
    def getGroupDist(self, dist):
        merit_distributions=[]
        for d in range(self.draws):
            merit_distributions.append([dist[:self.groupLen[0],d], dist[self.groupLen[0]: self.groupLen[0]+self.groupLen[1],d]])

        return merit_distributions

    def computeCosts(self, timestep):
        for top_k in range(self.num_docs): 
            for a, rankingAlg in enumerate(self.rankingAlgos):
                for d in range(self.draws):
                    utilCostObj= UtilityCost(self.ranking[a][timestep,top_k,...,d], self.num_docs, top_k+1, self.simulations, self.groups, n_labels=self.n_labels)
                    self.dcgUtil[a, top_k, d] = utilCostObj.getUtil(self.true_dist[d], self.merits)
                    utilCostObj.getCostArms(self.true_dist[d], self.merits)
                    self.cost_groups[a, :, top_k, d] =utilCostObj.cost_groups
                    self.total_cost[a, top_k, d]=utilCostObj.cost
                    EOR_obj=utilCostObj.EOR_constraint(self.true_dist[d], self.merits)
                    self.EO_constraint[a, top_k, d]=EOR_obj[0]
                    if self.verbose:
                        self.EOR_std[a, top_k, d]=EOR_obj[2]
                        self.total_cost_std[a, top_k, d]=utilCostObj.total_cost_std
                        self.group_cost_std[a, :, top_k, d]=utilCostObj.group_cost_std
        self.delta_max=EOR_obj[1]


    def posteriorPredictiveAlg(self, onlineAlgo, alpha_prior, beta_prior):
        merits = np.full((len(self.rankingAlgos), self.timesteps, self.num_docs, self.draws), np.inf)
        self.running_thetas = np.full((len(self.rankingAlgos), self.timesteps, self.num_docs, self.num_docs, self.draws), np.inf)

        algo = onlineAlgo(self.num_docs, alpha_prior=alpha_prior, beta_prior=beta_prior,
                draws=self.draws, groupLen=self.groupLen)
        true_thetas=algo.true_thetas
        self.true_dist=self.getGroupDist(np.vectorize(Bernoulli)(true_thetas))
        
        if self.plot_w_true:
            # -1 for the last draw
            self.createAssymety(algo)
            posterior_merit_dist=self.getGroupDist(algo.get_thetaDist())
            exp = simpleOfflineMultipleGroups(num_groups=len(posterior_merit_dist[-1]), num_docs=self.num_docs, 
                    predfined_ls=posterior_merit_dist[-1], distType=self.distType, merits=algo.get_merits()[...,-1][None,:])
            exp.experiment(rankingAlgos=self.rankingAlgos, simulations=1000)
        # add asymmetry between majority and minority groups
        # self.createAssymety(algo)

        #for last draw
        self.visualize_prior(algo)
        self.visualize_thetas(true_thetas[:,-1])

        
        ranking_deterministic = np.ones((len(self.rankingAlgos), self.timesteps, self.num_docs, 1, self.num_docs, self.draws), dtype=int)

        for top_k in range(self.num_docs):    
            for a, rankingAlg in enumerate(self.rankingAlgos):
                algo.reset()
                self.createAssymety(algo)
                for i in range(self.timesteps):
                    merits[a,i,...] = algo.get_merits()
                    posterior_merit_dist=self.getGroupDist(algo.get_thetaDist())
                    
                    self.running_thetas[a, i,top_k,...]=algo.get_theta()
                    self.regrets[a, i, top_k, ...] = algo.get_regret(algo.get_theta())
                    self.theta_diff[a, i, top_k, ...] = algo.get_thetaDiff(algo.get_theta())
                    for d in range(self.draws):
                        ranker = rankingAlg(posterior_merit_dist[d], distType=self.distType)

                        ranker.rank(self.num_docs)

                        ranking_deterministic[a, i, top_k, :,:,d] = ranker.ranking
                    
                    
                    self.ranking[a]=ranking_deterministic[a,...]
                    # self.ranking[a] = parallelRanker(self.draws, rankingAlg, self.distType, self.num_docs)
                    # not sure how to update for stochastic policy
                    algo.update_params(self.ranking[a][i, top_k, 0,:top_k,:], merits[a,i,:,:])

        if self.blind:
            print(f"plots for timestep=0")
            self.computeCosts(timestep=0)
            self.visualize_Cost()
            self.visualize_EO()
            self.visualize_Util()
        
        self.computeCosts(timestep=self.timesteps-1)
        print(f"plots for last timestep")
        if self.plot:
            self.visualize_Cost()
            self.visualize_EO()
            self.visualize_Util()
            self.visualize_regret()
            self.visualize_thetaDiff(true_thetas[:,-1])
        
        if self.sum_costs:
            print(f"plots for sum and mean over all timesteps")
            self.visualize_SumCost()
            self.visualize_SumEO()
            self.visualize_SumUtil()


    def visualize_prior(self, algo, draw_num=-1):
        fig, ax=plt.subplots()
        prior_beta_distributions = [(i,j) for (i,j) in zip(algo.alpha[:,draw_num], algo.beta[:,draw_num])]
        for a,b in prior_beta_distributions[:self.groupLen[0]]:
            x = np.linspace(beta_dist.ppf(0.01, a, b),
                            beta_dist.ppf(0.99, a, b), 100)
            ax.plot(x, beta_dist.pdf(x, a, b),
                'r-', lw=1, label='Group A')
            
        for a,b in prior_beta_distributions[self.groupLen[0]:]:
            x = np.linspace(beta_dist.ppf(0.01, a, b),
                            beta_dist.ppf(0.99, a, b), 100)
            ax.plot(x, beta_dist.pdf(x, a, b),
                'b-', lw=1, label='Group B')

        custom_lines = [Line2D([0], [0], color='r', lw=1),
                        Line2D([0], [0], color='b', lw=1)]
        ax.legend(custom_lines, ['Group A', 'Group B'], fontsize=15)
        plt.title(f"Beta prior for different arms after creating assymetry for draw {draw_num}",fontsize=15)
        plt.show()
    
    def visualize_thetas(self, true_thetas):
        plt.scatter(np.arange(self.groupLen[0]), true_thetas[:self.groupLen[0]], label="majority group", c='r')
        plt.scatter(np.arange(self.groupLen[0], self.num_docs), true_thetas[self.groupLen[0]:], label="minority group", c='b')
        plt.legend(fontsize=15)
        plt.ylabel(r'true thetas $\theta^*$', fontsize=30)
        plt.xlabel("candidates", fontsize=15)
        plt.show()
        plt.close()

        
        sns.kdeplot(true_thetas[:self.groupLen[0]], c='r').set(xlim=(0,1))
        sns.kdeplot(true_thetas[self.groupLen[0]:], c='b').set(xlim=(0,1))
        plt.plot([], [], ' ', label=f"n_A={sum(true_thetas[:self.groupLen[0]]):.3f}, n_B={sum(true_thetas[self.groupLen[0]:]):.3f}")
        plt.plot([], [], ' ', label=f"var_A={np.var(true_thetas[:self.groupLen[0]]):.3f}, var_B={np.var(true_thetas[self.groupLen[0]:]):.3f}")
        plt.scatter([],[], c='r', label="A")
        plt.scatter([],[], c='b', label="B")
        plt.plot([], [], ' ', label=f"size_A={len(true_thetas[:self.groupLen[0]])}, size_B={len(true_thetas[self.groupLen[0]:])}")
        plt.xlabel(r'$p(r_i)$')
        plt.legend(fontsize=20)
        plt.title(r'Density plot for the true thetas $\theta^*$', fontsize=15)
        plt.show()
        plt.close()


    def visualize_Util(self):
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(5,5))
        for a, rankingAlg in enumerate(self.rankingAlgos):
            ax.plot(np.arange(self.num_docs)+1, np.mean(self.dcgUtil[a, :,:], axis=-1), label=f"{rankingAlg.name()}", c=self.colorMap[rankingAlg.name()], marker = '.',linewidth=1, markersize=1)
 
        plt.xlabel("Length of Ranking (k)", fontsize=20)
        plt.ylabel(r'DCG $U[\pi_k]$', fontsize=20)
        plt.ylim(-self.offset, np.max(self.dcgUtil) + self.offset)
        plt.xlim(1-self.offset, self.num_docs+self.offset)
        plt.tight_layout() 
        if self.saveFig is not None:
            plt.savefig(f"{osp.join(self.saveFig,'DCG_Util.pdf')}")
        plt.show()
        plt.close()
    
    def visualize_Cost(self):
        fig, ax = plt.subplots(figsize=(10,5), ncols=2)
        plt.rc('font', family='serif')
        for a, rankingAlg in enumerate(self.rankingAlgos):
            for g in range(self.groups):
                ax[0].plot(np.arange(self.num_docs)+1, np.mean(self.cost_groups[a, g, :,:], axis=-1), linestyle=self.lineMap[g], c=self.colorMap[rankingAlg.name()], linewidth=3)
            ax[1].plot(np.arange(self.num_docs)+1, np.mean(self.total_cost[a, :,:], axis=-1), linestyle='solid', c=self.colorMap[rankingAlg.name()], linewidth=1)    
            
        handles, labels = plt.gca().get_legend_handles_labels()
        
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
        fig.tight_layout() 
        if self.saveFig is not None:
            plt.savefig(f"{osp.join(self.saveFig,'Cost.pdf')}")
        plt.show()
        plt.close()

    def visualize_EO(self):
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(6,6))
        for a, rankingAlg in enumerate(self.rankingAlgos):
            ax.plot(np.arange(self.num_docs)+1, np.mean(self.EO_constraint[a, :,:], axis=-1), label=str(rankingAlg.name()), c=self.colorMap[rankingAlg.name()])
        
            
        if self.groups==2:
            plt.ylabel(r'$\bf{\delta(\sigma_k) = \frac{n(A|\sigma_k) }{n(A)}- \frac{n(B|\sigma_k)}{n(B)}}$', fontsize=20)
        else:
            plt.ylabel(r'$\bf{\delta(\sigma_k) = \max_{g} \left(\frac{n(g|\sigma_k) }{n(g)}\right)- \min_{g} \left(\frac{n(g|\sigma_k)}{n(g)}\right)}$', fontsize=20)
        plt.xlabel("Length of Ranking (k)", fontsize=20)
        
        if self.delta_max is not None:
            plt.hlines(y=self.delta_max, xmin=1, xmax=self.num_docs, color='black', linestyle='dashed')
            plt.hlines(y=-self.delta_max, xmin=1, xmax=self.num_docs, color='black', linestyle='dashed')
            plt.text(self.num_docs/2, self.delta_max+self.offset, r'$\delta_{max}$', c='black', fontsize=20)
            plt.text(self.num_docs/2, -self.delta_max-self.offset, r'$-\delta_{max}$', c='black', fontsize=20)
        
        # n_majority=sum([p.getMean() for p in self.predfined_ls[0]])
        # n_minority=sum([p.getMean() for p in self.predfined_ls[1]])
        # if self.verbose:
        #     relevance_handle = ax.plot([], [], ' ', label=f'$n_A={{{n_majority}}}, n_B={{{n_minority}}}$')
        #     size_handle = ax.plot([], [], ' ', label=f'$\vert A \vert={{{self.start_minority_idx}}}, \vert B \vert={{{self.num_docs-self.start_minority_idx}}}$')
        handles, labels = ax.get_legend_handles_labels()
        majority_lines= Line2D([], [], color='black', linestyle='dashed', linewidth=3, label=r'Majority (group A) Cost')
        minority_lines= Line2D([], [], color='black', linestyle='dotted',  linewidth=3, label=r'Minority (group B) Cost')
        total_lines = Line2D([], [], color='black', linestyle='solid',  linewidth=3, label=r'Total Cost')
        # handles.extend([majority_lines, minority_lines, total_lines])
        # legend=plt.legend(handles=handles, fontsize=12, loc='upper center', 
        # bbox_to_anchor=(0.5, 1.35),
        # ncol=6)
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
    
    def visualize_SumUtil(self):
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(5,5))
        for a, rankingAlg in enumerate(self.rankingAlgos):
            ax.plot(np.arange(self.num_docs)+1, np.mean(self.dcgUtil[a, :,:], axis=-1), label=f"{rankingAlg.name()}", c=self.colorMap[rankingAlg.name()], marker = '.',linewidth=1, markersize=1)
 
        plt.xlabel("Length of Ranking (k)", fontsize=20)
        plt.ylabel(r'DCG $U[\pi_k]$', fontsize=20)
        plt.ylim(-self.offset, np.max(self.dcgUtil) + self.offset)
        plt.xlim(1-self.offset, self.num_docs+self.offset)
        plt.tight_layout() 
        plt.show()
        plt.close()
    
    def visualize_SumCost(self):
        fig, ax = plt.subplots(figsize=(10,5), ncols=2)
        plt.rc('font', family='serif')
        start_index=int(self.num_docs/3)
        end_index=int(self.num_docs/3)
        for a, rankingAlg in enumerate(self.rankingAlgos):
            for g in range(self.groups):
                ax[0].plot(np.arange(self.num_docs)+1, np.mean(self.cost_groups[a, g, :,:], axis=-1), linestyle=self.lineMap[g], c=self.colorMap[rankingAlg.name()], linewidth=3)
            ax[1].plot(np.arange(self.num_docs)+1, np.mean(self.total_cost[a, :,:], axis=-1), linestyle='solid', c=self.colorMap[rankingAlg.name()], linewidth=1)    
        fig.supxlabel("Length of Ranking (k)", fontsize=20)
        fig.supylabel(f"Costs ", fontsize=20)
        for axis in ax.ravel():
            axis.set_ylim(-self.offset, 1 + self.offset)
            axis.set_xlim(1-self.offset, self.num_docs+self.offset)
        fig.tight_layout() 
        plt.show()
        plt.close()

    def visualize_SumEO(self):
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(6,6))
        for a, rankingAlg in enumerate(self.rankingAlgos):
            ax.plot(np.arange(self.num_docs)+1, np.mean(self.EO_constraint[a, :,:], axis=-1), label=str(rankingAlg.name()), c=self.colorMap[rankingAlg.name()], linewidth=3)
        
            
        if self.groups==2:
            plt.ylabel(r'$\bf{\delta(\sigma_k) = \frac{n(A|\sigma_k) }{n(A)}- \frac{n(B|\sigma_k)}{n(B)}}$', fontsize=20)
        else:
            plt.ylabel(r'$\bf{\delta(\sigma_k) = \max_{g} \left(\frac{n(g|\sigma_k) }{n(g)}\right)- \min_{g} \left(\frac{n(g|\sigma_k)}{n(g)}\right)}$', fontsize=20)
        plt.xlabel("Length of Ranking (k)", fontsize=20)
        
        if self.delta_max is not None:
            plt.hlines(y=self.delta_max, xmin=1, xmax=self.num_docs, color='black', linestyle='dashed')
            plt.hlines(y=-self.delta_max, xmin=1, xmax=self.num_docs, color='black', linestyle='dashed')
            plt.text(self.num_docs/2, self.delta_max+self.offset, r'$\delta_{max}$', c='black', fontsize=20)
            plt.text(self.num_docs/2, -self.delta_max-self.offset, r'$-\delta_{max}$', c='black', fontsize=20)
        
        plt.xlim(1-self.offset, self.num_docs+self.offset)
        plt.tight_layout()
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
    
    def visualize_regret(self):
        # k = random.randint(2, self.num_docs-1)
        k= 15
        for a, rankingAlg in enumerate(self.rankingAlgos):
            plt.scatter(np.arange(self.timesteps), np.mean(self.regrets[a,:,k,:], axis=-1), marker='x', label=f"{rankingAlg.name()}", c=self.colorMap[rankingAlg.name()])
            # plt.fill_between(np.arange(self.timesteps), np.mean(self.regrets[a,:,k], axis=0)-np.std(self.regrets[a,:,k], axis=0), np.mean(self.regrets[a,:,k], axis=0)+np.std(self.regrets[a,:,k], axis=0), alpha=0.2)
        plt.xlabel("timesteps", fontsize=15)
        plt.ylabel(r' $\sum_{i}^{n} (\theta_i^* - \theta_i^t)^2$', fontsize=20)
        plt.title(f"Learning thetas for k={k}", fontsize=15)
        plt.legend(fontsize=15)
        plt.show() 
        plt.close() 

        # visualize regret for last simulation
        # for a, rankingAlg in enumerate(self.rankingAlgos):
        #     plt.scatter(np.arange(self.timesteps), self.regrets[a,:,k], marker='x', label=f"{rankingAlg.name()}", c=self.colorMap[rankingAlg.name()])
            
        # plt.xlabel("timesteps", fontsize=15)
        # plt.ylabel(r' $\sum_{i}^{n} (\theta_i^* - \theta_i^t)^2$', fontsize=20)
        # plt.title(f"Learning thetas for k={k} for last simulation", fontsize=15)
        # plt.legend(fontsize=15)
        # plt.show() 
        # plt.close() 
    
    def visualize_thetaDiff(self, true_thetas, timestep=10):
        k=15
        for a, rankingAlg in enumerate(self.rankingAlgos):
            plt.scatter(true_thetas, self.theta_diff[a, timestep, k, :,-1], marker='x', label=f"{rankingAlg.name()}", c=self.colorMap[rankingAlg.name()])
        plt.xlabel(r'$\theta^*$', fontsize=15)
        plt.ylabel(r' $\theta_i^* - \theta_i^t$', fontsize=20)
        plt.title(f"difference in thetas as compared to true thetas at timestep: {timestep} for last simulation", fontsize=15)
        plt.legend(fontsize=15)
        plt.show() 
        plt.close()

    def getExpectedRelevance(self, dist):
        """
        get n_A and n_B
        """
        n_majority = sum([d.getMean() for d in dist[0]]) #n_A
        n_minority = sum([d.getMean() for d in dist[1]]) #n_B
        return n_majority, n_minority

    def experiment(self, rankingAlgos, onlineAlgos, simulations, alpha_prior=1, beta_prior=1, 
                    timesteps=10, figsize=(20,5), draws=50, blind=False, sum_costs=False, plot_w_true=False):
        """
        Args:
            timesteps: (int) how many steps for the algo to learn the bandit
            simulations: (int) number of epochs
        """
        names=[]
        self.timesteps=timesteps
        self.blind=blind
        self.sum_costs=sum_costs
        self.rankingAlgos=rankingAlgos
        self.draws=draws
        self.simulations=simulations
        self.plot_w_true=plot_w_true
        self.dcgUtil = np.zeros((len(self.rankingAlgos), self.num_docs, draws))
        self.cost_groups = np.zeros((len(self.rankingAlgos), self.groups, self.num_docs, draws))
        self.total_cost = np.zeros((len(self.rankingAlgos), self.num_docs, draws))
        self.EO_constraint = np.zeros((len(self.rankingAlgos), self.num_docs, draws))
        self.regrets = np.zeros((len(self.rankingAlgos), self.timesteps, self.num_docs, draws))
        self.theta_diff = np.zeros((len(self.rankingAlgos), self.timesteps, self.num_docs, self.num_docs, draws))
        
        self.ranking={}
        if self.verbose:
            self.EOR_std=np.zeros((len(self.rankingAlgos), self.num_docs, draws))
            self.total_cost_std=np.zeros((len(self.rankingAlgos), self.num_docs, draws))
            self.group_cost_std=np.zeros((len(self.rankingAlgos), self.groups, self.num_docs, draws))

        plt.rcParams["figure.figsize"] = figsize
        for onlineAlgo in onlineAlgos:
            self.posteriorPredictiveAlg(onlineAlgo, alpha_prior, beta_prior)

class BayesianSynthetic(BayesianSetup):
    def __init__(self, num_groups, num_docs, groupLen=None, predfined_ls=None, distType=None, online=False, 
    merits=None, plot=True, saveFig=None, offset=0.05, verbose=False) -> None:
        super().__init__(num_groups, num_docs, groupLen, predfined_ls, distType, online, 
    merits, plot, saveFig, offset, verbose)
    
    def posteriorPredictiveAlg(self, onlineAlgo, alpha_prior, beta_prior):
        merits = np.full((len(self.rankingAlgos), self.timesteps, self.num_docs, self.draws), np.inf)
        self.running_thetas = np.full((len(self.rankingAlgos), self.timesteps, self.num_docs, self.num_docs, self.draws), np.inf)

        algo = onlineAlgo(self.num_docs, alpha_prior=alpha_prior, beta_prior=beta_prior,
                draws=self.draws, groupLen=self.groupLen)
        true_thetas=algo.true_thetas
        self.true_dist=self.getGroupDist(np.vectorize(Bernoulli)(true_thetas))
        
        
        # -1 for the last draw
        self.createAssymety(algo,100,10)
        self.posterior_merit_dist=self.getGroupDist(algo.get_thetaDist())
        exp = simpleOfflineMultipleGroups(num_groups=len(self.posterior_merit_dist[-1]), num_docs=self.num_docs, verbose=True,
                predfined_ls=self.posterior_merit_dist[-1], distType=self.distType, merits=algo.get_merits()[...,-1][None,:])
        exp.experiment(rankingAlgos=self.rankingAlgos, simulations=1000)

        #for last draw
        self.visualize_prior(algo)
        self.visualize_thetas(true_thetas[:,-1])
        self.total_cost=exp.total_cost
        self.cost_groups=exp.cost_groups
        self.total_cost_std=exp.total_cost_std
        self.group_cost_std=exp.group_cost_std
    
