import numpy as np
import random
import heapq
import matplotlib.pyplot as plt
from matplotlib import lines
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from rankingFairness.src.distributions import Drichlet, Multinomial, Bernoulli, Beta
import os.path as osp

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def getHigher(arr1, arr2, str1, str2):
    """
    print the indices and values for both lists if one is higher than the other
 
    >>> getHigher(np.array([1., 2.3, 3.4, 5.4]), np.array([1.1, 2.3, 3.3, 5.2]), "1st", "2nd")
    k: [3 4], 1st values:[3.4 5.4], 2nd values:[3.3 5.2]
    """
    idxs=None
    if len(np.where(arr1>arr2))>0:
        idxs = np.where(arr1>arr2)[0]
        print(f"k: {idxs+1}, {str1} values:{np.array(arr1)[idxs]}, {str2} values:{np.array(arr2)[idxs]}")


class MaxPriorityQueue:

    def __init__(self) -> None:
            self.pq = []

    def add(self, v, idx):
        heapq.heappush(self.pq, (-v, idx))
    
    def pop_max(self):
        neg_v, idx = heapq.heappop(self.pq)
        return (-neg_v, idx)

    def peek_max(self):
        neg_v, idx = self.pq[0]
        return (-neg_v, idx)

    def __len__(self):
        return len(self.pq)

def getGroupNames(start_minority_idx, ranking):
    groupNames=[0 if r<start_minority_idx else 1 for r in ranking]
    return groupNames

def plotDrichletDist(alphas_majority=[np.array([1,10,2])], alphas_minority=[np.array([10,1,7])], size=1000, loc='best'):
    assert alphas_majority[0].shape==alphas_minority[0].shape
    fig,ax = plt.subplots(figsize=(20,10))
    for a_major in alphas_majority:
        dirich_samples=Drichlet(a_major).sample(num_samples=size)
        exp_score = [Multinomial(d).getMean() for d in dirich_samples]
        sns.kdeplot(exp_score, 
                     color = "lightpink",
                     alpha=0.1,
                     fill=True,
                     ax = ax)
        means_majority = Multinomial(Drichlet(a_major).getMean()).getMean()
        ax.vlines(means_majority, 0, 0.5,  linewidth=2.0, zorder=3, color='lightpink')
    
    for a_minor in alphas_minority:
        dirich_samples=Drichlet(a_minor).sample(num_samples=size)
        exp_score = [Multinomial(d).getMean() for d in dirich_samples]
        sns.kdeplot(exp_score, 
                     color = "teal",
                     alpha=0.1,
                     fill=True,
                     ax = ax)
        means_minority = Multinomial(Drichlet(a_minor).getMean()).getMean()
        ax.vlines(means_minority, 0, 0.5,  linewidth=2.0, zorder=3, color='black')

    ax.set_title("Posterior Distribution and expected merit", fontsize=20)
    ax.set_xlabel(r'Expected merit $\mathbb{E}[r_i]$', fontsize=20)
    
    A_dist = [Multinomial(Drichlet(p).getMean()) for p in alphas_majority]
    B_dist = [Multinomial(Drichlet(p).getMean()) for p in alphas_minority]
    n_majority=sum([p.getMean() for p in A_dist])
    n_minority=sum([p.getMean() for p in B_dist])
    ax.plot([], [], ' ', label=f"n_A={n_majority:.3f}, n_B={n_minority:.3f}")
    ax.plot([], [], ' ', label=f"size_A={len(A_dist)}, size_B={len(B_dist)}")
    handles, labels = ax.get_legend_handles_labels()
    vertical_line = lines.Line2D([], [], color='black', marker='|', linestyle='None',
                          markersize=10, markeredgewidth=3.5, label=r'Expected Merits')
    horizontal_line_majority = lines.Line2D([], [], color='lightpink', marker='_', linestyle='None',
                          markersize=10, markeredgewidth=3.5, label=r'majority posterior')
    horizontal_line_minority = lines.Line2D([], [], color='teal', marker='_', linestyle='None',
                          markersize=10, markeredgewidth=3.5, label=r'minority posterior')
    handles.extend([vertical_line, horizontal_line_majority, horizontal_line_minority])
    ax.legend(handles=handles, fontsize=20, loc=loc)
    # ax.set_xlim(1, len(alphas_majority[0])+1)
    ax.set_yticks([])
    ax.set_ylabel(r'Posterior $P(r_i|D)$', fontsize=20)
    plt.show()

def plotBetaDist(alphas_majority=[(90,20), (95,15), (100,10),(10,100)], 
                alphas_minority=[(5,3), (3,3), (3,4),(5,4)], size=1000, loc='upper center', 
                offset=0.05, colors=['teal','lightpink']):
    assert len(alphas_majority)==len(alphas_minority)
    fig,ax = plt.subplots(figsize=(7,3))
    for alpha,beta in alphas_majority:
        beta_samples=Beta(alpha,beta).sample(num_samples=size)
        exp_score = [Bernoulli(d).getMean() for d in beta_samples]
        sns.kdeplot(exp_score, 
                     color = "teal",
                     alpha=0.1,
                     fill=True,
                     ax = ax)
        means_majority = Bernoulli(Beta(alpha,beta).getMean()).getMean()
        ax.vlines(means_majority, 0, 1.4,  linewidth=4.0, zorder=3, color='teal')
    
    for alpha,beta in alphas_minority:
        beta_samples=Beta(alpha,beta).sample(num_samples=size)
        exp_score = [Bernoulli(d).getMean() for d in beta_samples]
        sns.kdeplot(exp_score, 
                     color = "lightpink",
                     alpha=0.1,
                     fill=True,
                     ax = ax)
        means_minority = Bernoulli(Beta(alpha,beta).getMean()).getMean()
        ax.vlines(means_minority, 0, 1.4,  linewidth=4.0, zorder=3, color='lightpink')

    
    ax.set_xlabel(r'$\bf{\theta_i}$', fontsize=20)
    
    A_dist = [Bernoulli(Beta(alpha,beta).getMean()) for (alpha,beta) in alphas_majority]
    B_dist = [Bernoulli(Beta(alpha,beta).getMean()) for (alpha,beta) in alphas_minority]
    n_majority=sum([p.getMean() for p in A_dist])
    n_minority=sum([p.getMean() for p in B_dist])
    # ax.plot([], [], ' ', label=r'$\bf{\sum_{i \in A} p_i = }$'+f"{n_majority:.1f},"+ r'$\bf{\sum_{j \in B} p_j =}$'+f"{n_minority:.1f}")
    # ax.plot([], [], ' ', label=r'$\bf{\vert A \vert = }$'+f"{len(A_dist)},"+ r'$\bf{\vert B \vert = }$'+f"{len(B_dist)}")
    handles, labels = ax.get_legend_handles_labels()
    vertical_line = ax.scatter([], [], color='black', marker='|', s=150,  linewidths=3, label=r'$\bf{\mathbb{P}(r_i|\mathbb{D})}$')
    horizontal_line_majority = ax.scatter([], [], color='teal', marker='.', s=100, label=r'group A')
    horizontal_line_minority = ax.scatter([], [], color='lightpink', marker='.', s=100, label=r'group B')
    handles.extend([vertical_line, horizontal_line_majority, horizontal_line_minority])
    plt.legend(handles=handles, ncol=3, markerscale=1., scatterpoints=1, fontsize=12, loc=loc, bbox_to_anchor=[0.25, 0.5, 0.5, 0.5])
    ax.set_xlim(0-offset, 1+offset)
    ax.set_yticks([])
    plt.tick_params(top = False, bottom = False)
    ax.set_ylabel(r'Posterior $\bf{\mathbb{P}(\theta_i|\mathbb{D})}$', fontsize=17)
    plt.tight_layout()
    plt.savefig(f"{osp.join('/share/thorsten/rr568/CostOptimal_FairRankings/plots','posterior.pdf')}")
    plt.show()
    plt.close()

class PlattCalibrator(BaseEstimator):
    """
    Boils down to applying a Logistic Regression.

    Parameters
    ----------
    log_odds : bool, default True
        Logistic Regression assumes a linear relationship between its input
        and the log-odds of the class probabilities. Converting the probability
        to log-odds scale typically improves performance.

    Attributes
    ----------
    coef_ : ndarray of shape (1,)
        Binary logistic regression's coefficient.

    intercept_ : ndarray of shape (1,)
        Binary logistic regression's intercept.
    """

    def __init__(self, log_odds: bool=True):
        self.log_odds = log_odds

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray):
        """
        Learns the logistic regression weights.

        Parameters
        ----------
        y_prob : 1d ndarray
            Raw probability/score of the positive class.

        y_true : 1d ndarray
            Binary true targets.

        Returns
        -------
        self
        """
        self.fit_predict(y_prob, y_true)
        return self

    @staticmethod
    def _convert_to_log_odds(y_prob: np.ndarray) -> np.ndarray:
        eps = 1e-12
        y_prob = np.clip(y_prob, eps, 1 - eps)
        y_prob = np.log(y_prob / (1 - y_prob))
        return y_prob

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Predicts the calibrated probability.

        Parameters
        ----------
        y_prob : 1d ndarray
            Raw probability/score of the positive class.

        Returns
        -------
        y_calibrated_prob : 1d ndarray
            Calibrated probability.
        """
        if self.log_odds:
            y_prob = self._convert_to_log_odds(y_prob)

        output = self._transform(y_prob)
        return output

    def _transform(self, y_prob: np.ndarray) -> np.ndarray:
        output = y_prob * self.coef_[0] + self.intercept_
        output = 1 / (1 + np.exp(-output))
        return output

    def fit_predict(self, y_prob: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Chain the .fit and .predict step together.

        Parameters
        ----------
        y_prob : 1d ndarray
            Raw probability/score of the positive class.

        y_true : 1d ndarray
            Binary true targets.

        Returns
        -------
        y_calibrated_prob : 1d ndarray
            Calibrated probability. 
        """
        if self.log_odds:
            y_prob = self._convert_to_log_odds(y_prob)

        # the class expects 2d ndarray as input features
        logistic = LogisticRegression(C=1e10, solver='lbfgs')
        logistic.fit(y_prob.reshape(-1, 1), y_true)
        self.coef_ = logistic.coef_[0]
        self.intercept_ = logistic.intercept_

        y_calibrated_prob = self._transform(y_prob)
        return y_calibrated_prob