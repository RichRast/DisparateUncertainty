import numpy as np
import random
import heapq
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
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
    
class PlattCalibrator(BaseEstimator):
    """
    https://github.com/ethen8181/machine-learning/blob/master/model_selection/prob_calibration/calibration_module/calibrator.py

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