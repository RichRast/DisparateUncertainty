import numpy as np
from scipy.stats import beta as beta_dist 
from rankingFairness.src.tradeoffMultipleGroups import UtilityCost
import math

class Bernoulli():
    def __init__(self, p) -> None:
        super().__init__()
        assert p>=0.0
        assert p<=1.0
        self.p =p

    def sample(self, num_samples):
        return np.random.binomial(1, self.p, num_samples)
    
    def identity(x):
        return x

    def getMean(self, funcApply=identity):
        return funcApply(self.p)

    def update(self):
        raise NotImplementedError

class Beta():
    def __init__(self, alpha=1, beta=1) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def sample(self, num_samples):
        return np.random.beta(self.alpha, self.beta, num_samples)

    def getMean(self):
        return self.alpha/(self.alpha+self.beta)
        
    def update(self, s, f):
        self.alpha +=s
        self.beta +=f

class Rewards():
    def __init__(self, reward_dist, prior_dist) -> None:
        super().__init__()
        """
        Reward Distribution is the true merit distribution
        Prior Distribution is the estimated merit distrbution
        """
        self.reward_dist = reward_dist
        self.prior_dist = prior_dist

    def getRewards(self, num_samples):
        zeroOneRewards = self.reward_dist.sample(num_samples)
        return sum(zeroOneRewards), len(zeroOneRewards)-sum(zeroOneRewards)

    def getPosterior(self, s, f):
        self.prior_dist.update(s, f)

class BetaBernoulli():
    def __init__(self, dist) ->None:
        super().__init__()
        """
        Class for Beta Bernoulli Conjugate Model
        """
        self.dist=dist

    def sample(self):
        probs_all=[]
        for i in range(len(self.dist)):  
            probs_all.extend([d.getMean() for d in self.dist[i]])
        return np.random.binomial(1, probs_all)
    

class DrichletMultinomial():
    def __init__(self, dist) ->None:
        super().__init__()
        """
        Class for Drichlet Multinomial Conjugate Model
        """
        self.dist=dist

    def sample(self):
        probs_all = [d for d in self.dist[0]]
        probs_all.extend([d for d in self.dist[1]])
        merits=[p.sample(1) for p in probs_all]
        return merits


class Multinomial():
    def __init__(self, pvals) -> None:
        super().__init__()
        assert np.all(pvals)>=0.0
        assert np.all(pvals)<=1.0
        self.pvals =pvals
        assert self.pvals.ndim==1
        self.cat=np.arange(1,self.pvals.shape[0]+1)[:,None]

    def square(x):
        return np.square(x)/np.max(x)**2
    
    def cube(x):
        return np.power(x,3)/math.pow(np.max(x),3)
    
    def exponential(x):
        return np.exp(x)/math.exp(np.max(x))
    
    def identity(x):
        return x
    
    def sample(self, num_samples=1, funcApply=identity):
        arr_cat= np.random.multinomial(1, self.pvals, num_samples) #num_samples x cat
        return funcApply(np.where(arr_cat)[1]+1)
    
    def getMean(self, funcApply=identity):
        assert self.pvals.shape[0]==self.cat.shape[0]
        return (self.pvals.T@funcApply(self.cat)).item()

    def update(self):
        raise NotImplementedError

class Drichlet():
    def __init__(self, alpha=np.array([0,0,0,0,0])) -> None:
        super().__init__()
        assert isinstance(alpha, np.ndarray)
        self.alpha = np.array([1]*len(alpha)) + alpha

    def sample(self, num_samples=1):
        return np.random.dirichlet(self.alpha, num_samples)

    def getMean(self):
        return self.alpha/self.alpha.sum()
        
    def update(self, rewards):
        assert self.alpha.shape[0]==rewards.shape[0]
        self.alpha+=rewards
