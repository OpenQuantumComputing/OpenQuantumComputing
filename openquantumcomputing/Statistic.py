import numpy as np

class Statistic:
    """
    See: https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf
    """

    def __init__(self, alpha=1):
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.W=0
        self.E=0
        self.S=0
        self.all_values=np.array([])
        self.all_weights=np.array([])

    def add_sample(self, value, weight):
        self.W+=weight
        tmp_E=self.E
        self.E+=weight/self.W*(value-self.E)
        self.S+=weight*(value-tmp_E)*(value-self.E)
        if self.alpha < 1:
            idx = np.searchsorted(self.all_values, value)
            self.all_values = np.insert(self.all_values, idx, value)
            self.all_weights = np.insert(self.all_weights, idx, weight)

    def get_E(self):
        return self.E

    def get_Variance(self):
        return self.S/(self.W-1)

    def get_CVaR(self):
        if self.alpha < 1:
            # Q: What is the meaning of the weigth?
            alphaK = int(np.round(self.alpha*len(self.all_values)))
            #return np.sum(self.all_values[:alphaK]*self.all_weights[:alphaK])/(alphaK*np.sum(self.all_weights[:alphaK]))
            return np.sum(self.all_values[-alphaK:])/alphaK
        else:
            return self.get_E()
