class Statistic:
    """
    See: https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.W=0
        self.E=0
        self.S=0

    def add_sample(self, value, weight):
        self.W+=weight
        tmp_E=self.E
        self.E+=weight/self.W*(value-self.E)
        self.S+=weight*(value-tmp_E)*(value-self.E)

    def get_E(self):
        return self.E

    def get_Variance(self):
        return self.S/(self.W-1)

