import numpy as np

class Space(object):

    def __init__(self, shape):
        self.shape = shape
    
    def sample(self):
        raise NotImplementedError


class BuyHoldSellSpace(Space):
    def __init__(self):
        super().__init__(2)
    
    def sample(self,seed = None):
        if seed:
            np.random.seed(seed)
        action = np.zeros(self.shape)
        action[0] = np.random.randint(low=0,high=3)
        action[1] = np.random.randint(low=0,high=100)
        return action