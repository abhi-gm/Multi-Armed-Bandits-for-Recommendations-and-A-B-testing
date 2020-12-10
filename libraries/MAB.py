'''

Author - Abhishek Maheshwarappa and Jiaxin Tong

'''

import numpy as np


class ThompsonSamplingReplayer(ReplaySimulator):
    '''
    A class to provide functionality for simulating the replayer method on a Thompson Sampling bandit algorithm.
    '''

    def reset(self):
        self.alphas = np.ones(self.n_items)
        self.betas = np.ones(self.n_items)

    def select_item(self):
    
        samples = [np.random.beta(a,b) for a,b in zip(self.alphas, self.betas)]
        
        return np.argmax(samples)

    def record_result(self, visit, item_idx, reward):
        
        ## update value estimate
        if reward == 1:
            self.alphas[item_idx] += 1
        else:
            self.betas[item_idx] += 1