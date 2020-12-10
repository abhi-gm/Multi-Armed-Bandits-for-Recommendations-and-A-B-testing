'''

Author - Abhishek Maheshwarappa and Jiaxin Tong

'''

import numpy as np


class ABTestReplayer(ReplaySimulator):
    '''
    A class to provide functionality for simulating the replayer method on an A/B test.
    '''
    
    def __init__(self, n_visits, n_test_visits, reward_history, item_col_name, visitor_col_name, reward_col_name, n_iterations=1):
        super(ABTestReplayer, self).__init__(n_visits, reward_history, item_col_name, visitor_col_name, reward_col_name, n_iterations)
        
        # TODO: validate that n_test_visits <= n_visits
    
        self.n_test_visits = n_test_visits
        
        self.is_testing = True
        self.best_item_id = None
        
    def reset(self):
        super(ABTestReplayer, self).reset()
        
        self.is_testing = True
        self.best_item_idx = None
    
    def select_item(self):
        if self.is_testing:
            return super(ABTestReplayer, self).select_item()
        else:
            return self.best_item_idx
            
    def record_result(self, visit, item_idx, reward):
        super(ABTestReplayer, self).record_result(visit, item_idx, reward)
        
        if (visit == self.n_test_visits - 1): # this was the last visit during the testing phase
            
            self.is_testing = False
            self.best_item_idx = np.argmax(self.n_item_rewards)