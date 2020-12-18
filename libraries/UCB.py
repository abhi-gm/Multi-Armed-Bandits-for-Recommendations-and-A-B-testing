'''

Author - Abhishek Maheshwarappa and Jiaxin Tong

'''

import numpy as np
from tqdm import tqdm

class UCBSamplingReplayer():
    '''
    A class to provide functionality for simulating the replayer method on a Thompson Sampling bandit algorithm

    '''

    def __init__(self,ucb_c, n_visits, reward_history, item_col_name, visitor_col_name, reward_col_name, n_iterations=1):
        
        self.reward_history = reward_history
        self.item_col_name = item_col_name
        self.visitor_col_name = visitor_col_name
        self.reward_col_name = reward_col_name
        self.ucb_c = ucb_c

        # number of runs to average over
        self.n_iterations = n_iterations
    


        # number of visits to replay/simulate
        self.n_visits = n_visits

        # items under test
        self.items = self.reward_history[self.item_col_name].unique()
        self.n_items = len(self.items)
        
        # visitors in the historical reward_history (e.g., from ratings df)
        self.visitors = self.reward_history[self.visitor_col_name].unique()
        self.n_visitors = len(self.visitors)
        
        self.is_testing = True
        self.best_item_id = None

    def reset(self):
        self.Q = np.zeros(self.n_items) # q-value of actions
        self.N = np.zeros(self.n_items) + 0.0001 # action count
        self.timestep = 1


    def select_item(self):

        ln_timestep = np.log(np.full(self.n_items, self.timestep))
        confidence = self.ucb_c * np.sqrt(ln_timestep/self.N)

        action = np.argmax(self.Q + confidence)
        self.timestep += 1
        
        return action

    def record_result(self, visit, item_idx, reward):
        
        # update value estimate
        self.N[item_idx] += 1 # increment action count
        self.Q[item_idx] += 1/self.N[item_idx] * (reward - self.Q[item_idx]) # inc. update rule


    def simulator(self):

        results = []

        for iteration in tqdm(range(0, self.n_iterations)):
        
            self.reset()
            
            total_rewards = 0
            fraction_relevant = np.zeros(self.n_visits)

            for visit in range(0, self.n_visits):
            
                found_match = False
                while not found_match:
                
                    # choose a random visitor
                    visitor_idx = np.random.randint(self.n_visitors)
                    visitor_id = self.visitors[visitor_idx]

                    # select an item to offer the visitor
                    item_idx = self.select_item()
                    item_id = self.items[item_idx]
                    
                    # if this interaction exists in the history, count it
                    reward = self.reward_history.query(
                        '{} == @item_id and {} == @visitor_id'.format(self.item_col_name, self.visitor_col_name))[self.reward_col_name]
                    
                    found_match = reward.shape[0] > 0
                
                reward_value = reward.iloc[0]
                
                self.record_result(visit, item_idx, reward_value)
                
                # record metrics
                total_rewards += reward_value
                fraction_relevant[visit] = total_rewards * 1. / (visit + 1)
                
                result = {}
                result['iteration'] = iteration
                result['visit'] = visit
                result['item_id'] = item_id
                result['visitor_id'] = visitor_id
                result['reward'] = reward_value
                result['total_reward'] = total_rewards
                result['fraction_relevant'] = total_rewards * 1. / (visit + 1)
                
                results.append(result)
        
        return results