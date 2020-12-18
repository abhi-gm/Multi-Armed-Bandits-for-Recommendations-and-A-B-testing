'''
Author - Abhishek Maheshwarappa and Jiaxin Tong

'''

import os
import sys
import time
import json
import datetime
import logging
import numpy as np
import pandas as pd

from libraries.ABTestSimulator import ABTestReplayer
from libraries.MAB import ThompsonSamplingReplayer
from libraries.EpsilonGreedy import EpsilonGreedyReplayer
from libraries.RealtimeUserSimulator import ReplaySimulator
from libraries.UCB import UCBSamplingReplayer


class Main_class:
    """
    docstring
    """

    def __init__(self):
        # getting the current system time
        self.current_time = datetime.datetime.now().strftime("%Y-%m-%d__%H.%M")

        # run folder which will be unique always
        run_folder = str(datetime.datetime.now())
        # temprary folder location to export the results
        temp_folder = "./output/"
        # target folder to export all the result
        self.target_dir = temp_folder + '/' + run_folder
        # checking if the temp folder exists. Create one if not.
        check_folder = os.path.isdir(self.target_dir)
        if not check_folder:
            os.makedirs(self.target_dir)
            print("created folder : ", self.target_dir)


        # latency dictionary to hold execution time
        self.latency = dict()

        # removing any existing log files if present
        if os.path.exists( self.target_dir + '/main.log'):
            os.remove(self.target_dir+ '/main.log')

        # get custom logger
        self.logger = self.get_loggers(self.target_dir)
        

    @staticmethod
    def get_loggers(temp_path):
        # name the logger as HPC-AI skunkworks
        logger = logging.getLogger("HPC-AI skunkworks")
        logger.setLevel(logging.INFO)
        # file where the custom logs needs to be handled
        f_hand = logging.FileHandler(temp_path + '/.log')
        f_hand.setLevel(logging.INFO)  # level to set for logging the errors
        f_format = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s',
                                     datefmt='%d-%b-%y %H:%M:%S')
        # format in which the logs needs to be written
        f_hand.setFormatter(f_format)  # setting the format of the logs
        # setting the logging handler with the above formatter specification
        logger.addHandler(f_hand)

        stdout_handler = logging.StreamHandler()
        stdout_handler.setFormatter(f_format)
        logger.addHandler(stdout_handler)

        return logger

    def main(self):
        print("It started")
        self.logger.info(
            "Multi-Armed-Bandits-for-Recommendations-and-A-B-testing !!!")
        self.logger.info("Current time: " + str(self.current_time))

        self.logger.info("Reading the data..!!")
        start = time.time()
        header_list = ["User_ID", "Product_ID", "Rating", "Time_Stamp"]
        rating_df = pd.read_csv(
            'data/ratings_Electronics.csv', names=header_list)
        self.latency["Data_reading -"] = time.time() - start
        self.logger.info("Read the data Successfully ..!!!")

        reward_threshold = 4
        rating_df['reward'] = rating_df.eval(
            'Rating > @reward_threshold').astype(int)

        n_visits = 500
        n_iterations = 20
        n_test_visits = 100

        reward_history = rating_df[:1000]
        item_col_name = 'Product_ID'
        visitor_col_name = 'User_ID'
        reward_col_name = 'reward'

        #################### A/B testing ###############

        self.logger.info("A/B Test Simulations...starts...!!!")

        start = time.time()
        ab_results = ABTestReplayer(n_visits, n_test_visits, reward_history,
                                    item_col_name, visitor_col_name, reward_col_name,
                                    n_iterations=n_iterations).simulator()

        ab_results_df = pd.DataFrame(ab_results)
        self.latency["A/B testing -"] = time.time() - start
        self.logger.info("A/B testing completed Successfully..!!")

        ab_results_df.to_csv(self.target_dir + '/ab_results_df.csv')

        self.logger.info("Saving the A/B test results saved Successfully..!!")

        ################# Epsilon - Greedy Simulations ##############

        self.logger.info("Epsilon - Greedy Simulations...starts...!!!")

        start = time.time()
        epsilon = 0.05
        epsilon_results = EpsilonGreedyReplayer(epsilon, n_visits, reward_history,
                                                item_col_name, visitor_col_name, reward_col_name,
                                                n_iterations=n_iterations).simulator()

        epsilon_results_df = pd.DataFrame(epsilon_results)
        self.latency["Epsilon - Greedy Simulations  -"] = time.time() - start
        self.logger.info(
            "Epsilon - Greedy Simulations completed Successfully..!!")

        epsilon_results_df.to_csv(self.target_dir +'/epsilon_results_df.csv')

        self.logger.info(
            "Epsilon - Greedy Simulations results saved Successfully..!!")

        ################### Thompson Sampling Simulations #######################

        self.logger.info("Thompson Sampling Simulations...starts...!!!")

        start = time.time()

        thompson_results = ThompsonSamplingReplayer(n_visits, reward_history,
                                                    item_col_name, visitor_col_name, reward_col_name,
                                                    n_iterations=n_iterations).simulator()

        thompson_results_df = pd.DataFrame(thompson_results)
        self.latency["Thompson Sampling Simulations  -"] = time.time() - start
        self.logger.info(
            "Thompson Sampling Simulations completed Successfully..!!")

        thompson_results_df.to_csv(self.target_dir +'/thompson_results_df.csv')

        self.logger.info(
            "Thompson Sampling Simulations results saved Successfully..!!")

        ####################  Upper Confidence Bounds #########################

        self.logger.info("Upper Confidence Bounds Simulations...starts...!!!")

        start = time.time()

        ucb = 2

        ucb_results = UCBSamplingReplayer(ucb, n_visits, reward_history,
                                          item_col_name, visitor_col_name, reward_col_name,
                                          n_iterations=n_iterations).simulator()

        ucb_results_df = pd.DataFrame(ucb_results)
        self.latency["Upper Confidence Bounds Simulations  -"] = time.time() - \
            start
        self.logger.info(
            "Upper Confidence Bounds Simulations completed Successfully..!!")

        ucb_results_df.to_csv(self.target_dir +'/ucb_results_df.csv')

        self.logger.info(
            "Upper Confidence Bounds Simulations results saved Successfully..!!")

        self.logger.info('Exporting the latency')
        file_name = self.target_dir +'/latency_stats.json'
        self.export_to_json(self.latency, file_name)

    def export_to_json(self, dictionary, file_name):
        try:
            start = time.time()
            json_data = json.dumps(dictionary, indent=4)
            file = open(file_name, 'w')
            print(json_data, file=file)
            # updating into json
            file.close()
            stop = time.time()
            self.latency['export_to_json_'] = stop - start
            self.logger.info('Data exported to JSON successfully!')
        except Exception as e:
            self.logger.exception(e)
            sys.exit(1)


if __name__ == "__main__":
    main = Main_class()
    main.main()
