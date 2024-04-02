from Ethic_Ant import Ant
from concurrent.futures import ThreadPoolExecutor
import random as rng
import numpy as np
import logging
import math

# IMPLEMENTME: make a threaded version of initialize_ants
# IMPLEMENTME: make a threaded version of meet
# threading in the case of antclust is not a trivial problem.
# The similarity function is covered as an object and used by
# __similarity(). If the user definable similarity function is not
# thread save it will fail. Threads will as such not be easy to use
# within sophisticated similarity functions that utilize external
# librarys.
# However if store_computed_similaritys = False and the similarity
# object is thread save ant clust should be as well.
# Then the GIL will be the next point of slow down...
# IMPLEMENTME: inside constructor build a check that ensures
# len(datasetfatures) == len(similarity_function_per_feature)
# maybe use len(ant.genetics) == len(similarity_function_per_feature)?
# IMPLEMENTME:
# Improve logging
# https://docs.python.org/3.8/library/logging.html

# Idea for reassign:
# It would be also possible to make a probailistic method.
# this would be as folows:
# During the nest shrink, which is done before, it can easily be
# filtered out which ants belong to which nest.
# Then compare the ants that have no nest not to all other ants but
# instead to n ants taken from each nest.
# So lets say take randomly 50% int(nest_size*0.5) ants from each nest
# and compare the ant only to those. This reduces the load by 50%.
#


class AntClust:
    """implements the AntClust algorithm logic"""

    def __init__(self,
                 number_of_ants,
                 ruleset,
                 store_computed_similaritys=True,
                 alpha_ant_meeting_iterations=150,
                 betta_template_init_meetings=0.5,
                 nest_shrink_prop=0.2,
                 nest_removal_prop=0.3,
                 print_status=True,
                 threshold=None,
                 dropout=False,
                 dropout_prob = 0.3,
                 dynamic_template_adaptation=False,
                 visualization=False):
        """
        dataset: the data set as N dimensional array or list

        similarity_function_per_feature: array which holds the similarity
            classes for every feature inside the data set

        ruleset: rule class that implements the rule set

        store_computed_similaritys: will store every computed similarity
        (which is not subject to change) inside a dictionary to prevent
        recomputing. Increases memory load (needs a dictionary of size
        (N-1)*N/2, where N=numdataframes inside the data set)
        but saves computational cost

        alpha_ant_meeting_iterations: based on this the meetings for phase 0
        will be calculated by 0.5*alpha_ant_meeting_iterations*Num_data_points

        betta_template_init_meetings: used to calculate how many other ants
        a single ant will meet to initialize their template.
        betta_template_init_meetings
        is a percent value of alpha_ant_meeting_iterations.
        Template meetings = betta*alpha
        These meetings are meetings with randomly selected ants.

        nest_shrink_prop: this value will be used for the probabilistic nest
        fitness calculation in nest_shrink.

        nest_removal_prop: the value is a floating point percent value giving
        the fitness threshold for deleting a nest with a smaller fitness.
        """
        # variables
        self.print_status = print_status
        self.rules = ruleset
        self.threshold = threshold
        self.num_ants = number_of_ants
        # save ant objects
        self.ants = []
        # stores the computed labels for each data vector
        self.labels_ = []

        # AntClust parameters
        # droput
        self.dropout=dropout
        self.dropout_prob = dropout_prob
        # alpha used to calculate the meetings per ant in phase 1
        # and template initialization meetings
        self.alpha_ant_meeting_iterations = alpha_ant_meeting_iterations

        # calc template init meetings
        self.template_init_meetings = int(betta_template_init_meetings *
                                          alpha_ant_meeting_iterations)

        # used to calculate the nest deletion property in phase 2
        self.nest_shrink_prop = nest_shrink_prop
        self.nest_removal_prop = nest_removal_prop

        # store computed similarity's
        self.store_computed_similaritys = store_computed_similaritys
        self.saved_similaritys = {}

        # allow adaptation to ant's template over age
        self.dynamic_template_adaptation = dynamic_template_adaptation

        # visualization
        self.visualization = visualization
        self.rule_applied = []
        self.label_evolution = []

    def set_log_level(self, loglevel):
        """
        sets the log level.
        loglevel: string ('DEBUG, INFO', 'WARNING', 'ERROR', 'CRITICAL')
        """
        LOG_LEVELS = {
            'CRITICAL': logging.CRITICAL,
            'ERROR': logging.ERROR,
            'WARNING': logging.WARNING,
            'INFO': logging.INFO,
            'DEBUG': logging.DEBUG
        }

        logging.basicConfig(level=LOG_LEVELS[loglevel])

    def __initialize_ants(self,action):
        """
        Initialize_ants with their data frame and return a list of the ants
        """
        # initialize ants
        ant_list = []
        for i in range(self.num_ants):
            ant_list.append(Ant(action, i))
        return ant_list

    def __acceptance(self, ant_i, ant_j):
        """Evaluates if ants accept each other or not"""
        return ant_i.get_label() == ant_j.get_label()

    def get_acceptance(self, ant_i, ant_j):
        """
        Returns true if the ants accept each other false if not.
        """
        return self.__acceptance(ant_i, ant_j)

    def __meet(self):
        """let ants meet"""
        number_of_ants = len(self.ants)
        iterations_left = self.meeting_iterations
        while iterations_left:
            # print meetings?
            if self.print_status:
                if iterations_left % (int(self.meeting_iterations) * 0.1) == 0:
                    print(f'Meeting {iterations_left} / {self.meeting_iterations}')
            iterations_left = iterations_left - 1
            ant0_index = rng.randint(0, number_of_ants - 1)
            ant1_index = rng.randint(0, number_of_ants - 1)
            while ant0_index == ant1_index:
                ant1_index = rng.randint(0, number_of_ants - 1)
            """
            --------DROPOUT--------
            If dropout is true, reset random ants based on probability
            The idea is to simulate the same behaviour of dropout during training a ML model
            """ 
            time_to_dropout = (self.meeting_iterations - iterations_left) > math.ceil(self.meeting_iterations*0.5)
            if self.dropout and time_to_dropout:
                probability = rng.random()
                if probability <= self.dropout_prob:
                    self.dropout = False # stop dropout after happening once
                    lower_bound = math.ceil(number_of_ants*0.3)
                    upper_bound = math.ceil(number_of_ants*0.5)
                    list_length = rng.randint(lower_bound, upper_bound)
                    ants_index_reset  = [rng.randint(0, number_of_ants - 1) for _ in range(list_length)]
                    for index in ants_index_reset:
                        #print(f"Ant {index}: label ({self.ants[index].get_label()}) m ({self.ants[index].get_m()}) m+ ({self.ants[index].get_m_p()})")
                        self.ants[index].set_label(-1)
                        self.ants[index].set_m(0)
                        self.ants[index].set_m_p(0)
                        #print(f"Ant {index}: label ({self.ants[index].get_label()}) m ({self.ants[index].get_m()}) m+ ({self.ants[index].get_m_p()})")
                    if self.print_status:
                        print("--------DROPOUT--------")
                        print(f"Ants reseted: {ants_index_reset}")
                        print("-----------------------")

            # apply rules to ants
            ant_i = self.ants[ant0_index]
            ant_j = self.ants[ant1_index]
            rule = self.rules.apply_rules(ant_i, ant_j, self)
            if self.visualization:
                self.rule_applied.append(rule)
                self.__cluster_label_assignment()
                self.label_evolution.append(self.get_clusters())


    def __cluster_label_assignment(self):
        """makes an array holding the label, by index, for each data vector"""
        # make cluster array
        self.labels_ = np.zeros(len(self.ants), dtype=int)
        for i in range(len(self.ants)):
            self.labels_[i] = self.ants[i].label

    def get_clusters(self):
        """Returns the label, by index, for each data vector"""
        return self.labels_

    def __find_clusters(self):
        """will run AntClus algorithm on the given dataset"""
        # Steps:
        # simulate/meet ants with iterations=n
        # delete nests i.e. call nest_shrink()
        # re-assign ants with no nest i.e. call reassign()

        # meetings
        if self.print_status:
            print('AntClust: phase 1 of 3 -> meeting ants')
        self.__meet()

        # save clusters to a index readable array
        self.__cluster_label_assignment()

    def fit(self,X):
        """Find Clusters in data set X"""
        
        # initialize all the ants with the given data set
        self.ants = self.__initialize_ants(X)

        # number of random meetings per ant for phase 1
        self.meeting_iterations = int(0.5 * self.alpha_ant_meeting_iterations *
                                      len(self.ants))
        # run phases
        self.__find_clusters()
        if self.visualization:
            return self.label_evolution, self.rule_applied
